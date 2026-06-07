from __future__ import annotations

from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

from .layout import crop_by_box
from .text_utils import (
    correct_with_lexicon,
    extract_after_label,
    flatten_region_lines,
    normalize_text,
    normalize_time_string,
    search_time_after_label,
)


def _crop(image: np.ndarray, box: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return image[0:0, 0:0].copy()
    return image[y1:y2, x1:x2].copy()


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return img
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _enhance_region_for_ocr(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    if img is None or img.size == 0:
        return img
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.35, blur, -0.35, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def _text_score(text: str, prefer_keywords: Sequence[str] | None = None) -> int:
    text = normalize_text(text)
    if not text:
        return 0
    score = len(text)
    score += 3 * len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    score += 2 * len([c for c in text if c.isdigit()])
    score += 2 * text.count(':')
    score += 2 * text.count('：')
    if prefer_keywords:
        for kw in prefer_keywords:
            if kw and kw in text:
                score += 10
    return score


def _best_text(candidates: List[str], prefer_keywords: Sequence[str] | None = None) -> str:
    candidates = [normalize_text(c) for c in candidates if normalize_text(c)]
    if not candidates:
        return ''
    candidates = sorted(set(candidates), key=lambda s: _text_score(s, prefer_keywords), reverse=True)
    return candidates[0]


def _ocr_box(
    engine: Any,
    image: np.ndarray,
    box: Sequence[int],
    preprocess: bool = True,
    prefer_keywords: Sequence[str] | None = None,
) -> str:
    crop = _crop(image, box)
    if crop.size == 0:
        return ''

    candidates: List[str] = []

    try:
        candidates.append(normalize_text(engine.ocr_region_text(crop, preprocess=preprocess)))
    except Exception:
        pass

    try:
        candidates.append(normalize_text(engine.ocr_region_text(_ensure_bgr(crop), preprocess=False)))
    except Exception:
        pass

    try:
        candidates.append(normalize_text(engine.ocr_region_text(_enhance_region_for_ocr(crop, 2.0), preprocess=False)))
    except Exception:
        pass

    try:
        candidates.append(normalize_text(engine.ocr_region_text(_enhance_region_for_ocr(crop, 3.0), preprocess=False)))
    except Exception:
        pass

    return _best_text(candidates, prefer_keywords=prefer_keywords)


def _ocr_lines_box(engine: Any, image: np.ndarray, box: Sequence[int], preprocess: bool = True) -> List[Dict[str, Any]]:
    crop = _crop(image, box)
    if crop.size == 0:
        return []

    variants = [
        (crop, preprocess),
        (_ensure_bgr(crop), False),
        (_enhance_region_for_ocr(crop, 2.0), False),
    ]

    best_lines: List[Dict[str, Any]] = []
    best_score = -1

    for variant, pp in variants:
        try:
            lines = [line.to_dict() for line in engine.ocr_region(variant, preprocess=pp)]
            text = ' '.join(flatten_region_lines(lines))
            score = _text_score(text)
            if score > best_score:
                best_score = score
                best_lines = lines
        except Exception:
            continue

    return best_lines


def _apply_lexicon(text: str, lexicon: Dict[str, List[str]], field: str) -> str:
    text = normalize_text(text)
    if not text:
        return ''
    return correct_with_lexicon(text, lexicon, field) or text


def _clean_label_residue(text: str, labels: Sequence[str]) -> str:
    text = normalize_text(text)
    for lb in labels:
        text = text.replace(lb, ' ')
    text = text.replace(':', ' ').replace('：', ' ')
    return ' '.join(text.split()).strip()


def _normalize_time_like(text: str) -> str:
    text = normalize_text(text).replace('：', ':')
    if text.isdigit() and len(text) == 4:
        return text[:2] + ':' + text[2:]
    if text.isdigit() and len(text) == 3:
        return '0' + text[0] + ':' + text[1:]
    return normalize_time_string(text)


def _extract_time_candidates(text: str) -> List[str]:
    import re

    text = normalize_text(text).replace('：', ':')
    matches = re.findall(r'([0-2]?\d:\d{2}|\d{3,4})', text)
    out: List[str] = []
    for m in matches:
        t = _normalize_time_like(m)
        if t and t not in out:
            out.append(t)
    return out


def _contains_bad_symbol(text: str) -> bool:
    text = normalize_text(text)
    bad = ['#', '$', '}', '{', '[', ']', '<', '>', '=', '*', '@']
    return any(x in text for x in bad)


def _is_valid_name_like(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return False
    if _contains_bad_symbol(text):
        return False
    if len(text) == 1:
        return False
    useful = sum(1 for c in text if ('\u4e00' <= c <= '\u9fff') or c.isalpha() or c.isdigit())
    return useful >= 2


def _is_header_like_training_row(row_result: Dict[str, str]) -> bool:
    values = [normalize_text(v) for v in row_result.values() if normalize_text(v)]
    if not values:
        return True
    joined = ' '.join(values)
    header_words = ['序号', '参训机型', '数量', '架次', '核算架次', '时间']
    hit = sum(1 for w in header_words if w in joined)
    return hit >= 2


def _clean_training_cell(col_name: str, text: str, lexicon: Dict[str, List[str]]) -> str:
    text = normalize_text(text)
    if not text:
        return ''

    if text == col_name:
        return ''

    if col_name == '参训机型':
        fixed = correct_with_lexicon(text, lexicon, 'aircraft_types')
        return fixed or ''

    if col_name in ['数量', '架次', '核算架次']:
        digits = ''.join(ch for ch in text if ch.isdigit())
        return digits

    if col_name == '时间':
        t = _extract_time_candidates(text)
        return t[0] if t else ''

    if col_name == '序号':
        digits = ''.join(ch for ch in text if ch.isdigit())
        return digits

    return ''


def _extract_title_candidate_from_lines(lines: List[Dict[str, Any]]) -> str:
    candidates: List[str] = []
    for item in lines:
        if isinstance(item, dict):
            text = normalize_text(str(item.get('text', '')))
        else:
            text = normalize_text(str(item))
        if not text:
            continue
        if '计划' in text and not any(k in text for k in ['天亮时刻', '天黑时刻', '日出时刻', '日没时刻', '月出时刻', '月没时刻', '批准']):
            candidates.append(text)

    if not candidates:
        return ''

    candidates = sorted(candidates, key=lambda s: len(s), reverse=True)
    return candidates[0]


def extract_title_fields(aligned: np.ndarray, config: Dict[str, Any], engine: Any, lexicon: Dict[str, List[str]]) -> Dict[str, Any]:
    semantic = config['semantic']['title_fields']
    title_box = config['regions']['title']
    title_img = crop_by_box(aligned, tuple(title_box))
    title_lines = _ocr_lines_box(
        engine,
        title_img,
        (0, 0, title_img.shape[1], title_img.shape[0]),
        preprocess=True,
    ) if title_img.size > 0 else []

    full_text = ' '.join(flatten_region_lines(title_lines))

    confidentiality_raw = _ocr_box(
        engine, aligned, semantic['confidentiality'],
        preprocess=True,
        prefer_keywords=lexicon.get('labels', ['内部', '秘密', '机密'])
    )
    title_raw = _ocr_box(
        engine, aligned, semantic['title_text'],
        preprocess=True,
        prefer_keywords=['计划']
    )
    astro_raw = _ocr_box(
        engine, aligned, semantic['astronomical_times'],
        preprocess=True,
        prefer_keywords=['天亮时刻', '天黑时刻', '日出时刻', '日没时刻', '月出时刻', '月没时刻']
    )
    approved_raw = _ocr_box(
        engine, aligned, semantic['approved'],
        preprocess=True,
        prefer_keywords=['批准']
    )

    confidentiality = _clean_label_residue(confidentiality_raw, ['批准', '计划', '日期'])
    confidentiality = _apply_lexicon(confidentiality, lexicon, 'labels')
    if confidentiality in ['批准', '计划', '日期', '']:
        for cand in lexicon.get('labels', ['内部', '秘密', '机密']):
            if cand in full_text:
                confidentiality = cand
                break

    title_text = normalize_text(title_raw)
    title_from_lines = _extract_title_candidate_from_lines(title_lines)
    if len(title_from_lines) > len(title_text):
        title_text = title_from_lines

    title_text = _clean_label_residue(
        title_text,
        ['批准', '天亮时刻', '天黑时刻', '日出时刻', '日没时刻', '月出时刻', '月没时刻']
    )

    date_value = ''

    approved_name = extract_after_label(approved_raw, '批准')
    if not approved_name:
        approved_name = _clean_label_residue(approved_raw, ['批准'])
    approved_name = correct_with_lexicon(approved_name, lexicon, 'names') or approved_name
    if not _is_valid_name_like(approved_name):
        approved_name = ''

    astro_source = astro_raw or full_text
    astronomical_times = {
        '天亮时刻': _normalize_time_like(search_time_after_label(astro_source, '天亮时刻')),
        '天黑时刻': _normalize_time_like(search_time_after_label(astro_source, '天黑时刻')),
        '日出时刻': _normalize_time_like(search_time_after_label(astro_source, '日出时刻')),
        '日没时刻': _normalize_time_like(search_time_after_label(astro_source, '日没时刻')),
        '月出时刻': _normalize_time_like(search_time_after_label(astro_source, '月出时刻')),
        '月没时刻': _normalize_time_like(search_time_after_label(astro_source, '月没时刻')),
    }

    time_candidates = _extract_time_candidates(astro_source)
    order = ['天亮时刻', '天黑时刻', '日出时刻', '日没时刻', '月出时刻', '月没时刻']
    if len(time_candidates) >= 5:
        for idx, key in enumerate(order):
            if idx < len(time_candidates) and not astronomical_times[key]:
                astronomical_times[key] = time_candidates[idx]

    return {
        'confidentiality': confidentiality,
        'date': date_value,
        'title': title_text,
        'approved_name': approved_name,
        'astronomical_times': astronomical_times,
        'raw_text': full_text,
        'raw_lines': title_lines,
        'subregion_texts': {
            'confidentiality': confidentiality_raw,
            'title': title_raw,
            'astronomical_times': astro_raw,
            'approved': approved_raw,
        },
    }


def extract_remark_fields(aligned: np.ndarray, config: Dict[str, Any], engine: Any, lexicon: Dict[str, List[str]]) -> Dict[str, Any]:
    semantic = config['semantic']['remark_fields']
    remark_box = config['regions']['remark']
    remark_img = crop_by_box(aligned, tuple(remark_box))
    remark_lines = _ocr_lines_box(
        engine,
        remark_img,
        (0, 0, remark_img.shape[1], remark_img.shape[0]),
        preprocess=True,
    ) if remark_img.size > 0 else []

    full_text = ' '.join(flatten_region_lines(remark_lines))

    title_text = _ocr_box(
        engine, aligned, semantic['title'],
        preprocess=True,
        prefer_keywords=['备注']
    )
    occupancy_text = _ocr_box(
        engine, aligned, semantic['occupancy_time'],
        preprocess=True,
        prefer_keywords=['占场时间']
    )

    occupancy_time = search_time_after_label(occupancy_text or full_text, '占场时间')
    if not occupancy_time:
        occupancy_time = _normalize_time_like(_clean_label_residue(occupancy_text, ['占场时间']))

    col_names = semantic.get('training_columns', ['序号', '参训机型', '数量', '架次', '核算架次', '时间'])
    training_headers = list(col_names)

    training_rows = []
    row_boxes = semantic.get('training_rows', [])

    for row_box in row_boxes:
        row_result: Dict[str, Any] = {}
        for col_name, box in zip(col_names, row_box):
            text = _ocr_box(engine, aligned, box, preprocess=True, prefer_keywords=[col_name])
            row_result[col_name] = _clean_training_cell(col_name, text, lexicon)

        if not any(v for v in row_result.values()):
            continue

        if _is_header_like_training_row(row_result):
            continue

        nonempty_value_count = sum(1 for v in row_result.values() if v)
        if nonempty_value_count < 2:
            continue

        training_rows.append(row_result)

    total_text = _ocr_box(
        engine, aligned, semantic['total'],
        preprocess=True,
        prefer_keywords=['合计']
    )
    total_value = _clean_label_residue(total_text, ['合计'])

    return {
        'title': title_text or '备注',
        'training_headers': training_headers,
        'training_entries': training_rows,
        'total': total_value,
        'occupancy_time': occupancy_time,
        'raw_text': full_text,
        'raw_lines': remark_lines,
    }


def extract_bottom_fields(aligned: np.ndarray, config: Dict[str, Any], engine: Any, lexicon: Dict[str, List[str]]) -> Dict[str, Any]:
    semantic = config['semantic']['bottom_fields']
    bottom_box = config['regions']['bottom']
    bottom_img = crop_by_box(aligned, tuple(bottom_box))
    bottom_lines = _ocr_lines_box(engine, bottom_img, (0, 0, bottom_img.shape[1], bottom_img.shape[0]), preprocess=True) if bottom_img.size > 0 else []
    full_text = ' '.join(flatten_region_lines(bottom_lines))

    def parse_line(key: str) -> Dict[str, str]:
        row_cfg = semantic[key]
        captain_text = _ocr_box(engine, aligned, row_cfg['队长'], preprocess=True, prefer_keywords=['队长'])
        pc_text = _ocr_box(engine, aligned, row_cfg['政治委员'], preprocess=True, prefer_keywords=['政治委员'])

        captain_text = correct_with_lexicon(_clean_label_residue(captain_text, ['队长']), lexicon, 'names') or _clean_label_residue(captain_text, ['队长'])
        pc_text = correct_with_lexicon(_clean_label_residue(pc_text, ['政治委员']), lexicon, 'names') or _clean_label_residue(pc_text, ['政治委员'])

        if not _is_valid_name_like(captain_text):
            captain_text = ''
        if not _is_valid_name_like(pc_text):
            pc_text = ''

        return {'队长': captain_text, '政治委员': pc_text}

    return {
        'line1': parse_line('line1'),
        'line2': parse_line('line2'),
        'raw_text': full_text,
        'raw_lines': bottom_lines,
    }


__all__ = ['extract_title_fields', 'extract_remark_fields', 'extract_bottom_fields']
