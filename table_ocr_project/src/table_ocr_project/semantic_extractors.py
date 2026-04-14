from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .layout import crop_by_box
from .preprocess import preprocess_cell_for_ocr
from .text_utils import (
    correct_with_lexicon,
    extract_after_label,
    extract_first_date,
    flatten_region_lines,
    normalize_text,
    normalize_time_string,
    search_time_after_label,
)

BBox = Tuple[int, int, int, int]


# =========================
# Basic helpers
# =========================

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


def _ocr_box(engine: Any, image: np.ndarray, box: Sequence[int], preprocess: bool = True, prefer_keywords: Sequence[str] | None = None) -> str:
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


def _is_probably_blank_cell(image: np.ndarray) -> bool:
    if image.size == 0:
        return True
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mx = max(1, int(w * 0.12))
    my = max(1, int(h * 0.12))
    if w - 2 * mx > 2 and h - 2 * my > 2:
        gray = gray[my:h - my, mx:w - mx]
    if gray.size == 0:
        return True
    dark_ratio = float((gray < 210).sum()) / float(gray.size)
    std = float(gray.std())
    return dark_ratio < 0.01 and std < 18.0


def _apply_lexicon(text: str, lexicon: Dict[str, List[str]], field: str) -> str:
    text = normalize_text(text)
    if not text:
        return ''
    return correct_with_lexicon(text, lexicon, field) or text


def _fill_down(values: List[str]) -> List[str]:
    out: List[str] = []
    last = ''
    for v in values:
        vv = normalize_text(v)
        if vv:
            last = vv
        out.append(last)
    return out


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


def _quality_text(text: str) -> int:
    text = normalize_text(text)
    if not text:
        return 0
    score = 0
    score += len([c for c in text if '\u4e00' <= c <= '\u9fff']) * 3
    score += len([c for c in text if c.isdigit()]) * 2
    score += len([c for c in text if c.isalpha()]) * 1
    score += len(text)
    return score


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


def _is_noise_text(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return True
    if _contains_bad_symbol(text):
        return True
    useful = sum(1 for c in text if ('\u4e00' <= c <= '\u9fff') or c.isdigit() or c.isalpha())
    if useful == 0:
        return True
    if len(text) <= 2 and useful <= 1:
        return True
    bad_tokens = {
        '.', '..', '...', ':', '：', ';', '；', ',', '，', '-', '—',
        "'", '"', '`', '·', '!', '！', '?', '？', '/', '\\', '|',
        '心', '电', '中', '北', '丁', '一', '二', '三', '四', '五', '六', '七',
        'SP', 'T', 'C', 'W', 'N', 'X', 'Y'
    }
    if text in bad_tokens:
        return True
    return False


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


def _valid_hour(text: str) -> str:
    text = normalize_text(text).replace('：', ':')
    if not text:
        return ''
    if ':' in text:
        hh = text.split(':')[0]
    else:
        hh = ''.join(ch for ch in text if ch.isdigit())
    if not hh:
        return ''
    try:
        h = int(hh)
    except Exception:
        return ''
    if 0 <= h <= 23:
        return f'{h:02d}'
    return ''


def _valid_minute(text: str) -> str:
    text = normalize_text(text).replace('O', '0').replace('o', '0').replace('：', ':')
    digits = ''.join(ch for ch in text if ch.isdigit())
    if not digits:
        return ''
    if len(digits) == 1:
        digits = '0' + digits
    mm = digits[:2]
    try:
        m = int(mm)
    except Exception:
        return ''
    if 0 <= m <= 59:
        return f'{m:02d}'
    return ''


def _group_hour_labels(hour_texts: List[str], group_size: int = 6) -> List[str]:
    result: List[str] = []
    current = ''
    for i, text in enumerate(hour_texts):
        hh = _valid_hour(text)
        if hh:
            current = hh
        if not current and i % group_size == 0:
            current = ''
        result.append(current)
    return result


def _compose_slot_times(hours: List[str], minutes: List[str]) -> List[str]:
    slots: List[str] = []
    for hh, raw_m in zip(hours, minutes):
        mm = _valid_minute(raw_m)
        if hh and mm:
            slots.append(f'{hh}:{mm}')
        else:
            slots.append('')
    return slots


# =========================
# Title
# =========================

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


# =========================
# Remark
# =========================

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


# =========================
# Bottom
# =========================

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


# =========================
# Main table: geometry-aware OCR
# =========================

def _ink_ratio(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mx = max(1, int(w * 0.10))
    my = max(1, int(h * 0.12))
    if w - 2 * mx > 2 and h - 2 * my > 2:
        gray = gray[my:h - my, mx:w - mx]
    if gray.size == 0:
        return 0.0
    return float((gray < 210).sum()) / float(gray.size)


def _remove_table_lines_light(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return img

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    h, w = inv.shape

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 12), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h // 2)))

    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vk)
    line_mask = cv2.bitwise_or(horiz, vert)

    cleaned = cv2.inpaint(gray, line_mask, 3, cv2.INPAINT_TELEA)
    return cleaned


def _preprocess_main_window_for_ocr(img: np.ndarray, scale: float = 2.5) -> np.ndarray:
    if img is None or img.size == 0:
        return img

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _remove_table_lines_light(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.4, blur, -0.4, 0)

    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def _ocr_best_window_text(engine: Any, crop: np.ndarray, lexicon: Dict[str, List[str]] | None = None) -> str:
    if crop is None or crop.size == 0:
        return ''

    candidates: List[str] = []

    variants = [
        crop,
        _preprocess_main_window_for_ocr(crop, 2.0),
        _preprocess_main_window_for_ocr(crop, 2.8),
        _preprocess_main_window_for_ocr(crop, 3.2),
    ]

    for img in variants:
        if img is None or img.size == 0:
            continue
        try:
            txt = normalize_text(engine.ocr_region_text(img, preprocess=False))
            if txt:
                candidates.append(txt)
        except Exception:
            pass

    if not candidates:
        return ''

    def score(s: str) -> int:
        s = normalize_text(s)
        useful = sum(1 for c in s if ('\u4e00' <= c <= '\u9fff') or c.isalpha() or c.isdigit())
        return useful * 3 + len(s)

    candidates = sorted(set(candidates), key=score, reverse=True)
    best = candidates[0]

    if lexicon is not None:
        fixed = correct_with_lexicon(best, lexicon, 'flight_codes')
        if fixed:
            return fixed

    return best


def _build_local_grid_guides(
    matrix: List[List[Dict[str, Any]]],
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    window_box: Tuple[int, int, int, int],
) -> Tuple[List[int], List[int]]:
    x0, y0, x1, y1 = [int(v) for v in window_box]
    w = max(0, x1 - x0)
    h = max(0, y1 - y0)

    xs = set()
    ys = set()

    row_start = max(0, int(row_start))
    row_end = min(len(matrix) - 1, int(row_end))
    col_start = max(0, int(col_start))
    col_end = min(len(matrix[0]) - 1, int(col_end))

    ref_row = row_start
    for c in range(col_start, col_end + 1):
        bx1, _, bx2, _ = [int(v) for v in matrix[ref_row][c]['bbox']]
        xs.add(bx1 - x0)
        xs.add(bx2 - x0)

    for r in range(row_start, row_end + 1):
        _, by1, _, by2 = [int(v) for v in matrix[r][col_start]['bbox']]
        ys.add(by1 - y0)
        ys.add(by2 - y0)

    xs = sorted(x for x in xs if 0 <= x < w)
    ys = sorted(y for y in ys if 0 <= y < h)
    return xs, ys


def _erase_grid_lines_by_guides(
    mask: np.ndarray,
    grid_xs: Sequence[int] | None = None,
    grid_ys: Sequence[int] | None = None,
    vertical_thickness: int = 2,
    horizontal_thickness: int = 2,
) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask

    out = mask.copy()
    h, w = out.shape[:2]
    grid_mask = np.zeros((h, w), dtype=np.uint8)

    for x in list(grid_xs or []):
        x = int(round(x))
        if 0 <= x < w:
            cv2.line(grid_mask, (x, 0), (x, h - 1), 255, max(1, int(vertical_thickness)))

    for y in list(grid_ys or []):
        y = int(round(y))
        if 0 <= y < h:
            cv2.line(grid_mask, (0, y), (w - 1, y), 255, max(1, int(horizontal_thickness)))

    if cv2.countNonZero(grid_mask) > 0:
        out = cv2.bitwise_and(out, cv2.bitwise_not(grid_mask))

    return out


def _remove_grid_lines_for_geometry(
    img: np.ndarray,
    grid_xs: Sequence[int] | None = None,
    grid_ys: Sequence[int] | None = None,
) -> np.ndarray:
    if img is None or img.size == 0:
        return img

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    h, w = bw.shape[:2]

    if grid_xs or grid_ys:
        fg = _erase_grid_lines_by_guides(
            bw,
            grid_xs=grid_xs,
            grid_ys=grid_ys,
            vertical_thickness=2,
            horizontal_thickness=2,
        )
    else:
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(60, int(round(w * 0.72))), 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(round(h * 0.72)))))
        horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
        vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
        grid = cv2.bitwise_or(horiz, vert)
        fg = cv2.bitwise_and(bw, cv2.bitwise_not(grid))

    # 只做轻微连接，避免把二叉前的短水平线也抹掉
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)))
    return fg

def _select_geometry_components(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    selected = np.zeros_like(mask)

    for idx in range(1, num_labels):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < 8:
            continue

        long_side = max(w, h)
        short_side = max(1, min(w, h))
        fill_ratio = area / float(max(1, w * h))
        aspect = long_side / float(short_side)

        keep = False
        if long_side >= 6 and aspect >= 1.6:
            keep = True
        if long_side >= 8 and fill_ratio <= 0.70:
            keep = True
        if w >= 6 and h <= 5:
            keep = True
        if h >= 6 and w <= 5:
            keep = True

        if keep:
            selected[labels == idx] = 255

    if cv2.countNonZero(selected) == 0:
        return mask
    return selected


def _skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask

    bin_img = ((mask > 0).astype(np.uint8)) * 255

    try:
        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
            return cv2.ximgproc.thinning(bin_img)
    except Exception:
        pass

    skel = np.zeros_like(bin_img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = bin_img.copy()

    while True:
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(work, opened)
        eroded = cv2.erode(work, element)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break

    return skel


def _find_skeleton_keypoints(skel: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    branch_points: List[Tuple[int, int]] = []
    end_points: List[Tuple[int, int]] = []

    if skel is None or skel.size == 0:
        return branch_points, end_points

    ys, xs = np.where(skel > 0)
    h, w = skel.shape

    for y, x in zip(ys.tolist(), xs.tolist()):
        if x <= 0 or x >= w - 1 or y <= 0 or y >= h - 1:
            continue
        roi = skel[y - 1:y + 2, x - 1:x + 2]
        neighbors = int((roi > 0).sum()) - 1

        if neighbors >= 3:
            branch_points.append((x, y))
        elif neighbors == 1:
            end_points.append((x, y))

    return branch_points, end_points



def _detect_hough_line_points(mask: np.ndarray) -> List[Tuple[int, int]]:
    if mask is None or mask.size == 0:
        return []

    h, w = mask.shape
    lines = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 180.0,
        threshold=max(8, w // 18),
        minLineLength=max(6, w // 10),
        maxLineGap=5,
    )

    points: List[Tuple[int, int]] = []
    if lines is None:
        return points

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in line]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue

        length = float(np.hypot(dx, dy))
        if length < 6.0:
            continue

        angle = abs(np.degrees(np.arctan2(dy, dx)))
        # 保留水平线和斜线，只排除接近竖直的残余网格线
        if 75 <= angle <= 105:
            continue

        points.append((x1, y1))
        points.append((x2, y2))

    return points


def _analyze_event_geometry(
    window: np.ndarray,
    grid_xs: Sequence[int] | None = None,
    grid_ys: Sequence[int] | None = None,
) -> Dict[str, Any]:
    if window is None or window.size == 0:
        return {
            'anchor_x': None,
            'tail_x': None,
            'foreground': window,
            'skeleton': window,
            'branch_points': [],
            'end_points': [],
            'line_points': [],
        }

    fg = _remove_grid_lines_for_geometry(window, grid_xs=grid_xs, grid_ys=grid_ys)
    fg = _select_geometry_components(fg)

    if fg is None or fg.size == 0 or cv2.countNonZero(fg) == 0:
        return {
            'anchor_x': None,
            'tail_x': None,
            'foreground': fg,
            'skeleton': fg,
            'branch_points': [],
            'end_points': [],
            'line_points': [],
        }

    skel = _skeletonize_binary(fg)
    branch_points, end_points = _find_skeleton_keypoints(skel)
    line_points = _detect_hough_line_points(fg)

    h, w = fg.shape[:2]

    def _valid(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for x, y in points:
            if 1 <= x < w - 1 and 1 <= y < h - 1:
                out.append((int(x), int(y)))
        return out

    branch_points = _valid(branch_points)
    end_points = _valid(end_points)
    line_points = _valid(line_points)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((fg > 0).astype(np.uint8), connectivity=8)
    components: List[Dict[str, Any]] = []
    for idx in range(1, num_labels):
        x, y, ww, hh, area = [int(v) for v in stats[idx]]
        if area < 6:
            continue
        pts_y, pts_x = np.where(labels == idx)
        if pts_x.size == 0:
            continue
        min_x = int(pts_x.min())
        max_x = int(pts_x.max())
        min_y = int(pts_y.min())
        max_y = int(pts_y.max())
        comp_branch = [p for p in branch_points if labels[p[1], p[0]] == idx]
        comp_end = [p for p in end_points if labels[p[1], p[0]] == idx]
        comp_line = [p for p in line_points if labels[p[1], p[0]] == idx]
        fill_ratio = area / float(max(1, ww * hh))
        aspect = max(ww, hh) / float(max(1, min(ww, hh)))
        line_like = bool(comp_branch or comp_line or (ww >= 6 and (aspect >= 1.6 or fill_ratio <= 0.72)))

        score = 0
        if comp_branch:
            score += 100
        if comp_line:
            score += 60
        if comp_end:
            score += 15
        score += min(40, ww)
        if ww >= 6 and hh <= max(8, ww):
            score += 20

        components.append({
            'label': idx,
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            'w': ww,
            'h': hh,
            'area': area,
            'branch_points': comp_branch,
            'end_points': comp_end,
            'line_points': comp_line,
            'line_like': line_like,
            'score': score,
        })

    target: Dict[str, Any] | None = None
    candidates = [c for c in components if c['line_like'] or c['score'] >= 40]

    if candidates:
        target = sorted(
            candidates,
            key=lambda c: (
                -bool(c['branch_points']),
                -bool(c['line_points']),
                c['min_x'],
                -c['w'],
                -c['score'],
                -c['area'],
            ),
        )[0]
    elif components:
        target = sorted(components, key=lambda c: (c['min_x'], -c['w'], -c['area']))[0]

    anchor_x = None
    tail_x = None
    if target is not None:
        if target['line_points']:
            anchor_x = min(p[0] for p in target['line_points'])
            tail_x = max(p[0] for p in target['line_points'])
        elif target['end_points']:
            anchor_x = min(p[0] for p in target['end_points'])
            tail_x = max(p[0] for p in target['end_points'])
        elif target['branch_points']:
            anchor_x = min(p[0] for p in target['branch_points'])
            tail_x = max(p[0] for p in target['branch_points'])
        else:
            anchor_x = int(target['min_x'])
            tail_x = int(target['max_x'])
    else:
        col_sum = (fg > 0).sum(axis=0)
        nonzero_cols = np.where(col_sum > 0)[0]
        if nonzero_cols.size > 0:
            anchor_x = int(nonzero_cols[0])
            tail_x = int(nonzero_cols[-1])

    return {
        'anchor_x': anchor_x,
        'tail_x': tail_x,
        'foreground': fg,
        'skeleton': skel,
        'branch_points': branch_points,
        'end_points': end_points,
        'line_points': line_points,
    }


def _remove_grid_lines_for_segmentation(
    img: np.ndarray,
    grid_xs: Sequence[int] | None = None,
    grid_ys: Sequence[int] | None = None,
) -> np.ndarray:
    if img is None or img.size == 0:
        return img

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    h, w = bw.shape[:2]
    if grid_xs or grid_ys:
        fg = _erase_grid_lines_by_guides(
            bw,
            grid_xs=grid_xs,
            grid_ys=grid_ys,
            vertical_thickness=2,
            horizontal_thickness=2,
        )
    else:
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, int(round(w * 0.68))), 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(18, int(round(h * 0.68)))))
        horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
        vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
        grid = cv2.bitwise_or(horiz, vert)
        fg = cv2.bitwise_and(bw, cv2.bitwise_not(grid))

    # 轻微连接事件线和相邻字符，避免投影断得太碎
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)))
    return fg

def _smooth_projection(values: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)

    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size,), dtype=np.float32) / float(kernel_size)
    return np.convolve(values.astype(np.float32), kernel, mode='same')


def _find_projection_runs(
    projection: np.ndarray,
    threshold: float,
    min_width: int = 3,
    allow_gap: int = 3,
) -> List[Tuple[int, int]]:
    if projection.size == 0:
        return []

    active = np.where(projection >= float(threshold))[0]
    if active.size == 0:
        return []

    runs: List[List[int]] = [[int(active[0]), int(active[0])]]
    for x in active[1:].tolist():
        x = int(x)
        if x - runs[-1][1] <= int(allow_gap) + 1:
            runs[-1][1] = x
        else:
            runs.append([x, x])

    out: List[Tuple[int, int]] = []
    for s, e in runs:
        if e - s + 1 >= int(min_width):
            out.append((int(s), int(e)))
    return out


def _split_projection_run_recursive(
    projection: np.ndarray,
    start_x: int,
    end_x: int,
    max_width: int,
    min_piece_width: int = 3,
) -> List[Tuple[int, int]]:
    start_x = int(start_x)
    end_x = int(end_x)
    width = end_x - start_x + 1
    if width <= max_width:
        return [(start_x, end_x)]

    seg = projection[start_x:end_x + 1].astype(np.float32)
    if seg.size == 0:
        return [(start_x, end_x)]

    margin = max(4, width // 8)
    if width <= margin * 2 + min_piece_width * 2:
        return [(start_x, end_x)]

    center = seg[margin:width - margin]
    if center.size < min_piece_width * 2:
        return [(start_x, end_x)]

    center_min = float(center.min())
    center_max = float(center.max())
    valley_threshold = max(1.0, center_max * 0.30)

    candidate_rel = np.where(center <= valley_threshold)[0]
    if candidate_rel.size > 0:
        center_mid = center.size / 2.0
        best = int(candidate_rel[np.argmin(np.abs(candidate_rel - center_mid))]) + margin
    else:
        best = int(np.argmin(center)) + margin

    valley_x = start_x + best
    valley_value = float(projection[valley_x])

    left_peak = float(projection[start_x:valley_x].max()) if valley_x > start_x else 0.0
    right_peak = float(projection[valley_x + 1:end_x + 1].max()) if valley_x < end_x else 0.0
    ref_peak = min(left_peak, right_peak)

    if (
        valley_x - start_x >= min_piece_width
        and end_x - valley_x >= min_piece_width
        and valley_value <= max(1.0, ref_peak * 0.45)
    ):
        left = _split_projection_run_recursive(
            projection,
            start_x,
            valley_x - 1,
            max_width=max_width,
            min_piece_width=min_piece_width,
        )
        right = _split_projection_run_recursive(
            projection,
            valley_x + 1,
            end_x,
            max_width=max_width,
            min_piece_width=min_piece_width,
        )
        return left + right

    return [(start_x, end_x)]


def _find_active_segments_by_ink(
    matrix: List[List[Dict[str, Any]]],
    row_idx: int,
    main_table_img: np.ndarray,
    time_col_start: int,
    time_col_end: int,
    min_ratio: float = 0.008,
    allow_gap: int = 1,
    row_expand: int = 1,
) -> List[Tuple[int, int]]:
    def _merge_col_segments(segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not segments:
            return []

        merged: List[List[int]] = []
        for s, e in sorted(segments):
            if not merged or s - merged[-1][1] > allow_gap + 1:
                merged.append([int(s), int(e)])
            else:
                merged[-1][1] = max(int(e), merged[-1][1])

        return [(seg[0], seg[1]) for seg in merged]

    def _legacy_segments() -> List[Tuple[int, int]]:
        active_cols: List[int] = []
        for c in range(time_col_start, time_col_end + 1):
            x1 = min(int(matrix[r][c]['bbox'][0]) for r in range(top_row, bottom_row + 1))
            y1 = min(int(matrix[r][c]['bbox'][1]) for r in range(top_row, bottom_row + 1))
            x2 = max(int(matrix[r][c]['bbox'][2]) for r in range(top_row, bottom_row + 1))
            y2 = max(int(matrix[r][c]['bbox'][3]) for r in range(top_row, bottom_row + 1))
            crop = main_table_img[y1:y2, x1:x2]
            ratio = _ink_ratio(crop)
            if ratio >= min_ratio:
                active_cols.append(c)

        if not active_cols:
            return []

        parts: List[List[int]] = [[active_cols[0]]]
        for c in active_cols[1:]:
            if c - parts[-1][-1] <= allow_gap + 1:
                parts[-1].append(c)
            else:
                parts.append([c])

        return [(seg[0], seg[-1]) for seg in parts]

    top_row = max(0, row_idx - row_expand)
    bottom_row = min(len(matrix) - 1, row_idx + row_expand)

    band_x1 = int(matrix[row_idx][time_col_start]['bbox'][0])
    band_x2 = int(matrix[row_idx][time_col_end]['bbox'][2])
    band_y1 = int(matrix[top_row][0]['bbox'][1])
    band_y2 = int(matrix[bottom_row][0]['bbox'][3])

    if band_x2 <= band_x1 or band_y2 <= band_y1:
        return _legacy_segments()

    band_box = (band_x1, band_y1, band_x2, band_y2)
    band = main_table_img[band_y1:band_y2, band_x1:band_x2]
    grid_xs, grid_ys = _build_local_grid_guides(
        matrix,
        row_start=top_row,
        row_end=bottom_row,
        col_start=time_col_start,
        col_end=time_col_end,
        window_box=band_box,
    )
    mask = _remove_grid_lines_for_segmentation(band, grid_xs=grid_xs, grid_ys=grid_ys)

    if mask is None or mask.size == 0 or cv2.countNonZero(mask) == 0:
        return _legacy_segments()

    cell_widths = [
        max(1, int(matrix[row_idx][c]['bbox'][2]) - int(matrix[row_idx][c]['bbox'][0]))
        for c in range(time_col_start, time_col_end + 1)
    ]
    cell_w = max(1, int(np.median(np.asarray(cell_widths, dtype=np.int32)))) if cell_widths else 1

    proj = (mask > 0).sum(axis=0).astype(np.float32)
    smooth = _smooth_projection(proj, kernel_size=max(5, cell_w))

    nonzero = smooth[smooth > 0.2]
    if nonzero.size == 0:
        return _legacy_segments()

    active_threshold = max(1.2, float(np.percentile(nonzero, 30)))
    pixel_runs = _find_projection_runs(
        smooth,
        threshold=active_threshold,
        min_width=max(3, cell_w // 3),
        allow_gap=max(2, cell_w // 4),
    )
    if not pixel_runs:
        return _legacy_segments()

    max_run_width = max(cell_w * 12, 40)
    split_runs: List[Tuple[int, int]] = []
    for s, e in pixel_runs:
        split_runs.extend(
            _split_projection_run_recursive(
                smooth,
                s,
                e,
                max_width=max_run_width,
                min_piece_width=max(3, cell_w // 3),
            )
        )

    def _pixel_to_col(local_x: int) -> int:
        global_x = band_x1 + int(local_x)
        for c in range(time_col_start, time_col_end + 1):
            x1, _, x2, _ = [int(v) for v in matrix[row_idx][c]['bbox']]
            if x1 <= global_x < x2:
                return c
        centers: List[Tuple[float, int]] = []
        for c in range(time_col_start, time_col_end + 1):
            x1, _, x2, _ = [int(v) for v in matrix[row_idx][c]['bbox']]
            centers.append((((x1 + x2) / 2.0), c))
        centers = sorted(centers, key=lambda item: abs(item[0] - global_x))
        return centers[0][1] if centers else time_col_start

    col_segments: List[Tuple[int, int]] = []
    for start_x, end_x in split_runs:
        local = mask[:, start_x:end_x + 1]
        if local.size == 0:
            continue
        area = int(cv2.countNonZero(local))
        if area < 6:
            continue

        start_col = _pixel_to_col(int(start_x))
        end_col = _pixel_to_col(int(max(start_x, end_x)))
        start_col = max(time_col_start, min(start_col, time_col_end))
        end_col = max(time_col_start, min(end_col, time_col_end))
        if end_col < start_col:
            start_col, end_col = end_col, start_col

        if end_col - start_col + 1 == 1 and area < 10:
            continue

        col_segments.append((start_col, end_col))

    col_segments = _merge_col_segments(col_segments)
    if not col_segments:
        return _legacy_segments()

    return col_segments

def _expand_segment(
    start_col: int,
    end_col: int,
    time_col_start: int,
    time_col_end: int,
    pad_slots: int = 1,
) -> Tuple[int, int]:
    s = max(time_col_start, start_col - pad_slots)
    e = min(time_col_end, end_col + pad_slots)
    return s, e


def _crop_segment_window(
    matrix: List[List[Dict[str, Any]]],
    row_idx: int,
    main_table_img: np.ndarray,
    start_col: int,
    end_col: int,
    row_expand: int = 1,
    vpad: int = 3,
    hpad: int = 2,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    top_row = max(0, row_idx - row_expand)
    bottom_row = min(len(matrix) - 1, row_idx + row_expand)

    x1 = int(matrix[row_idx][start_col]['bbox'][0]) - hpad
    x2 = int(matrix[row_idx][end_col]['bbox'][2]) + hpad
    y1 = int(matrix[top_row][0]['bbox'][1]) - vpad
    y2 = int(matrix[bottom_row][0]['bbox'][3]) + vpad

    H, W = main_table_img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    if x2 <= x1 or y2 <= y1:
        return main_table_img[0:0, 0:0].copy(), (0, 0, 0, 0)

    return main_table_img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def _segment_to_slots(start_col: int, end_col: int, time_col_start: int, slot_times: List[str]) -> List[str]:
    slots: List[str] = []
    for c in range(start_col, end_col + 1):
        idx = c - time_col_start
        if 0 <= idx < len(slot_times):
            if slot_times[idx]:
                slots.append(slot_times[idx])
    return slots


def _map_global_x_to_col(
    global_x: int | None,
    row_cells: List[Dict[str, Any]],
    time_col_start: int,
    time_col_end: int,
) -> int | None:
    if global_x is None:
        return None

    for c in range(time_col_start, time_col_end + 1):
        x1, _, x2, _ = [int(v) for v in row_cells[c]['bbox']]
        if x1 <= global_x < x2:
            return c

    centers: List[Tuple[float, int]] = []
    for c in range(time_col_start, time_col_end + 1):
        x1, _, x2, _ = [int(v) for v in row_cells[c]['bbox']]
        centers.append((((x1 + x2) / 2.0), c))

    centers = sorted(centers, key=lambda item: abs(item[0] - global_x))
    return centers[0][1] if centers else None


def extract_main_table(
    aligned: np.ndarray,
    config: Dict[str, Any],
    engine: Any,
    output_dir: Path,
    lexicon: Dict[str, List[str]],
) -> Dict[str, Any]:
    from .pipeline import save_image

    main_box = config['regions']['main_table']
    main_table_img = crop_by_box(aligned, tuple(main_box))
    x_lines = config['grid']['x_lines']
    y_lines = config['grid']['y_lines']
    schema = config['semantic']['main_table_schema']

    rows = len(y_lines) - 1
    cols = len(x_lines) - 1

    cell_dir = output_dir / 'cell_preprocessed'
    segment_dir = output_dir / 'main_segments'
    geometry_dir = output_dir / 'main_geometry'
    cell_dir.mkdir(parents=True, exist_ok=True)
    segment_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir.mkdir(parents=True, exist_ok=True)

    matrix: List[List[Dict[str, Any]]] = []
    for r in range(rows):
        row_data: List[Dict[str, Any]] = []
        for c in range(cols):
            x1, y1, x2, y2 = x_lines[c], y_lines[r], x_lines[c + 1], y_lines[r + 1]
            crop = main_table_img[y1:y2, x1:x2]

            if _is_probably_blank_cell(crop):
                ocr = {'text': '', 'score': 0.0}
            else:
                ocr = engine.ocr_cell(crop, remove_lines=True)

            text = normalize_text(ocr['text'])
            score = float(ocr['score'])

            if _is_noise_text(text) and score < 0.90:
                text = ''

            entry = {
                'row': r,
                'col': c,
                'bbox': [x1, y1, x2, y2],
                'text': text,
                'score': score,
            }
            row_data.append(entry)

            if r < 12:
                pre = preprocess_cell_for_ocr(crop, remove_lines=True)
                save_image(cell_dir / f'r{r:03d}_c{c:03d}.png', pre)

        matrix.append(row_data)

    raw_grid = [[cell['text'] for cell in row] for row in matrix]

    left_cols = schema['left_cols']
    time_col_start = int(schema['time_cols'][0])
    time_col_end = int(schema['time_cols'][1])
    right_cols = schema['right_cols']
    hour_row = int(schema['hour_row'])
    minute_row = int(schema['minute_row'])
    body_start = int(schema['body_start_row'])

    hour_texts = [raw_grid[hour_row][c] for c in range(time_col_start, time_col_end + 1)]
    minute_texts = [raw_grid[minute_row][c] for c in range(time_col_start, time_col_end + 1)]

    # 小时行和分钟行继续沿用原逻辑，只负责生成时间轴
    hour_filled = _group_hour_labels(hour_texts, group_size=6)
    slot_times = _compose_slot_times(hour_filled, minute_texts)

    valid_slot_indices = [idx for idx, val in enumerate(slot_times) if normalize_text(val)]
    effective_time_col_end = (
        time_col_start + valid_slot_indices[-1]
        if valid_slot_indices
        else time_col_end
    )

    left_fields_raw = {idx: [raw_grid[r][idx] for r in range(body_start, rows)] for idx in left_cols}
    left_fields_filled = {idx: _fill_down(vals) for idx, vals in left_fields_raw.items()}

    body_rows = []
    for rr in range(body_start, rows):
        row_cells = matrix[rr]

        aircraft_type = _apply_lexicon(left_fields_filled[left_cols[0]][rr - body_start], lexicon, 'aircraft_types')
        aircraft_no = _apply_lexicon(left_fields_filled[left_cols[1]][rr - body_start], lexicon, 'aircraft_numbers')
        secondary_code = normalize_text(left_fields_filled[left_cols[2]][rr - body_start])

        name = correct_with_lexicon(raw_grid[rr][right_cols[0]], lexicon, 'names') or normalize_text(raw_grid[rr][right_cols[0]])
        code_name = correct_with_lexicon(raw_grid[rr][right_cols[1]], lexicon, 'code_names') or normalize_text(raw_grid[rr][right_cols[1]])

        if not _is_valid_name_like(name):
            name = ''
        if not _is_valid_name_like(code_name):
            code_name = ''

        # 关键修改1：活动段不再只看当前单行，而是看上一行/当前行/下一行联合墨迹
        segments = _find_active_segments_by_ink(
            matrix=matrix,
            row_idx=rr,
            main_table_img=main_table_img,
            time_col_start=time_col_start,
            time_col_end=effective_time_col_end,
            min_ratio=0.008,
            allow_gap=1,
            row_expand=1,
        )

        events = []
        raw_nonempty_cells = []

        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            exp_start, exp_end = _expand_segment(
                seg_start,
                seg_end,
                time_col_start=time_col_start,
                time_col_end=effective_time_col_end,
                pad_slots=1,
            )

            # 关键修改2：窗口改成跨行裁剪，避免分叉线被裁断
            window, window_box = _crop_segment_window(
                matrix=matrix,
                row_idx=rr,
                main_table_img=main_table_img,
                start_col=exp_start,
                end_col=exp_end,
                row_expand=1,
                vpad=3,
                hpad=2,
            )

            if window.size > 0:
                save_image(segment_dir / f'row_{rr:03d}_seg_{seg_idx:02d}_{exp_start}_{exp_end}.png', window)

            # 关键修改3：时间起点用几何锚点，不再直接拿 segment 左边界
            top_row_geom = max(0, rr - 1)
            bottom_row_geom = min(len(matrix) - 1, rr + 1)
            local_grid_xs, local_grid_ys = _build_local_grid_guides(
                matrix,
                row_start=top_row_geom,
                row_end=bottom_row_geom,
                col_start=exp_start,
                col_end=exp_end,
                window_box=window_box,
            )
            geom = _analyze_event_geometry(window, grid_xs=local_grid_xs, grid_ys=local_grid_ys)

            if geom.get('foreground') is not None and getattr(geom['foreground'], 'size', 0) > 0:
                save_image(geometry_dir / f'row_{rr:03d}_seg_{seg_idx:02d}_fg.png', geom['foreground'])
            if geom.get('skeleton') is not None and getattr(geom['skeleton'], 'size', 0) > 0:
                save_image(geometry_dir / f'row_{rr:03d}_seg_{seg_idx:02d}_skel.png', geom['skeleton'])

            anchor_x_local = geom.get('anchor_x')
            tail_x_local = geom.get('tail_x')

            anchor_x_global = None if anchor_x_local is None else int(window_box[0] + anchor_x_local)
            tail_x_global = None if tail_x_local is None else int(window_box[0] + tail_x_local)

            anchor_col = _map_global_x_to_col(anchor_x_global, row_cells, time_col_start, effective_time_col_end)
            tail_col = _map_global_x_to_col(tail_x_global, row_cells, time_col_start, effective_time_col_end)

            if anchor_col is None:
                anchor_col = exp_start
            if tail_col is None:
                tail_col = exp_end

            anchor_col = max(time_col_start, min(anchor_col, effective_time_col_end))
            tail_col = max(time_col_start, min(tail_col, effective_time_col_end))
            if tail_col < anchor_col:
                tail_col = anchor_col

            window_text = _ocr_best_window_text(engine, window, lexicon=lexicon)
            window_text = normalize_text(window_text)

            note_like_tokens = {'SD', 'STP', 'CQY', 'GC', 'MF', 'WP', '项目注释', '课目注释'}
            note_key = window_text.replace(' ', '').replace(':', '').replace('：', '').upper()
            has_geom_signal = bool(geom.get('branch_points')) or bool(geom.get('end_points')) or bool(geom.get('line_points'))
            if note_key in note_like_tokens and not has_geom_signal:
                continue

            # 关键修改4：文字识别失败时也不直接丢事件，只要几何锚点能映射到时间就保留
            if window_text and (_is_noise_text(window_text) or _quality_text(window_text) < 3):
                window_text = ''

            slots = _segment_to_slots(anchor_col, tail_col, time_col_start, slot_times)
            if not slots:
                slots = _segment_to_slots(exp_start, exp_end, time_col_start, slot_times)
            if not slots:
                continue

            events.append({
                'start_time': slots[0],
                'end_time': slots[-1],
                'slots': slots,
                'text': window_text,
                'anchor_time': slot_times[anchor_col - time_col_start] if 0 <= anchor_col - time_col_start < len(slot_times) else '',
                'anchor_col': anchor_col,
                'tail_col': tail_col,
                'raw_cells': [{'row': rr, 'col': c} for c in range(anchor_col, tail_col + 1)],
            })

            raw_nonempty_cells.extend(
                [
                    {
                        'row': rr,
                        'col': c,
                        'slot': slot_times[c - time_col_start] if 0 <= c - time_col_start < len(slot_times) else '',
                    }
                    for c in range(anchor_col, tail_col + 1)
                ]
            )

        reliable_side = any([aircraft_type, aircraft_no, secondary_code, name, code_name])

        if not reliable_side and not events:
            continue

        body_rows.append({
            'grid_row': rr,
            'aircraft_type': aircraft_type,
            'aircraft_no': aircraft_no,
            'secondary_code': secondary_code,
            'name': name,
            'code_name': code_name,
            'events': events,
            'raw_nonempty_cells': raw_nonempty_cells,
        })

    header_labels = []
    for r in schema.get('top_label_rows', []):
        texts = [normalize_text(raw_grid[r][c]) for c in left_cols]
        joined = ' '.join(t for t in texts if t)
        if joined:
            header_labels.append({'row': r, 'label': joined})

    subheader = {str(c): raw_grid[schema['subheader_row']][c] for c in left_cols}

    return {
        'raw_grid': raw_grid,
        'hours': hour_filled,
        'minutes': [_valid_minute(v) for v in minute_texts],
        'slot_times': slot_times,
        'header_labels': header_labels,
        'subheader': subheader,
        'body_rows': body_rows,
        'schema': schema,
    }
