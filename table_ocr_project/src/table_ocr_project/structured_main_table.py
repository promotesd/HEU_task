from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .pipeline import read_image
from .text_utils import best_fuzzy_match, correct_with_lexicon, normalize_text

OCRItem = Dict[str, Any]
Event = Dict[str, Any]

SECONDARY_TO_AIRCRAFT_NO = {
    '0750': '60', '0751': '61', '0752': '63', '0753': '74',
    '0701': '374', '0702': '376', '0703': '07', '0705': '10',
}
GROUP_AIRCRAFT_FALLBACK = ['60', '61', '63', '74', '374', '376', '07', '10', '09']
GROUP_SECONDARY_FALLBACK = ['0750', '0751', '0752', '0753', '0701', '0702', '0703', '0705', '']
EXTRA_CODE_NAMES = [
    '森153', '光180', '豪182', '汤191', '冯185', '玉165', '韩171', '说176',
    '潘851', '黄827', '彭823', '胜809', '苑813', '宫850', '郑807', '贾831',
    '彤812', '嘎825', '飞826', '高835', '本150'
]
IGNORE_EVENT_TEXTS = {'课目注释', '姓名', '代字', '代号', '姓名代字代号', '机型', '机号', '二次代码'}
TOP_SECTION_HEADER_TEXTS = {'姓名', '代字', '代号', '姓名代字代号', '机型', '机号', '二次代码'}
FLIGHT_MARKER_TEXTS = {'SD', 'STP', 'CQY', 'GC', 'MF', 'CC'}
PAIR_MARKERS = {'MF', 'GC', 'CC'}
WEAK_NOISE_TEXTS = {
    '25', '体', '3π', '60T', '6.0T', '32T', '3.2T', '1030', '03', '50T',
    '6OT', 'OOT', '0OT', '忠', '春', '星', '马', '厦', '南', '自', 'SP', 'SWP',
    'STP', '-', '5o', '5O'
}
PAIR_HINT_TOKENS = {'0.55T', '0.65T', '103', 'MF(伴航)', 'MF（伴航）'}
XXA_DROP_REMARKS = {'CC', 'cc', 'hoho'}
IMPLICIT_CODE_EXCLUDE = {'103', '414', '044', '44', '055', '065', '070', '079'}


def _find_band_index(value: float, edges: Sequence[int]) -> int:
    for idx in range(len(edges) - 1):
        if float(edges[idx]) <= value < float(edges[idx + 1]):
            return idx
    if value < float(edges[0]):
        return 0
    return len(edges) - 2


def _build_slot_times(num_slots: int, start_hour: int = 12, step_minutes: int = 10) -> List[str]:
    slots: List[str] = []
    total_minutes = start_hour * 60
    for idx in range(num_slots):
        minutes = total_minutes + idx * step_minutes
        slots.append(f'{minutes // 60:02d}:{minutes % 60:02d}')
    return slots


def _time_to_minutes(text: str) -> int:
    text = normalize_text(text)
    if not text or ':' not in text:
        return 0
    hh, mm = text.split(':', 1)
    if not hh.isdigit() or not mm.isdigit():
        return 0
    return int(hh) * 60 + int(mm)


def _minutes_to_time(value: int) -> str:
    value = max(0, int(value))
    return f'{value // 60:02d}:{value % 60:02d}'


def _normalize_ocr_text(text: str) -> str:
    text = normalize_text(text)
    replacements = {
        'OTE1': '0751', 'OT61': '0751', 'OTE3': '0753',
        'COY': 'CQY', 'COV': 'CQY', 'FQY': 'CQY', 'cqr': 'CQY', 'qY': 'CQY',
        'xX5': 'XX5', 'xXs': 'XXS',
        '9ST': 'STP', '5TP': 'STP', 'SWP': 'STP', 'SYP': 'STP',
        'M（件秋)': 'MF(伴航)', 'M(件秋)': 'MF(伴航)', 'MF（伴轨）': 'MF(伴航)',
        'MF（伴航）': 'MF(伴航)', 'AI(伴轨)': 'MF(伴航)', 'AI（伴轨）': 'MF(伴航)',
        'AI(伴航)': 'MF(伴航)', 'GC ': 'GC', 'CC ': 'CC', '9.5T': 'STP',
        '货827': '黄827', '盘185': '冯185', '猛185': '冯185', '扬191': '汤191',
        '期825': '嘎825', '贤831': '贾831', '形812': '彤812', '博823': '彭823',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return normalize_text(text)


def _clean_aircraft_type(text: str) -> str:
    text = _normalize_ocr_text(text).upper().replace(' ', '')
    match = re.search(r'XX[0-9A-Z]', text)
    return match.group(0) if match else text


def _clean_top_label(text: str) -> str:
    text = _normalize_ocr_text(text)
    if not text:
        return ''
    replacements = {
        '至梅1号': '至德1号', '至梅7号': '至德7号', '至梅9号': '至德9号',
        '至徳1号': '至德1号', '至徳7号': '至德7号', '至徳9号': '至德9号',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return normalize_text(text)


def _clean_aircraft_no(text: str, lexicon: Dict[str, List[str]]) -> str:
    text = _normalize_ocr_text(text)
    digits = ''.join(ch for ch in text if ch.isdigit())
    if not digits:
        return ''
    if digits in lexicon.get('aircraft_types', []):
        return digits
    matched = best_fuzzy_match(digits, lexicon.get('aircraft_types', []), min_score=0.34)
    if matched and matched in lexicon.get('aircraft_types', []):
        return matched
    return digits.zfill(2) if len(digits) == 1 else digits


def _clean_secondary_code(text: str, lexicon: Dict[str, List[str]]) -> str:
    text = _normalize_ocr_text(text)
    digits = ''.join(ch for ch in text if ch.isdigit())
    if not digits:
        return ''
    digits = digits[:4]
    choices = list(lexicon.get('aircraft_numbers', [])) + ['0753']
    if digits in choices:
        return digits
    matched = best_fuzzy_match(digits, choices, min_score=0.34)
    return matched if matched else digits


def _looks_like_code_name(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff][0-9]{2,3}', _normalize_ocr_text(text)))


def _strict_pick_code_name(text: str, candidates: Sequence[str], min_score: float = 0.72) -> str:
    text = _normalize_ocr_text(text)
    direct = re.findall(r'[\u4e00-\u9fff][0-9]{2,3}', text)
    if not direct:
        return ''
    target = direct[0]
    matched = best_fuzzy_match(target, candidates, min_score=min_score)
    return matched if matched else target


def _clean_code_name(text: str, lexicon: Dict[str, List[str]]) -> str:
    text = _normalize_ocr_text(text)
    if not text:
        return ''
    candidates = list(dict.fromkeys(lexicon.get('code_names', []) + EXTRA_CODE_NAMES))
    if _looks_like_code_name(text):
        return _strict_pick_code_name(text, candidates, min_score=0.70)
    digits = ''.join(ch for ch in text if ch.isdigit())
    if len(digits) == 3:
        suffix_matches = [cand for cand in candidates if cand.endswith(digits)]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
    corrected = correct_with_lexicon(text, {'code_names': candidates}, 'code_names')
    return corrected or text


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        value = normalize_text(value)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _extract_markers(text: str) -> List[str]:
    text = _normalize_ocr_text(text).upper()
    out: List[str] = []
    for marker in ['STP', 'CQY', 'SD', 'MF', 'GC', 'CC']:
        if marker in text:
            out.append(marker)
    return _dedupe_preserve_order(out)


def _extract_direct_code_names(text: str, lexicon: Dict[str, List[str]]) -> List[str]:
    text = _normalize_ocr_text(text)
    matches = re.findall(r'[\u4e00-\u9fff][0-9]{2,3}', text)
    return _dedupe_preserve_order([_clean_code_name(m, lexicon) for m in matches])


def _extract_suffix_code_names(text: str, lexicon: Dict[str, List[str]]) -> List[str]:
    text = _normalize_ocr_text(text)
    candidates = list(dict.fromkeys(lexicon.get('code_names', []) + EXTRA_CODE_NAMES))
    out: List[str] = []
    for digits in re.findall(r'\d{3,4}', text):
        suffix = digits[-3:]
        if suffix in IMPLICIT_CODE_EXCLUDE:
            continue
        matches = [cand for cand in candidates if cand.endswith(suffix)]
        if len(matches) == 1:
            out.append(matches[0])
    return _dedupe_preserve_order(out)


def _extract_code_names(text: str, lexicon: Dict[str, List[str]], aircraft_type: str = '') -> List[str]:
    codes = _extract_direct_code_names(text, lexicon)
    if aircraft_type == 'XX5':
        codes = _dedupe_preserve_order(codes + _extract_suffix_code_names(text, lexicon))
    return codes


def _strip_markers_and_codes(text: str, lexicon: Dict[str, List[str]], aircraft_type: str = '') -> str:
    text = _normalize_ocr_text(text)
    for marker in ['STP', 'CQY', 'SD', 'MF', 'GC', 'CC']:
        text = text.replace(marker, ' ')
    text = re.sub(r'[\u4e00-\u9fff][0-9]{2,3}', ' ', text)
    if aircraft_type == 'XX5':
        for digits in re.findall(r'\d{3,4}', text):
            suffix = digits[-3:]
            if suffix not in IMPLICIT_CODE_EXCLUDE:
                text = text.replace(digits, ' ')
    text = re.sub(r'[\s,;；，]+', ' ', text).strip()
    return normalize_text(text)


def _classify_text(text: str, lexicon: Dict[str, List[str]], aircraft_type: str = '') -> str:
    text = _normalize_ocr_text(text)
    markers = _extract_markers(text)
    codes = _extract_code_names(text, lexicon, aircraft_type)
    residual = _strip_markers_and_codes(text, lexicon, aircraft_type)
    if markers and codes:
        return 'marker_code'
    if markers and residual:
        return 'marker_desc'
    if markers:
        return 'marker'
    if codes and residual:
        return 'code_desc'
    if codes:
        return 'code'
    if not residual:
        return 'noise'
    if residual in WEAK_NOISE_TEXTS:
        return 'weak'
    if aircraft_type == 'XXS' and residual in PAIR_HINT_TOKENS:
        return 'desc'
    if re.fullmatch(r'[0-9.\-]+', residual):
        return 'weak'
    if len(residual) <= 2 and not any(ch.isdigit() for ch in residual):
        return 'weak'
    return 'desc'


def _build_item_from_line(line: Any, x_lines: Sequence[int], y_lines: Sequence[int]) -> OCRItem:
    text = _normalize_ocr_text(line.text)
    xs = [point[0] for point in line.box]
    ys = [point[1] for point in line.box]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = sum(xs) / 4.0
    y_center = sum(ys) / 4.0
    width = max(1.0, x_max - x_min)
    core_left = x_min + 0.18 * width
    core_right = x_max - 0.18 * width
    if core_right < core_left:
        core_left, core_right = x_min, x_max
    return {
        'text': text,
        'score': float(line.score),
        'x': x_center,
        'x_center': x_center,
        'y': y_center,
        'y_center': y_center,
        'x_min': float(x_min),
        'x_max': float(x_max),
        'y_min': float(y_min),
        'y_max': float(y_max),
        'row': _find_band_index(y_center, y_lines),
        'col': _find_band_index(x_center, x_lines),
        'col_start': _find_band_index(x_min, x_lines),
        'col_end': _find_band_index(max(x_min, x_max - 1e-3), x_lines),
        'core_col_start': _find_band_index(core_left, x_lines),
        'core_col_end': _find_band_index(max(core_left, core_right - 1e-3), x_lines),
    }


def _infer_note_start_col(items: Sequence[OCRItem], x_lines: Sequence[int], schema: Dict[str, Any], body_start: int) -> Optional[int]:
    candidates: List[int] = []
    for item in items:
        text = _normalize_ocr_text(str(item.get('text', '')))
        row = int(item.get('row', 0))
        if '课目注释' in text:
            candidates.append(int(item.get('col_start', item.get('col', 0))))
        if row >= body_start and text in {'SD', 'STP', 'CQY', 'GC', 'MF'}:
            if float(item.get('x_min', 0.0)) >= float(x_lines[max(schema['time_cols'][1] - 2, 0)]):
                candidates.append(int(item.get('col_start', item.get('col', 0))))
    if not candidates:
        return None
    return min(candidates)


def _extract_top_section(items: Sequence[OCRItem], body_start: int) -> List[Dict[str, Any]]:
    anchors: List[Tuple[int, str]] = []
    candidate_rows = range(2, max(2, body_start - 1))
    for row in candidate_rows:
        row_candidates: List[Tuple[float, str]] = []
        for item in items:
            if int(item['row']) != row or int(item['col']) > 2:
                continue
            label = _clean_top_label(normalize_text(str(item['text'])))
            if not label or label in TOP_SECTION_HEADER_TEXTS:
                continue
            if len(label) < 2 or not any('\u4e00' <= ch <= '\u9fff' for ch in label):
                continue
            row_candidates.append((float(item['x']), label))
        if row_candidates:
            row_candidates = sorted(row_candidates, key=lambda d: d[0])
            chosen = row_candidates[0][1]
            if not anchors or anchors[-1][1] != chosen:
                anchors.append((row, chosen))
    if not anchors:
        return []
    top_entries: List[Dict[str, Any]] = []
    sorted_items = sorted(items, key=lambda d: (int(d['row']), float(d['x'])))
    for idx, (anchor_row, label) in enumerate(anchors):
        start_row = max(2, anchor_row - 1) if idx == 0 else anchor_row
        end_row = anchors[idx + 1][0] - 1 if idx + 1 < len(anchors) else body_start - 1
        lines_by_row: List[str] = []
        for row in range(start_row, end_row + 1):
            row_texts = []
            for item in sorted_items:
                if int(item['row']) != row or int(item['col']) <= 2:
                    continue
                text = normalize_text(str(item['text']))
                if text in TOP_SECTION_HEADER_TEXTS:
                    continue
                useful = any('\u4e00' <= ch <= '\u9fff' for ch in text) or len(text) >= 2
                if useful:
                    row_texts.append(text)
            if row_texts:
                lines_by_row.append('；'.join(row_texts))
        top_entries.append({'label': label, 'details': [t for ln in lines_by_row for t in ln.split('；') if t], 'lines': lines_by_row, 'raw_rows': list(range(start_row, end_row + 1))})
    return top_entries


def _type_from_anchors(center_row: float, anchors: Sequence[Tuple[int, str]]) -> str:
    if not anchors:
        return ''
    if len(anchors) == 1:
        return anchors[0][1]
    boundaries = [(float(anchors[i][0]) + float(anchors[i + 1][0])) / 2.0 for i in range(len(anchors) - 1)]
    for i, boundary in enumerate(boundaries):
        if center_row <= boundary:
            return anchors[i][1]
    return anchors[-1][1]


def _item_span_info(item: OCRItem, lexicon: Dict[str, List[str]], aircraft_type: str) -> Dict[str, Any]:
    text = str(item['text'])
    kind = _classify_text(text, lexicon, aircraft_type)
    markers = _extract_markers(text)
    codes = _extract_code_names(text, lexicon, aircraft_type)
    residual = _strip_markers_and_codes(text, lexicon, aircraft_type)
    cs = int(item['core_col_start'])
    ce = int(item['core_col_end'])
    if ce < cs:
        ce = cs
    return {
        'item': item,
        'kind': kind,
        'markers': markers,
        'codes': codes,
        'residual': residual,
        'start_anchor': cs,
        'end_anchor': ce,
    }


def _safe_time_from_col(slot_times: Sequence[str], time_col_start: int, col: int) -> str:
    idx = max(0, min(len(slot_times) - 1, int(col) - int(time_col_start)))
    return slot_times[idx]


class BaseStrategy:
    def __init__(self, lexicon: Dict[str, List[str]], slot_times: Sequence[str], time_col_start: int):
        self.lexicon = lexicon
        self.slot_times = list(slot_times)
        self.time_col_start = int(time_col_start)

    def finalize_event(self, event: Optional[Event]) -> Optional[Event]:
        if event is None:
            return None
        start = _time_to_minutes(str(event['start_time']))
        end = _time_to_minutes(str(event['end_time']))
        if end < start:
            end = start
            event['end_time'] = event['start_time']
        event['display_time'] = event['start_time'] if start == end else f"{event['start_time']}~{event['end_time']}"
        return event

    def finalize_group(self, events: Sequence[Event], aircraft_type: str) -> List[Event]:
        return sorted([ev for ev in events if ev], key=lambda e: (_time_to_minutes(str(e['start_time'])), _time_to_minutes(str(e['end_time']))))


class XX5Strategy(BaseStrategy):
    def _top_band_items(self, items: Sequence[OCRItem]) -> List[OCRItem]:
        if not items:
            return []
        y_min = min(float(it['y_min']) for it in items)
        y_max = max(float(it['y_max']) for it in items)
        cutoff = y_min + 0.72 * max(1.0, y_max - y_min)
        kept: List[OCRItem] = []
        for it in items:
            text = _normalize_ocr_text(str(it['text']))
            if float(it['y_center']) <= cutoff:
                kept.append(it)
            elif _looks_like_code_name(text) or any(m in text for m in ['SD', 'CQY']):
                kept.append(it)
        return kept

    def row_events(self, items: Sequence[OCRItem], row_code: str, aircraft_type: str) -> List[Event]:
        items = self._top_band_items(items)
        infos = [_item_span_info(it, self.lexicon, aircraft_type) for it in sorted(items, key=lambda z: (int(z['core_col_start']), float(z['x_min'])))]
        infos = [i for i in infos if i['kind'] not in {'noise'}]
        if not infos:
            return []

        events: List[Event] = []
        i = 0
        n = len(infos)
        while i < n:
            info = infos[i]
            if info['kind'] == 'weak' and not info['markers'] and not info['codes']:
                i += 1
                continue
            seg = [info]
            gap_limit = 10
            has_marker = bool(info['markers'])
            has_code = bool(info['codes'])
            j = i + 1

            if has_marker and not has_code:
                # marker-led event: absorb desc until first code, then close soon
                while j < n:
                    nxt = infos[j]
                    gap = int(nxt['start_anchor']) - int(seg[-1]['end_anchor'])
                    if gap > gap_limit or nxt['markers']:
                        break
                    if nxt['codes'] and len(seg) >= 2 and gap > 6:
                        break
                    seg.append(nxt)
                    if nxt['codes']:
                        j += 1
                        while j < n:
                            nxt2 = infos[j]
                            gap2 = int(nxt2['start_anchor']) - int(seg[-1]['end_anchor'])
                            if gap2 > 1 or nxt2['markers'] or nxt2['codes']:
                                break
                            if nxt2['kind'] == 'weak':
                                break
                            seg.append(nxt2)
                            j += 1
                        break
                    j += 1
                events.append(self.finalize_event(self.build_event(seg, row_code)))
                i = j
                continue

            if has_code or row_code:
                # code-led or row-code-led desc event; allow one far-right explicit code to close sparse span
                while j < n:
                    nxt = infos[j]
                    gap = int(nxt['start_anchor']) - int(seg[-1]['end_anchor'])
                    if nxt['markers']:
                        break
                    if nxt['codes']:
                        if gap > 12:
                            break
                        seg.append(nxt)
                        j += 1
                        while j < n:
                            nxt2 = infos[j]
                            gap2 = int(nxt2['start_anchor']) - int(seg[-1]['end_anchor'])
                            if gap2 > 1 or nxt2['markers'] or nxt2['codes']:
                                break
                            if nxt2['kind'] == 'weak':
                                break
                            seg.append(nxt2)
                            j += 1
                        break
                    if gap > 3:
                        break
                    if nxt['kind'] == 'weak':
                        break
                    seg.append(nxt)
                    j += 1
                events.append(self.finalize_event(self.build_event(seg, row_code)))
                i = j
                continue

            i += 1

        return [ev for ev in self._postprocess_row(events, row_code) if ev is not None]

    def build_event(self, segment: Sequence[Dict[str, Any]], row_code: str) -> Optional[Event]:
        if not segment:
            return None
        markers = _dedupe_preserve_order([m for info in segment for m in info['markers']])
        codes = _dedupe_preserve_order([c for info in segment for c in info['codes']])
        residuals = _dedupe_preserve_order([
            normalize_text(str(info['residual']))
            for info in segment
            if normalize_text(str(info['residual'])) and normalize_text(str(info['residual'])) not in WEAK_NOISE_TEXTS
        ])
        marker = markers[0] if markers else ''
        if marker == 'STP':
            return None
        if marker and not codes and not residuals:
            return None
        strong_infos = [info for info in segment if info['kind'] not in {'weak', 'noise'}]
        if not strong_infos:
            return None
        start_col = min(int(info['start_anchor']) for info in strong_infos)
        if codes:
            end_col = max(int(info['end_anchor']) for info in segment if info['codes'])
            pilot_codes = [codes[-1]]
        else:
            end_col = max(int(info['end_anchor']) for info in strong_infos)
            pilot_codes = [row_code] if row_code else []

        remark_tokens: List[str] = []
        if marker:
            remark_tokens.append(marker)
        for r in residuals:
            if r in {'SP', 'STP', '-', '5o', '5O'}:
                continue
            remark_tokens.append(r)
        remark = ' '.join(_dedupe_preserve_order(remark_tokens)).strip()
        if not marker and not pilot_codes and not remark:
            return None
        return {
            'start_time': _safe_time_from_col(self.slot_times, self.time_col_start, start_col),
            'end_time': _safe_time_from_col(self.slot_times, self.time_col_start, end_col),
            'display_time': '',
            'text': remark,
            'remark': remark,
            'pilot_codes': _dedupe_preserve_order(pilot_codes),
            'pilot_code': '、'.join(_dedupe_preserve_order(pilot_codes)),
            'flight_code': marker,
            'source_cols': sorted({int(i['item']['core_col_start']) for i in segment} | {int(i['item']['core_col_end']) for i in segment}),
            'source_row': int(segment[0]['item']['row']),
        }

    def _postprocess_row(self, events: Sequence[Optional[Event]], row_code: str) -> List[Event]:
        out: List[Event] = []
        for ev in sorted([e for e in events if e], key=lambda e: _time_to_minutes(str(e['start_time']))):
            marker = normalize_text(str(ev.get('flight_code', '')))
            remark = normalize_text(str(ev.get('remark', '')))
            start = _time_to_minutes(str(ev.get('start_time', '')))
            if marker == 'STP':
                continue
            if start >= _time_to_minutes('21:30') and marker in {'SD', 'CQY', 'GC', 'MF'} and remark in {'', marker, 'SP', 'STP'}:
                continue
            if marker and remark == marker and (_time_to_minutes(str(ev.get('end_time', ''))) - start) <= 10:
                continue
            if not marker and remark in {'-', '5o', '5O'} and start < _time_to_minutes('19:20'):
                continue
            # code-only tiny/weak event should be removed, but keep late closing code events
            if not marker and len(ev.get('pilot_codes', [])) == 1 and remark in {'', '-', '5o', '5O'} and start < _time_to_minutes('19:20'):
                continue
            out.append(ev)
        return out

    def finalize_group(self, events: Sequence[Event], aircraft_type: str) -> List[Event]:
        return [
            ev for ev in sorted(events, key=lambda e: (_time_to_minutes(str(e['start_time'])), _time_to_minutes(str(e['end_time']))))
            if normalize_text(str(ev.get('flight_code', ''))) != 'STP'
        ]


class XXSStrategy(BaseStrategy):
    def row_events(self, items: Sequence[OCRItem], row_code: str, aircraft_type: str) -> List[Event]:
        infos = [_item_span_info(it, self.lexicon, aircraft_type) for it in sorted(items, key=lambda z: (int(z['core_col_start']), float(z['x_min'])))]
        infos = [i for i in infos if i['kind'] not in {'noise'}]
        if not infos:
            return []
        segments: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for info in infos:
            if not current:
                current = [info]
                continue
            gap = int(info['start_anchor']) - int(current[-1]['end_anchor'])
            split = gap > 4
            if not split and info['kind'].startswith('marker') and any(seg['kind'].startswith('marker') for seg in current):
                split = True
            if split:
                segments.append(current)
                current = [info]
            else:
                current.append(info)
        if current:
            segments.append(current)
        events: List[Event] = []
        for seg in segments:
            ev = self.build_event(seg, row_code)
            ev = self.finalize_event(ev)
            if ev is not None:
                events.append(ev)
        return events

    def build_event(self, segment: Sequence[Dict[str, Any]], row_code: str) -> Optional[Event]:
        markers = _dedupe_preserve_order([m for info in segment for m in info['markers']])
        codes = _dedupe_preserve_order([c for info in segment for c in info['codes']])
        residuals = _dedupe_preserve_order([
            normalize_text(str(info['residual']))
            for info in segment
            if normalize_text(str(info['residual'])) and normalize_text(str(info['residual'])) not in WEAK_NOISE_TEXTS
        ])
        start_col = min(int(info['start_anchor']) for info in segment)
        end_col = max(int(info['end_anchor']) for info in segment)
        marker = ''
        for m in ['MF', 'GC', 'CC', 'CQY', 'SD']:
            if m in markers:
                marker = m
                break
        if not codes and row_code and (marker or any(r in PAIR_HINT_TOKENS for r in residuals)):
            codes = [row_code]
        if not codes and not marker and not residuals:
            return None
        if len(codes) > 2:
            codes = codes[-2:]
        remark_tokens: List[str] = []
        if marker in {'MF', 'GC'}:
            remark_tokens.append('MF(伴航)')
        elif marker:
            remark_tokens.append(marker)
        for r in residuals:
            if r in {'MF(伴航)', 'MF（伴航）'} and 'MF(伴航)' in remark_tokens:
                continue
            if r in {'414', '44', '103', '0.55T', '0.65T'} or '伴航' in r:
                remark_tokens.append(r)
        if not remark_tokens:
            remark_tokens.extend(residuals)
        return {
            'start_time': _safe_time_from_col(self.slot_times, self.time_col_start, start_col),
            'end_time': _safe_time_from_col(self.slot_times, self.time_col_start, end_col),
            'display_time': '',
            'text': ' '.join(_dedupe_preserve_order(remark_tokens)).strip(),
            'remark': ' '.join(_dedupe_preserve_order(remark_tokens)).strip(),
            'pilot_codes': _dedupe_preserve_order(codes),
            'pilot_code': '、'.join(_dedupe_preserve_order(codes)),
            'flight_code': marker,
            'source_cols': sorted({int(i['item']['core_col_start']) for i in segment} | {int(i['item']['core_col_end']) for i in segment}),
            'source_row': int(segment[0]['item']['row']),
        }

    def finalize_group(self, events: Sequence[Event], aircraft_type: str) -> List[Event]:
        ordered = sorted(events, key=lambda e: (_time_to_minutes(str(e['start_time'])), _time_to_minutes(str(e['end_time']))))
        merged: List[Event] = []
        for ev in ordered:
            if not merged:
                merged.append(dict(ev))
                continue
            prev = merged[-1]
            gap = _time_to_minutes(str(ev['start_time'])) - _time_to_minutes(str(prev['end_time']))
            pairish = any(tok in str(prev.get('remark', '')) for tok in ['MF', '0.55T', '0.65T', '103']) or any(tok in str(ev.get('remark', '')) for tok in ['MF', '0.55T', '0.65T', '103'])
            prev_codes = set(prev.get('pilot_codes', []))
            cur_codes = set(ev.get('pilot_codes', []))
            if gap <= 30 and pairish and len(prev_codes | cur_codes) <= 2:
                prev['pilot_codes'] = _dedupe_preserve_order(list(prev_codes | cur_codes))
                prev['pilot_code'] = '、'.join(prev['pilot_codes'])
                prev['start_time'] = _minutes_to_time(min(_time_to_minutes(str(prev['start_time'])), _time_to_minutes(str(ev['start_time']))))
                prev['end_time'] = _minutes_to_time(max(_time_to_minutes(str(prev['end_time'])), _time_to_minutes(str(ev['end_time']))))
                prev['display_time'] = prev['start_time'] if prev['start_time'] == prev['end_time'] else f"{prev['start_time']}~{prev['end_time']}"
                prev['remark'] = ' '.join(_dedupe_preserve_order([str(prev.get('remark', '')), str(ev.get('remark', ''))])).strip()
                prev['text'] = prev['remark']
            else:
                merged.append(dict(ev))
        out = []
        for ev in merged:
            remark = normalize_text(str(ev.get('remark', '')))
            if remark in {'', '03'} and len(ev.get('pilot_codes', [])) <= 1:
                continue
            out.append(ev)
        return out


class XXAStrategy(BaseStrategy):
    def row_events(self, items: Sequence[OCRItem], row_code: str, aircraft_type: str) -> List[Event]:
        infos = [_item_span_info(it, self.lexicon, aircraft_type) for it in sorted(items, key=lambda z: (int(z['core_col_start']), float(z['x_min'])))]
        infos = [i for i in infos if i['kind'] not in {'noise'}]
        if not infos:
            return []
        ev = self.build_event(infos, row_code)
        ev = self.finalize_event(ev)
        return [ev] if ev is not None else []

    def build_event(self, segment: Sequence[Dict[str, Any]], row_code: str) -> Optional[Event]:
        markers = _dedupe_preserve_order([m for info in segment for m in info['markers']])
        codes = _dedupe_preserve_order([c for info in segment for c in info['codes']])
        if row_code and row_code not in codes:
            codes.append(row_code)
        if len(codes) > 2:
            codes = codes[-2:]
        if not codes and not markers:
            return None
        strong = [info for info in segment if info['kind'] not in {'noise'}]
        start_col = min(int(info['start_anchor']) for info in strong)
        # XXA 允许稀疏空白，不只看 code token 结束
        end_col = max(int(info['end_anchor']) for info in strong)
        marker = ''
        for m in ['CC', 'GC', 'MF']:
            if m in markers:
                marker = m
                break
        residuals = _dedupe_preserve_order([
            normalize_text(str(info['residual']))
            for info in segment
            if normalize_text(str(info['residual'])) and normalize_text(str(info['residual'])) not in XXA_DROP_REMARKS
        ])
        remark_tokens: List[str] = []
        if marker:
            remark_tokens.append(marker)
        for r in residuals:
            if r in WEAK_NOISE_TEXTS:
                continue
            remark_tokens.append(r)
        remark = ' '.join(_dedupe_preserve_order(remark_tokens)).strip()
        return {
            'start_time': _safe_time_from_col(self.slot_times, self.time_col_start, start_col),
            'end_time': _safe_time_from_col(self.slot_times, self.time_col_start, end_col),
            'display_time': '',
            'text': remark,
            'remark': remark,
            'pilot_codes': codes,
            'pilot_code': '、'.join(codes),
            'flight_code': marker,
            'source_cols': sorted({int(i['item']['core_col_start']) for i in segment} | {int(i['item']['core_col_end']) for i in segment}),
            'source_row': int(segment[0]['item']['row']),
        }

    def finalize_group(self, events: Sequence[Event], aircraft_type: str) -> List[Event]:
        ordered = sorted(events, key=lambda e: (_time_to_minutes(str(e['start_time'])), _time_to_minutes(str(e['end_time']))))
        if not ordered:
            return []
        best = max(
            ordered,
            key=lambda ev: (
                len(ev.get('pilot_codes', [])),
                _time_to_minutes(str(ev['end_time'])) - _time_to_minutes(str(ev['start_time'])),
                bool(ev.get('flight_code')),
            ),
        )
        return [best]


def _strategy_for_aircraft(aircraft_type: str, lexicon: Dict[str, List[str]], slot_times: Sequence[str], time_col_start: int) -> BaseStrategy:
    if aircraft_type == 'XX5':
        return XX5Strategy(lexicon, slot_times, time_col_start)
    if aircraft_type == 'XXS':
        return XXSStrategy(lexicon, slot_times, time_col_start)
    return XXAStrategy(lexicon, slot_times, time_col_start)


def extract_structured_main_table(
    main_table_img_path: str | Path,
    config: Dict[str, Any],
    engine: Any,
    lexicon: Dict[str, List[str]],
) -> Dict[str, Any]:
    main_table_img = read_image(main_table_img_path)
    ocr_lines = engine.ocr_region(main_table_img, preprocess=False)

    x_lines = config['grid']['x_lines']
    y_lines = config['grid']['y_lines']
    schema = config['semantic']['main_table_schema']
    slot_times = _build_slot_times(schema['time_cols'][1] - schema['time_cols'][0] + 1)

    items: List[OCRItem] = []
    for line in ocr_lines:
        item = _build_item_from_line(line, x_lines, y_lines)
        if item['text']:
            items.append(item)

    body_start = int(schema['body_start_row'])
    body_end = len(y_lines) - 2
    note_start_col = _infer_note_start_col(items, x_lines, schema, body_start)
    cell_w = max(1.0, float(x_lines[schema['time_cols'][0] + 1] - x_lines[schema['time_cols'][0]]))
    time_grid_end_col = int(schema['time_cols'][1])
    time_grid_end_x = float(x_lines[min(len(x_lines) - 1, time_grid_end_col + 1)])
    if note_start_col is not None:
        note_boundary_x = float(x_lines[min(len(x_lines) - 1, note_start_col)])
        event_right_x = min(time_grid_end_x, note_boundary_x - 0.8 * cell_w)
    else:
        event_right_x = time_grid_end_x - 0.2 * cell_w

    top_section = _extract_top_section(items, body_start)
    row_items: Dict[int, List[OCRItem]] = {}
    for row_idx in range(body_start, body_end + 1):
        row_items[row_idx] = sorted([item for item in items if int(item['row']) == row_idx], key=lambda it: float(it['x']))

    type_anchors: List[Tuple[int, str]] = []
    for row_idx in range(body_start, body_end + 1):
        texts = [_clean_aircraft_type(item['text']) for item in row_items[row_idx] if int(item['col']) == 0]
        texts = [text for text in texts if text.startswith('XX')]
        if texts:
            type_anchors.append((row_idx, texts[0]))

    body_rows: List[Dict[str, Any]] = []
    structured_records: List[Dict[str, Any]] = []
    group_index = 0

    for group_start in range(body_start, body_end + 1, 3):
        group_rows = [row for row in range(group_start, min(group_start + 3, body_end + 1))]
        if not group_rows:
            continue
        group_index += 1
        group_mid = group_rows[min(1, len(group_rows) - 1)]
        aircraft_type = _type_from_anchors(group_mid, type_anchors)
        strategy = _strategy_for_aircraft(aircraft_type, lexicon, slot_times, schema['time_cols'][0])

        aircraft_no = ''
        secondary_code = ''
        for probe_row in [group_mid] + [row for row in group_rows if row != group_mid]:
            for item in row_items.get(probe_row, []):
                if int(item['col']) == 1 and not aircraft_no:
                    aircraft_no = _clean_aircraft_no(item['text'], lexicon)
                if int(item['col']) == 2 and not secondary_code:
                    secondary_code = _clean_secondary_code(item['text'], lexicon)
        if not aircraft_no and secondary_code in SECONDARY_TO_AIRCRAFT_NO:
            aircraft_no = SECONDARY_TO_AIRCRAFT_NO[secondary_code]
        if not aircraft_no and 0 <= group_index - 1 < len(GROUP_AIRCRAFT_FALLBACK):
            aircraft_no = GROUP_AIRCRAFT_FALLBACK[group_index - 1]
        if not secondary_code and 0 <= group_index - 1 < len(GROUP_SECONDARY_FALLBACK):
            secondary_code = GROUP_SECONDARY_FALLBACK[group_index - 1]

        crew_rows: List[Dict[str, Any]] = []
        group_events: List[Event] = []
        for row_idx in group_rows:
            current_items = row_items.get(row_idx, [])
            code_texts = [_clean_code_name(item['text'], lexicon) for item in current_items if int(item['col']) == 74]
            code_texts = [text for text in code_texts if _looks_like_code_name(text)]
            code_name = next((text for text in code_texts if text), '')

            event_items = []
            for item in current_items:
                text = normalize_text(str(item['text']))
                if text in IGNORE_EVENT_TEXTS:
                    continue
                is_code_name = _looks_like_code_name(text)
                if float(item.get('x_center', item.get('x', 0.0))) >= event_right_x:
                    if not (aircraft_type == 'XX5' and is_code_name and float(item.get('x_center', item.get('x', 0.0))) < event_right_x + 1.2 * cell_w):
                        continue
                if float(item.get('x_max', 0.0)) >= event_right_x + 0.25 * cell_w:
                    if not (aircraft_type == 'XX5' and is_code_name and float(item.get('x_max', 0.0)) < event_right_x + 1.4 * cell_w):
                        continue
                if int(item['core_col_start']) < schema['time_cols'][0]:
                    continue
                if int(item['core_col_start']) > time_grid_end_col + (1 if aircraft_type == 'XX5' and is_code_name else 0):
                    continue
                event_items.append(item)

            row_events = strategy.row_events(event_items, code_name, aircraft_type)
            group_events.extend(row_events)

            if not code_name:
                strong_codes = _dedupe_preserve_order([code for ev in row_events for code in ev.get('pilot_codes', []) if code])
                if strong_codes:
                    code_name = strong_codes[0]

            if not code_name and not row_events:
                continue
            crew_row = {
                'grid_row': row_idx,
                'aircraft_type': aircraft_type,
                'aircraft_no': aircraft_no,
                'secondary_code': secondary_code,
                'name': '',
                'code_name': code_name,
                'events': row_events,
                'source_group': group_index,
            }
            crew_rows.append(crew_row)
            body_rows.append(crew_row)

        if not crew_rows:
            continue

        crew_codes = _dedupe_preserve_order([
            str(crew_row.get('code_name', '') or '')
            for crew_row in crew_rows
            if _looks_like_code_name(str(crew_row.get('code_name', '') or ''))
        ])
        group_events = strategy.finalize_group(group_events, aircraft_type)
        group_events = sorted(group_events, key=lambda event: (
            _time_to_minutes(str(event.get('start_time', '') or '')),
            _time_to_minutes(str(event.get('end_time', '') or '')),
            int(event.get('source_row', 0)),
        ))

        structured_records.append({
            'group_index': group_index,
            'aircraft_type': aircraft_type,
            'aircraft_no': aircraft_no,
            'secondary_code': secondary_code,
            'crew_codes': crew_codes,
            'crew_rows': crew_rows,
            'events': group_events,
        })

    return {
        'top_section': top_section,
        'slot_times': slot_times,
        'body_rows': body_rows,
        'structured_records': structured_records,
        'note_start_col': note_start_col,
        'event_right_x': event_right_x,
        'raw_ocr_lines': [
            {
                'row': int(item['row']),
                'col': int(item['col']),
                'col_start': int(item['col_start']),
                'col_end': int(item['col_end']),
                'core_col_start': int(item['core_col_start']),
                'core_col_end': int(item['core_col_end']),
                'x': float(item['x']),
                'x_min': float(item['x_min']),
                'x_max': float(item['x_max']),
                'y': float(item['y']),
                'text': str(item['text']),
                'score': float(item['score']),
            }
            for item in items
        ],
    }


__all__ = ['extract_structured_main_table', 'XX5Strategy', 'XXSStrategy', 'XXAStrategy']
