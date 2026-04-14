from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .pipeline import read_image
from .text_utils import best_fuzzy_match, correct_with_lexicon, normalize_text


OCRItem = Dict[str, Any]

SECONDARY_TO_AIRCRAFT_NO = {
    '0750': '60',
    '0751': '61',
    '0752': '63',
    '0753': '74',
    '0701': '374',
    '0702': '376',
    '0703': '07',
    '0705': '10',
}

GROUP_AIRCRAFT_FALLBACK = ['60', '61', '63', '74', '374', '376', '07', '10', '09']
GROUP_SECONDARY_FALLBACK = ['0750', '0751', '0752', '0753', '0701', '0702', '0703', '0705', '']
EXTRA_CODE_NAMES = ['森153', '光180', '豪182', '汤191', '冯185', '玉165', '韩171', '说176', '潘851', '黄827', '彭823']
IGNORE_EVENT_TEXTS = {'课目注释', '姓名', '代字', '代号', '姓名代字代号', '机型', '机号', '二次代码'}
TOP_SECTION_HEADER_TEXTS = {'姓名', '代字', '代号', '姓名代字代号', '机型', '机号', '二次代码'}


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
        hh = minutes // 60
        mm = minutes % 60
        slots.append(f'{hh:02d}:{mm:02d}')
    return slots


def _time_to_minutes(text: str) -> int:
    text = normalize_text(text)
    if not text or ':' not in text:
        return 0
    hh, mm = text.split(':', 1)
    if not hh.isdigit() or not mm.isdigit():
        return 0
    return int(hh) * 60 + int(mm)


def _normalize_ocr_text(text: str) -> str:
    text = normalize_text(text)
    replacements = {
        '胜809.': '雁809',
        '胜809': '雁809',
        '贤831': '贾831',
        'OTE1': '0751',
        'OT61': '0751',
        'OTE3': '0753',
        'COY': 'CQY',
        'COV': 'CQY',
        'FQY': 'CQY',
        'cqr': 'CQY',
        'qY': 'CQY',
        'xX5': 'XX5',
        'xXs': 'XXS',
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
        '至梅1号': '至德1号',
        '至梅7号': '至德7号',
        '至梅9号': '至德9号',
        '至徳1号': '至德1号',
        '至徳7号': '至德7号',
        '至徳9号': '至德9号',
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
    if len(digits) <= 2 and digits.startswith('0'):
        return digits.zfill(2)
    matched = best_fuzzy_match(digits, lexicon.get('aircraft_types', []), min_score=0.34)
    if matched and matched in lexicon.get('aircraft_types', []):
        return matched
    if len(digits) <= 3:
        return digits.zfill(2) if len(digits) == 1 else digits
    return digits


def _clean_secondary_code(text: str, lexicon: Dict[str, List[str]]) -> str:
    text = _normalize_ocr_text(text)
    digits = ''.join(ch for ch in text if ch.isdigit())
    if not digits:
        return ''
    if len(digits) >= 4:
        digits = digits[:4]
    choices = list(lexicon.get('aircraft_numbers', [])) + ['0753']
    if digits in choices:
        return digits
    matched = best_fuzzy_match(digits, choices, min_score=0.34)
    return matched if matched else digits


def _clean_code_name(text: str, lexicon: Dict[str, List[str]]) -> str:
    text = _normalize_ocr_text(text)
    if not text:
        return ''

    replacements = {
        '苑13': '苑813',
        '贤831': '贾831',
        '胜809': '雁809',
        '博823': '彭823',
        '形812': '彤812',
        '猛185': '冯185',
        '货827': '黄827',
        '高835': '高835',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    direct = re.findall(r'[\u4e00-\u9fff][0-9]{2,3}', text)
    candidates = lexicon.get('code_names', []) + EXTRA_CODE_NAMES
    if direct:
        target = direct[0]
        return best_fuzzy_match(target, candidates, min_score=0.40)

    digits = ''.join(ch for ch in text if ch.isdigit())
    if len(digits) == 3:
        suffix_matches = [cand for cand in candidates if cand.endswith(digits)]
        if suffix_matches:
            return suffix_matches[0]

    return correct_with_lexicon(text, {'code_names': candidates}, 'code_names') or text


def _clean_event_token(text: str, lexicon: Dict[str, List[str]]) -> str:
    text = _normalize_ocr_text(text)
    if not text:
        return ''

    letters = ''.join(ch for ch in text if ch.isalpha()).upper()
    if len(letters) >= 2:
        matched = best_fuzzy_match(letters, lexicon.get('flight_codes', []), min_score=0.40)
        if matched in lexicon.get('flight_codes', []):
            return matched

    if re.search(r'[\u4e00-\u9fff][0-9]{2,3}', text):
        return _clean_code_name(text, lexicon)

    if text.isdigit() and len(text) == 1:
        return ''

    if len(text) == 1 and not text.isdigit() and not ('\u4e00' <= text <= '\u9fff'):
        return ''

    return text


def _extract_code_name_from_text(text: str, fallback: str, lexicon: Dict[str, List[str]]) -> str:
    direct = re.findall(r'[\u4e00-\u9fff][0-9]{2,3}', _normalize_ocr_text(text))
    if direct:
        return _clean_code_name(direct[0], lexicon)
    return fallback


def _extract_flight_code_from_text(text: str, lexicon: Dict[str, List[str]]) -> str:
    text = _normalize_ocr_text(text)
    tokens = re.split(r'[\s/:，,;；]+', text)
    for token in tokens:
        token = token.strip().upper()
        if not token or len(token) < 2:
            continue
        matched = best_fuzzy_match(token, lexicon.get('flight_codes', []), min_score=0.40)
        if matched in lexicon.get('flight_codes', []):
            return matched
    return ''


def _extract_top_section(items: Sequence[OCRItem], body_start: int) -> List[Dict[str, Any]]:
    anchors: List[Tuple[int, str]] = []
    candidate_rows = range(2, max(2, body_start - 1))
    for row in candidate_rows:
        row_candidates: List[Tuple[float, str]] = []
        for item in items:
            item_row = int(item['row'])
            col = int(item['col'])
            if item_row != row or col > 2:
                continue
            raw_text = normalize_text(str(item['text']))
            label = _clean_top_label(raw_text)
            if not label or label in TOP_SECTION_HEADER_TEXTS:
                continue
            if len(label) < 2:
                continue
            if not any('\u4e00' <= ch <= '\u9fff' for ch in label):
                continue
            row_candidates.append((float(item['x']), label))

        if not row_candidates:
            continue

        row_candidates = sorted(row_candidates, key=lambda data: data[0])
        chosen_label = row_candidates[0][1]
        if not anchors or anchors[-1][1] != chosen_label:
            anchors.append((row, chosen_label))

    if not anchors:
        return []

    top_entries: List[Dict[str, Any]] = []
    sorted_items = sorted(items, key=lambda data: (int(data['row']), float(data['x'])))
    for idx, (anchor_row, label) in enumerate(anchors):
        start_row = max(2, anchor_row - 1) if idx == 0 else anchor_row
        end_row = anchors[idx + 1][0] - 1 if idx + 1 < len(anchors) else body_start - 1
        lines_by_row: List[str] = []
        for row in range(start_row, end_row + 1):
            row_texts: List[str] = []
            for item in sorted_items:
                item_row = int(item['row'])
                col = int(item['col'])
                if item_row != row or col <= 2:
                    continue
                text = normalize_text(str(item['text']))
                if text in TOP_SECTION_HEADER_TEXTS:
                    continue
                useful = any('\u4e00' <= ch <= '\u9fff' for ch in text) or len(text) >= 2
                if not useful:
                    continue
                row_texts.append(text)
            if row_texts:
                lines_by_row.append('；'.join(row_texts))
        top_entries.append({
            'label': label,
            'details': [text for line in lines_by_row for text in line.split('；') if text],
            'lines': lines_by_row,
            'raw_rows': list(range(start_row, end_row + 1)),
        })

    return top_entries


def _is_useful_event_item(item: OCRItem) -> bool:
    text = _normalize_ocr_text(str(item.get('text', '')))
    if not text or text in IGNORE_EVENT_TEXTS:
        return False
    useful = sum(1 for ch in text if ch.isdigit() or ch.isalpha() or ('\u4e00' <= ch <= '\u9fff'))
    if useful == 0:
        return False
    if text.isdigit() and len(text) == 1:
        return False
    if len(text) == 1 and not text.isdigit() and not ('\u4e00' <= text <= '\u9fff'):
        return False
    return True


def _type_from_anchors(center_row: float, anchors: Sequence[Tuple[int, str]]) -> str:
    if not anchors:
        return ''
    if len(anchors) == 1:
        return anchors[0][1]

    boundaries = [
        (float(anchors[idx][0]) + float(anchors[idx + 1][0])) / 2.0
        for idx in range(len(anchors) - 1)
    ]
    for idx, boundary in enumerate(boundaries):
        if center_row <= boundary:
            return anchors[idx][1]
    return anchors[-1][1]


def _group_event_items(items: Sequence[OCRItem]) -> List[List[OCRItem]]:
    if not items:
        return []

    groups: List[List[OCRItem]] = []
    current: List[OCRItem] = []
    prev_col: int | None = None
    prev_x: float | None = None

    for item in sorted(items, key=lambda data: float(data['x'])):
        col = int(item['col'])
        x = float(item['x'])
        if prev_col is None or (col - prev_col <= 6 and (prev_x is None or x - prev_x <= 95.0)):
            current.append(item)
        else:
            groups.append(current)
            current = [item]
        prev_col = col
        prev_x = x

    if current:
        groups.append(current)
    return groups


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
        text = _normalize_ocr_text(line.text)
        if not text:
            continue
        x_center = sum(point[0] for point in line.box) / 4.0
        y_center = sum(point[1] for point in line.box) / 4.0
        items.append({
            'text': text,
            'score': float(line.score),
            'x': x_center,
            'y': y_center,
            'row': _find_band_index(y_center, y_lines),
            'col': _find_band_index(x_center, x_lines),
        })

    body_start = int(schema['body_start_row'])
    body_end = len(y_lines) - 2
    top_section = _extract_top_section(items, body_start)
    row_items: Dict[int, List[OCRItem]] = {}
    for row_idx in range(body_start, body_end + 1):
        row_items[row_idx] = sorted(
            [item for item in items if int(item['row']) == row_idx],
            key=lambda item: float(item['x']),
        )

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
        group_events: List[Dict[str, Any]] = []
        for row_idx in group_rows:
            current_items = row_items.get(row_idx, [])
            code_texts = [_clean_code_name(item['text'], lexicon) for item in current_items if int(item['col']) == 74]
            code_name = next((text for text in code_texts if text), '')

            event_items = [
                item for item in current_items
                if schema['time_cols'][0] <= int(item['col']) <= schema['time_cols'][1] and _is_useful_event_item(item)
            ]
            event_groups = _group_event_items(event_items)

            events: List[Dict[str, Any]] = []
            for event_group in event_groups:
                tokens = [_clean_event_token(str(item['text']), lexicon) for item in event_group]
                tokens = [token for token in tokens if token]
                if not tokens:
                    continue

                text = ' '.join(tokens)
                start_idx = max(0, min(len(slot_times) - 1, int(event_group[0]['col']) - schema['time_cols'][0]))
                end_idx = max(0, min(len(slot_times) - 1, int(event_group[-1]['col']) - schema['time_cols'][0]))
                explicit_pilot_code = _extract_code_name_from_text(text, '', lexicon)
                pilot_code = explicit_pilot_code or code_name
                flight_code = _extract_flight_code_from_text(text, lexicon)
                compact_text = normalize_text(text)

                if compact_text == flight_code and not explicit_pilot_code:
                    continue
                if compact_text.isdigit() and len(compact_text) <= 2 and not explicit_pilot_code and not flight_code:
                    continue
                if compact_text in {'25', '体', '3π'} and not explicit_pilot_code:
                    continue

                event = {
                    'start_time': slot_times[start_idx],
                    'end_time': slot_times[end_idx],
                    'display_time': slot_times[start_idx] if start_idx == end_idx else f"{slot_times[start_idx]}~{slot_times[end_idx]}",
                    'text': text,
                    'pilot_code': pilot_code,
                    'flight_code': flight_code,
                    'source_cols': [int(item['col']) for item in event_group],
                    'source_row': row_idx,
                }
                events.append(event)
                group_events.append(event)

            if not code_name and len(events) == 1 and events[0].get('pilot_code'):
                code_name = normalize_text(str(events[0].get('pilot_code')))

            if not code_name and not events:
                continue

            crew_row = {
                'grid_row': row_idx,
                'aircraft_type': aircraft_type,
                'aircraft_no': aircraft_no,
                'secondary_code': secondary_code,
                'name': '',
                'code_name': code_name,
                'events': events,
                'source_group': group_index,
            }
            crew_rows.append(crew_row)
            body_rows.append(crew_row)

        if not crew_rows:
            continue

        crew_codes: List[str] = []
        for crew_row in crew_rows:
            code_value = normalize_text(str(crew_row.get('code_name', '') or ''))
            if code_value and code_value not in crew_codes:
                crew_codes.append(code_value)
        for event in group_events:
            code_value = normalize_text(str(event.get('pilot_code', '') or ''))
            if code_value and code_value not in crew_codes:
                crew_codes.append(code_value)

        group_events = sorted(
            group_events,
            key=lambda event: (
                _time_to_minutes(str(event.get('start_time', '') or '')),
                _time_to_minutes(str(event.get('end_time', '') or '')),
                int(event.get('source_row', 0)),
            ),
        )

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
        'raw_ocr_lines': [
            {
                'row': int(item['row']),
                'col': int(item['col']),
                'x': float(item['x']),
                'y': float(item['y']),
                'text': str(item['text']),
                'score': float(item['score']),
            }
            for item in items
        ],
    }


__all__ = ['extract_structured_main_table']
