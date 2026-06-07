from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from table_ocr_project.semantic_extractors import (
    STRICT_OCR_CANDIDATE_MODE,
    available_ocr_candidate_modes,
)
from table_ocr_project.structured_process import run_process_form_workflow


TITLE_KEYS = ['confidentiality', 'date', 'title', 'approved_name', 'astronomical_times']
REMARK_KEYS = ['title', 'training_headers', 'training_entries', 'total', 'occupancy_time']
BOTTOM_KEYS = ['line1', 'line2']
MAIN_TABLE_KEYS = [
    'top_section',
    'slot_times',
    'body_rows',
    'structured_records',
    'note_start_col',
    'event_right_x',
]


def _pick(data: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    return {key: data.get(key) for key in keys}


def semantic_projection(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'title': _pick(result.get('title', {}) or {}, TITLE_KEYS),
        'remark': _pick(result.get('remark', {}) or {}, REMARK_KEYS),
        'bottom': _pick(result.get('bottom', {}) or {}, BOTTOM_KEYS),
        'main_table': _pick(result.get('main_table', {}) or {}, MAIN_TABLE_KEYS),
    }


def _local_name(tag: str) -> str:
    return tag.rsplit('}', 1)[-1] if '}' in tag else tag


def xml_visible_projection(path: Path) -> List[List[Dict[str, str]]]:
    root = ET.parse(path).getroot()
    rows: List[List[Dict[str, str]]] = []
    for row in root.iter():
        if _local_name(row.tag) != 'Row':
            continue
        out_row: List[Dict[str, str]] = []
        for cell in row:
            if _local_name(cell.tag) != 'Cell':
                continue
            text = ''
            for child in cell:
                if _local_name(child.tag) == 'Data':
                    text = ''.join(child.itertext())
                    break
            out_row.append({
                'index': cell.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Index', ''),
                'merge_across': cell.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}MergeAcross', ''),
                'text': text,
            })
        rows.append(out_row)
    return rows


def first_diff(a: Any, b: Any, path: str = '$') -> str:
    if type(a) is not type(b):
        return f'{path}: type {type(a).__name__} != {type(b).__name__}'
    if isinstance(a, dict):
        keys = sorted(set(a) | set(b))
        for key in keys:
            if key not in a:
                return f'{path}.{key}: missing in baseline'
            if key not in b:
                return f'{path}.{key}: missing in candidate'
            diff = first_diff(a[key], b[key], f'{path}.{key}')
            if diff:
                return diff
        return ''
    if isinstance(a, list):
        if len(a) != len(b):
            return f'{path}: len {len(a)} != {len(b)}'
        for idx, (left, right) in enumerate(zip(a, b)):
            diff = first_diff(left, right, f'{path}[{idx}]')
            if diff:
                return diff
        return ''
    if a != b:
        return f'{path}: {a!r} != {b!r}'
    return ''


def run_one(args: argparse.Namespace, mode: str, output_dir: Path) -> Dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile: Dict[str, float] = {}
    start = time.perf_counter()
    _, result = run_process_form_workflow(
        image_path=args.input,
        config_path=args.config,
        output_dir=output_dir,
        lexicon_path=args.lexicon,
        lang=args.lang,
        debug_output=False,
        save_cells=False,
        profile=profile,
        ocr_candidate_mode=mode,
    )
    elapsed = time.perf_counter() - start
    return {
        'mode': mode,
        'output_dir': str(output_dir),
        'elapsed': elapsed,
        'profile': profile,
        'result': result,
        'semantic': semantic_projection(result),
        'xml_visible': xml_visible_projection(output_dir / 'report.xml'),
    }


def _profile_value(item: Dict[str, Any], key: str) -> float:
    return float((item.get('profile') or {}).get(key, 0.0))


def _format_seconds(value: float) -> str:
    return f'{value:.3f}s'


def main() -> None:
    parser = argparse.ArgumentParser(description='Run conservative OCR candidate ablations.')
    parser.add_argument('--input', required=True, help='Input image.')
    parser.add_argument('--config', required=True, help='Template config JSON.')
    parser.add_argument('--output-root', default='/tmp/table_ocr_ablation', help='Ablation output root.')
    parser.add_argument('--lexicon', default=None, help='Optional domain lexicon JSON.')
    parser.add_argument('--lang', default='ch', help='PaddleOCR language, default: ch')
    parser.add_argument(
        '--modes',
        nargs='*',
        default=[mode for mode in available_ocr_candidate_modes() if mode != STRICT_OCR_CANDIDATE_MODE],
        choices=available_ocr_candidate_modes(),
        help='Candidate modes to compare against strict baseline.',
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    modes = [STRICT_OCR_CANDIDATE_MODE] + [mode for mode in args.modes if mode != STRICT_OCR_CANDIDATE_MODE]
    results: List[Dict[str, Any]] = []
    for mode in modes:
        print(f'Running mode: {mode}', flush=True)
        results.append(run_one(args, mode, output_root / mode))

    baseline = results[0]
    summary: List[Dict[str, Any]] = []
    for item in results:
        semantic_diff = first_diff(baseline['semantic'], item['semantic'])
        xml_diff = first_diff(baseline['xml_visible'], item['xml_visible'])
        full_json_diff = first_diff(baseline['result'], item['result'])
        accepted = not semantic_diff and not xml_diff and not full_json_diff
        row = {
            'mode': item['mode'],
            'accepted': accepted,
            'elapsed': item['elapsed'],
            'ocr_title': _profile_value(item, 'ocr_title'),
            'ocr_remark': _profile_value(item, 'ocr_remark'),
            'ocr_bottom': _profile_value(item, 'ocr_bottom'),
            'ocr_main_table': _profile_value(item, 'ocr_main_table'),
            'ocr_calls_title': int(_profile_value(item, 'ocr_calls_title')),
            'ocr_calls_remark': int(_profile_value(item, 'ocr_calls_remark')),
            'ocr_calls_bottom': int(_profile_value(item, 'ocr_calls_bottom')),
            'ocr_calls_main_table': int(_profile_value(item, 'ocr_calls_main_table')),
            'semantic_diff': semantic_diff,
            'xml_diff': xml_diff,
            'full_json_diff': full_json_diff,
            'semantic_xml_same': not semantic_diff and not xml_diff,
            'full_json_same': not full_json_diff,
            'output_dir': item['output_dir'],
        }
        summary.append(row)

    summary_path = output_root / 'ablation_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print('')
    print('Ablation summary:')
    print('mode\taccepted\tsemantic/xml\tfull_json\telapsed\ttitle\tremark\tbottom\tmain\tcalls(title/remark/bottom/main)\tfirst_diff')
    for row in summary:
        first_problem = row['semantic_diff'] or row['xml_diff'] or row['full_json_diff'] or ''
        print(
            f"{row['mode']}\t"
            f"{'yes' if row['accepted'] else 'no'}\t"
            f"{'same' if row['semantic_xml_same'] else 'diff'}\t"
            f"{'same' if row['full_json_same'] else 'diff'}\t"
            f"{_format_seconds(row['elapsed'])}\t"
            f"{_format_seconds(row['ocr_title'])}\t"
            f"{_format_seconds(row['ocr_remark'])}\t"
            f"{_format_seconds(row['ocr_bottom'])}\t"
            f"{_format_seconds(row['ocr_main_table'])}\t"
            f"{row['ocr_calls_title']}/{row['ocr_calls_remark']}/{row['ocr_calls_bottom']}/{row['ocr_calls_main_table']}\t"
            f"{first_problem}"
        )
    print(f'Summary JSON: {summary_path}')


if __name__ == '__main__':
    main()
