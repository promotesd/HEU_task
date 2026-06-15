#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def local_name(tag: str) -> str:
    return tag.split('}', 1)[1] if '}' in tag else tag


def norm_text(value: Any, keep_space: bool = False) -> str:
    if value is None:
        return ''
    text = unicodedata.normalize('NFKC', str(value)).replace('\u3000', ' ').replace('\xa0', ' ').strip()
    return re.sub(r'\s+', ' ', text) if keep_space else re.sub(r'\s+', '', text)


def levenshtein(a: Any, b: Any) -> int:
    a, b = norm_text(a), norm_text(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: Any, gt: Any) -> float:
    p, g = norm_text(pred), norm_text(gt)
    if not g:
        return 0.0 if not p else 1.0
    return levenshtein(p, g) / max(1, len(g))


def edit_similarity(pred: Any, gt: Any) -> float:
    p, g = norm_text(pred), norm_text(gt)
    return 1.0 - levenshtein(p, g) / max(len(p), len(g), 1)


def time_to_minutes(t: str) -> Optional[int]:
    m = re.match(r'^(\d{1,2})[:：](\d{2})$', norm_text(t))
    if not m:
        return None
    h, minute = int(m.group(1)), int(m.group(2))
    if 0 <= h <= 23 and 0 <= minute <= 59:
        return h * 60 + minute
    return None


def parse_time_interval(text: Any) -> Optional[Tuple[int, int]]:
    t = norm_text(text).replace('～', '~').replace('—', '-').replace('–', '-').replace('至', '~')
    if not t:
        return None
    if '~' in t:
        a, b = t.split('~', 1)
    elif '-' in t:
        a, b = t.split('-', 1)
    else:
        a = b = t
    s, e = time_to_minutes(a), time_to_minutes(b)
    if s is None or e is None or e < s:
        return None
    return (s, e + 10) if e == s else (s, e)


def time_iou(pred: Any, gt: Any) -> float:
    p, g = parse_time_interval(pred), parse_time_interval(gt)
    if p is None or g is None:
        return 1.0 if norm_text(pred) == norm_text(gt) else 0.0
    inter = max(0, min(p[1], g[1]) - max(p[0], g[0]))
    union = max(p[1], g[1]) - min(p[0], g[0])
    return inter / union if union > 0 else 0.0


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {'precision': p, 'recall': r, 'f1': f1}


def child_text(elem: ET.Element, tag: str, default: str = '') -> str:
    for child in list(elem):
        if local_name(child.tag) == tag:
            return norm_text(child.text, keep_space=True) if child.text else default
    return default


def find_child(elem: ET.Element, tag: str) -> Optional[ET.Element]:
    for child in list(elem):
        if local_name(child.tag) == tag:
            return child
    return None


def iter_children(elem: ET.Element, tag: str) -> Iterable[ET.Element]:
    for child in list(elem):
        if local_name(child.tag) == tag:
            yield child


@dataclass
class XMLPayload:
    kind: str
    structured: Optional[Dict[str, Any]]
    sheets: Dict[str, List[List[str]]]


def parse_xml(path: Path) -> XMLPayload:
    tree = ET.parse(path)
    root = tree.getroot()
    name = local_name(root.tag)
    if name == 'structured_report':
        return XMLPayload('structured_report', parse_structured_report(root), {})
    sheets = parse_spreadsheet_xml(root)
    if sheets:
        return XMLPayload('spreadsheet', None, sheets)
    return XMLPayload('generic_xml', {'flat_fields': flatten_any_xml(root)}, {})


def parse_structured_report(root: ET.Element) -> Dict[str, Any]:
    data: Dict[str, Any] = {'fields': {}, 'top_section': [], 'records': []}
    title = find_child(root, 'title_info')
    if title is not None:
        for key in ['confidentiality', 'date', 'title', 'approved_name']:
            data['fields'][f'title.{key}'] = child_text(title, key)
        astro = find_child(title, 'astronomical_times')
        if astro is not None:
            for t in iter_children(astro, 'time'):
                n = t.attrib.get('name', '')
                if n:
                    data['fields'][f'title.astronomical_times.{n}'] = norm_text(t.text, keep_space=True)
    remark = find_child(root, 'remark_section')
    if remark is not None:
        data['fields']['remark.occupancy_time'] = child_text(remark, 'occupancy_time')
        headers_elem = find_child(remark, 'training_headers')
        headers = [] if headers_elem is None else [norm_text(h.text, keep_space=True) for h in iter_children(headers_elem, 'header')]
        data['fields']['remark.training_headers'] = '、'.join([h for h in headers if h])
    main = find_child(root, 'main_table')
    if main is not None:
        top = find_child(main, 'top_section')
        if top is not None:
            for g in iter_children(top, 'group'):
                data['top_section'].append({
                    'label': norm_text(g.attrib.get('label', '')),
                    'lines': [norm_text(x.text, keep_space=True) for x in iter_children(g, 'line')],
                })
        recs = find_child(main, 'flight_records')
        if recs is not None:
            for r in iter_children(recs, 'record'):
                rec: Dict[str, Any] = {
                    'index': int(r.attrib.get('index', len(data['records']) + 1)),
                    'aircraft_type': child_text(r, 'aircraft_type'),
                    'aircraft_no': child_text(r, 'aircraft_no'),
                    'secondary_code': child_text(r, 'secondary_code'),
                    'crew_codes': [],
                    'events': [],
                }
                crew = find_child(r, 'crew_codes')
                if crew is not None:
                    rec['crew_codes'] = [norm_text(c.text, keep_space=True) for c in iter_children(crew, 'code') if norm_text(c.text)]
                events = find_child(r, 'events')
                if events is not None:
                    for e in iter_children(events, 'event'):
                        rec['events'].append({
                            'index': int(e.attrib.get('index', len(rec['events']) + 1)),
                            'display_time': child_text(e, 'display_time'),
                            'pilot_code': child_text(e, 'pilot_code'),
                            'flight_code': child_text(e, 'flight_code'),
                            'remark': child_text(e, 'remark'),
                        })
                data['records'].append(rec)
        desc = find_child(main, 'description')
        if desc is not None:
            data['fields']['main_table.description'] = norm_text(desc.text, keep_space=True)
    bottom = find_child(root, 'bottom_signatures')
    if bottom is not None:
        for line_tag in ['line1', 'line2']:
            line = find_child(bottom, line_tag)
            if line is not None:
                data['fields'][f'bottom.{line_tag}.captain'] = child_text(line, 'captain')
                data['fields'][f'bottom.{line_tag}.political_commissar'] = child_text(line, 'political_commissar')
    return data


def flatten_any_xml(root: ET.Element) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    def walk(e: ET.Element, prefix: str) -> None:
        cur = f'{prefix}/{local_name(e.tag)}' if prefix else local_name(e.tag)
        text = norm_text(e.text, keep_space=True)
        if text:
            fields[cur] = text
        for k, v in e.attrib.items():
            fields[f'{cur}@{local_name(k)}'] = norm_text(v, keep_space=True)
        for child in list(e):
            walk(child, cur)
    walk(root, '')
    return fields


def get_index_attr(elem: ET.Element) -> Optional[int]:
    for k, v in elem.attrib.items():
        if local_name(k) == 'Index':
            try:
                return int(v) - 1
            except Exception:
                return None
    return None


def parse_spreadsheet_xml(root: ET.Element) -> Dict[str, List[List[str]]]:
    sheets: Dict[str, List[List[str]]] = {}
    worksheets = [e for e in root.iter() if local_name(e.tag) == 'Worksheet']
    if not worksheets:
        tables = [e for e in root.iter() if local_name(e.tag) == 'Table']
        if tables:
            sheets['Sheet1'] = parse_table_rows(tables[0])
        return sheets
    for idx, ws in enumerate(worksheets, 1):
        sheet_name = ws.attrib.get('Name') or ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') or f'Sheet{idx}'
        table = next((e for e in list(ws) if local_name(e.tag) == 'Table'), None)
        if table is not None:
            sheets[sheet_name] = parse_table_rows(table)
    return sheets


def parse_table_rows(table: ET.Element) -> List[List[str]]:
    rows: List[List[str]] = []
    for row_elem in [e for e in list(table) if local_name(e.tag) == 'Row']:
        row_idx = get_index_attr(row_elem)
        if row_idx is not None:
            while len(rows) < row_idx:
                rows.append([])
        row: List[str] = []
        for cell in [c for c in list(row_elem) if local_name(c.tag) == 'Cell']:
            col_idx = get_index_attr(cell)
            if col_idx is not None:
                while len(row) < col_idx:
                    row.append('')
            text = ''
            for child in list(cell):
                if local_name(child.tag) == 'Data':
                    text = ''.join(child.itertext())
                    break
            if not text:
                text = ''.join(cell.itertext())
            row.append(norm_text(text, keep_space=True))
        rows.append(row)
    return rows


def flatten_structured(data: Dict[str, Any]) -> Dict[str, str]:
    flat: Dict[str, str] = dict(data.get('fields', {}) or {})
    for i, g in enumerate(data.get('top_section', []) or [], 1):
        flat[f'top_section.{i}.label'] = g.get('label', '')
        for j, line in enumerate(g.get('lines', []) or [], 1):
            flat[f'top_section.{i}.line.{j}'] = line
    for r in data.get('records', []) or []:
        rid = int(r.get('index', 0))
        base = f'record.{rid}'
        for key in ['aircraft_type', 'aircraft_no', 'secondary_code']:
            flat[f'{base}.{key}'] = r.get(key, '')
        flat[f'{base}.crew_codes'] = '、'.join(r.get('crew_codes', []) or [])
        for e in r.get('events', []) or []:
            eid = int(e.get('index', 0))
            for key in ['display_time', 'pilot_code', 'flight_code', 'remark']:
                flat[f'{base}.event.{eid}.{key}'] = e.get(key, '')
    return flat


def eval_flat_fields(pred: Dict[str, str], gt: Dict[str, str], max_errors: int = 100) -> Dict[str, Any]:
    keys = sorted(set(pred) | set(gt))
    gt_keys = sorted(gt)
    exact_all = exact_gt = dist_sum = gt_chars = 0
    sim_sum = 0.0
    errors = []
    for key in keys:
        p, g = norm_text(pred.get(key, '')), norm_text(gt.get(key, ''))
        if p == g:
            exact_all += 1
        if key in gt:
            if p == g:
                exact_gt += 1
            dist_sum += levenshtein(p, g)
            gt_chars += max(1, len(g))
            sim_sum += edit_similarity(p, g)
            if p != g and len(errors) < max_errors:
                errors.append({'key': key, 'gt': gt.get(key, ''), 'pred': pred.get(key, ''), 'cer': round(cer(p, g), 4)})
    pred_nonempty = {k for k, v in pred.items() if norm_text(v)}
    gt_nonempty = {k for k, v in gt.items() if norm_text(v)}
    tp = sum(1 for k in pred_nonempty & gt_nonempty if norm_text(pred.get(k)) == norm_text(gt.get(k)))
    fp = len(pred_nonempty) - tp
    fn = len(gt_nonempty) - tp
    return {
        'field_count_pred': len(pred),
        'field_count_gt': len(gt),
        'field_key_union_count': len(keys),
        'field_exact_acc_on_gt_keys': exact_gt / len(gt_keys) if gt_keys else 1.0,
        'field_exact_acc_on_key_union': exact_all / len(keys) if keys else 1.0,
        'field_cer_on_gt_keys': dist_sum / gt_chars if gt_chars else 0.0,
        'field_edit_similarity_avg': sim_sum / len(gt_keys) if gt_keys else 1.0,
        'nonempty_field_prf': prf(tp, fp, fn),
        'errors': errors,
    }


def eval_time_fields(pred_flat: Dict[str, str], gt_flat: Dict[str, str]) -> Dict[str, Any]:
    time_keys = [k for k in gt_flat if k.endswith('.display_time') or 'time' in k.lower() or '时刻' in k]
    vals, exact = [], 0
    for k in time_keys:
        p, g = pred_flat.get(k, ''), gt_flat.get(k, '')
        exact += int(norm_text(p) == norm_text(g))
        vals.append(time_iou(p, g))
    return {'time_field_count': len(time_keys), 'time_exact_acc': exact / len(time_keys) if time_keys else 1.0, 'time_interval_iou_avg': sum(vals) / len(vals) if vals else 1.0}


def eval_crew_codes(pred_data: Dict[str, Any], gt_data: Dict[str, Any]) -> Dict[str, Any]:
    pred_records = {int(r.get('index', i + 1)): r for i, r in enumerate(pred_data.get('records', []) or [])}
    gt_records = {int(r.get('index', i + 1)): r for i, r in enumerate(gt_data.get('records', []) or [])}
    tp = fp = fn = 0
    per_record = []
    for k in sorted(set(pred_records) | set(gt_records)):
        ps = set(norm_text(x) for x in pred_records.get(k, {}).get('crew_codes', []) if norm_text(x))
        gs = set(norm_text(x) for x in gt_records.get(k, {}).get('crew_codes', []) if norm_text(x))
        ctp, cfp, cfn = len(ps & gs), len(ps - gs), len(gs - ps)
        tp += ctp; fp += cfp; fn += cfn
        per_record.append({'record_index': k, **prf(ctp, cfp, cfn), 'gt': sorted(gs), 'pred': sorted(ps)})
    return {'crew_code_micro_prf': prf(tp, fp, fn), 'per_record': per_record}


def eval_event_counts(pred_data: Dict[str, Any], gt_data: Dict[str, Any]) -> Dict[str, Any]:
    pred_records = {int(r.get('index', i + 1)): r for i, r in enumerate(pred_data.get('records', []) or [])}
    gt_records = {int(r.get('index', i + 1)): r for i, r in enumerate(gt_data.get('records', []) or [])}
    details, exact, abs_errors = [], 0, []
    for k in sorted(set(pred_records) | set(gt_records)):
        pc = len(pred_records.get(k, {}).get('events', []) or [])
        gc = len(gt_records.get(k, {}).get('events', []) or [])
        exact += int(pc == gc)
        abs_errors.append(abs(pc - gc))
        details.append({'record_index': k, 'pred_event_count': pc, 'gt_event_count': gc, 'abs_error': abs(pc - gc)})
    return {'record_count_pred': len(pred_records), 'record_count_gt': len(gt_records), 'event_count_exact_acc_by_record': exact / len(details) if details else 1.0, 'event_count_mae_by_record': sum(abs_errors) / len(abs_errors) if abs_errors else 0.0, 'details': details}


def eval_structured_payload(pred: Dict[str, Any], gt: Dict[str, Any], max_errors: int = 100) -> Dict[str, Any]:
    pred_flat, gt_flat = flatten_structured(pred), flatten_structured(gt)
    return {'mode': 'structured_report', 'field_metrics': eval_flat_fields(pred_flat, gt_flat, max_errors), 'time_metrics': eval_time_fields(pred_flat, gt_flat), 'crew_code_metrics': eval_crew_codes(pred, gt), 'event_count_metrics': eval_event_counts(pred, gt)}


def cell_matrix_to_dict(rows: List[List[str]]) -> Dict[Tuple[int, int], str]:
    out: Dict[Tuple[int, int], str] = {}
    for r, row in enumerate(rows, 1):
        for c, v in enumerate(row, 1):
            if norm_text(v):
                out[(r, c)] = norm_text(v, keep_space=True)
    return out


def eval_sheet(pred_rows: List[List[str]], gt_rows: List[List[str]], max_errors: int = 100) -> Dict[str, Any]:
    pred_cells, gt_cells = cell_matrix_to_dict(pred_rows), cell_matrix_to_dict(gt_rows)
    union_pos, gt_pos = sorted(set(pred_cells) | set(gt_cells)), sorted(gt_cells)
    exact_union = exact_gt = dist_sum = gt_chars = 0
    sim_sum = 0.0
    errors = []
    for pos in union_pos:
        p, g = pred_cells.get(pos, ''), gt_cells.get(pos, '')
        if norm_text(p) == norm_text(g):
            exact_union += 1
        if pos in gt_cells:
            if norm_text(p) == norm_text(g):
                exact_gt += 1
            dist_sum += levenshtein(p, g)
            gt_chars += max(1, len(norm_text(g)))
            sim_sum += edit_similarity(p, g)
            if norm_text(p) != norm_text(g) and len(errors) < max_errors:
                errors.append({'row': pos[0], 'col': pos[1], 'gt': g, 'pred': p, 'cer': round(cer(p, g), 4)})
    tp = exact_gt
    fp = len(pred_cells) - tp
    fn = len(gt_cells) - tp
    return {'pred_nonempty_cells': len(pred_cells), 'gt_nonempty_cells': len(gt_cells), 'cell_position_union_count': len(union_pos), 'cell_exact_acc_on_gt_nonempty': exact_gt / len(gt_pos) if gt_pos else 1.0, 'cell_exact_acc_on_nonempty_union': exact_union / len(union_pos) if union_pos else 1.0, 'cell_cer_on_gt_nonempty': dist_sum / gt_chars if gt_chars else 0.0, 'cell_edit_similarity_avg': sim_sum / len(gt_pos) if gt_pos else 1.0, 'nonempty_cell_prf': prf(tp, fp, fn), 'errors': errors}


def eval_spreadsheet_payload(pred_sheets: Dict[str, List[List[str]]], gt_sheets: Dict[str, List[List[str]]], max_errors: int = 100) -> Dict[str, Any]:
    per_sheet = {name: eval_sheet(pred_sheets.get(name, []), gt_sheets.get(name, []), max_errors) for name in sorted(set(pred_sheets) | set(gt_sheets))}
    pred_cells, gt_cells = {}, {}
    for s, rows in pred_sheets.items():
        for (r, c), v in cell_matrix_to_dict(rows).items():
            pred_cells[f'{s}!R{r}C{c}'] = v
    for s, rows in gt_sheets.items():
        for (r, c), v in cell_matrix_to_dict(rows).items():
            gt_cells[f'{s}!R{r}C{c}'] = v
    return {'mode': 'spreadsheet', 'sheet_count_pred': len(pred_sheets), 'sheet_count_gt': len(gt_sheets), 'overall_cell_metrics': eval_flat_fields(pred_cells, gt_cells, max_errors), 'per_sheet': per_sheet}


def evaluate(pred_path: Path, gt_path: Path, max_errors: int = 100) -> Dict[str, Any]:
    pred, gt = parse_xml(pred_path), parse_xml(gt_path)
    result: Dict[str, Any] = {'pred_path': str(pred_path), 'gt_path': str(gt_path), 'pred_kind': pred.kind, 'gt_kind': gt.kind}
    if pred.structured is not None and gt.structured is not None:
        if 'flat_fields' in pred.structured or 'flat_fields' in gt.structured:
            result['metrics'] = {'mode': 'generic_xml', 'field_metrics': eval_flat_fields(pred.structured.get('flat_fields', {}), gt.structured.get('flat_fields', {}), max_errors)}
        else:
            result['metrics'] = eval_structured_payload(pred.structured, gt.structured, max_errors)
    elif pred.sheets and gt.sheets:
        result['metrics'] = eval_spreadsheet_payload(pred.sheets, gt.sheets, max_errors)
    else:
        pred_flat = flatten_structured(pred.structured) if pred.structured else {f'{s}!R{r}C{c}': v for s, rows in pred.sheets.items() for (r, c), v in cell_matrix_to_dict(rows).items()}
        gt_flat = flatten_structured(gt.structured) if gt.structured else {f'{s}!R{r}C{c}': v for s, rows in gt.sheets.items() for (r, c), v in cell_matrix_to_dict(rows).items()}
        result['metrics'] = {'mode': 'mixed_fallback_flatten', 'field_metrics': eval_flat_fields(pred_flat, gt_flat, max_errors)}
    return result


def pct(x: float) -> str:
    return f'{x * 100:.2f}%'


def render_text_report(result: Dict[str, Any]) -> str:
    lines = ['XML 结构化表格识别评估报告', '=' * 42, f"Pred: {result.get('pred_path')}", f"GT  : {result.get('gt_path')}", f"Pred XML kind: {result.get('pred_kind')}", f"GT XML kind  : {result.get('gt_kind')}", '']
    m = result.get('metrics', {}) or {}
    mode = m.get('mode')
    lines += [f'评估模式: {mode}', '']
    if mode == 'structured_report':
        fm = m['field_metrics']; pr = fm['nonempty_field_prf']
        lines += ['[字段级指标]', f"字段准确率(gt keys): {pct(fm['field_exact_acc_on_gt_keys'])}", f"字段CER(gt keys): {fm['field_cer_on_gt_keys']:.4f}", f"字段编辑相似度均值: {fm['field_edit_similarity_avg']:.4f}", f"非空字段 P/R/F1: {pct(pr['precision'])} / {pct(pr['recall'])} / {pct(pr['f1'])}", '']
        tm = m['time_metrics']
        lines += ['[时间字段指标]', f"时间字段数量: {tm['time_field_count']}", f"时间完全匹配率: {pct(tm['time_exact_acc'])}", f"时间区间IoU均值: {tm['time_interval_iou_avg']:.4f}", '']
        cm = m['crew_code_metrics']['crew_code_micro_prf']
        lines += ['[代字代号集合指标]', f"代字代号 P/R/F1: {pct(cm['precision'])} / {pct(cm['recall'])} / {pct(cm['f1'])}", '']
        em = m['event_count_metrics']
        lines += ['[事件数量指标]', f"记录数量 pred/gt: {em['record_count_pred']} / {em['record_count_gt']}", f"每条记录事件数完全匹配率: {pct(em['event_count_exact_acc_by_record'])}", f"每条记录事件数MAE: {em['event_count_mae_by_record']:.4f}", '']
        errors = fm.get('errors', [])
        if errors:
            lines.append('[字段错误样例]')
            for e in errors[:30]:
                lines.append(f"- {e['key']}: GT='{e['gt']}' | Pred='{e['pred']}' | CER={e['cer']}")
            lines.append('')
    elif mode == 'spreadsheet':
        overall = m['overall_cell_metrics']; pr = overall['nonempty_field_prf']
        lines += ['[单元格级总体指标]', f"非空单元格准确率(gt cells): {pct(overall['field_exact_acc_on_gt_keys'])}", f"非空单元格CER(gt cells): {overall['field_cer_on_gt_keys']:.4f}", f"非空单元格编辑相似度均值: {overall['field_edit_similarity_avg']:.4f}", f"非空单元格 P/R/F1: {pct(pr['precision'])} / {pct(pr['recall'])} / {pct(pr['f1'])}", '', '[各工作表指标]']
        for s, sm in m.get('per_sheet', {}).items():
            pr2 = sm['nonempty_cell_prf']
            lines.append(f"- {s}: acc={pct(sm['cell_exact_acc_on_gt_nonempty'])}, CER={sm['cell_cer_on_gt_nonempty']:.4f}, F1={pct(pr2['f1'])}, pred/gt cells={sm['pred_nonempty_cells']}/{sm['gt_nonempty_cells']}")
        lines += ['', '[单元格错误样例]']
        count = 0
        for s, sm in m.get('per_sheet', {}).items():
            for e in sm.get('errors', []):
                lines.append(f"- {s}!R{e['row']}C{e['col']}: GT='{e['gt']}' | Pred='{e['pred']}' | CER={e['cer']}")
                count += 1
                if count >= 30:
                    break
            if count >= 30:
                break
        if count == 0:
            lines.append('无')
    else:
        fm = m.get('field_metrics', {}); pr = fm.get('nonempty_field_prf', {'precision':0,'recall':0,'f1':0})
        lines += ['[通用字段指标]', f"准确率(gt keys): {pct(fm.get('field_exact_acc_on_gt_keys', 0))}", f"CER(gt keys): {fm.get('field_cer_on_gt_keys', 0):.4f}", f"P/R/F1: {pct(pr['precision'])} / {pct(pr['recall'])} / {pct(pr['f1'])}"]
    return '\n'.join(lines)


def write_csv_errors(result: Dict[str, Any], path: Path) -> None:
    rows = []
    m = result.get('metrics', {}) or {}
    if m.get('mode') == 'spreadsheet':
        for sheet, sm in m.get('per_sheet', {}).items():
            for e in sm.get('errors', []):
                rows.append({'type': 'cell', 'location': f'{sheet}!R{e["row"]}C{e["col"]}', 'gt': e['gt'], 'pred': e['pred'], 'cer': e['cer']})
    else:
        fm = m.get('field_metrics', {})
        if m.get('mode') == 'structured_report':
            fm = m.get('field_metrics', {})
        for e in fm.get('errors', []):
            rows.append({'type': 'field', 'location': e.get('key', ''), 'gt': e.get('gt', ''), 'pred': e.get('pred', ''), 'cer': e.get('cer', '')})
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['type', 'location', 'gt', 'pred', 'cer'])
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate report.xml against report_gt.xml.')
    parser.add_argument('--pred', required=True, help='Predicted report.xml')
    parser.add_argument('--gt', required=True, help='Ground-truth report_gt.xml')
    parser.add_argument('--out-dir', default=None, help='Output directory')
    parser.add_argument('--max-errors', type=int, default=200)
    args = parser.parse_args()
    pred_path, gt_path = Path(args.pred), Path(args.gt)
    out_dir = Path(args.out_dir) if args.out_dir else pred_path.parent / 'eval_test'
    out_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(pred_path, gt_path, args.max_errors)
    (out_dir / 'eval_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'eval_report.txt').write_text(render_text_report(result), encoding='utf-8')
    write_csv_errors(result, out_dir / 'eval_errors.csv')
    print(render_text_report(result))
    print('\n[Saved]', out_dir / 'eval_result.json')
    print('[Saved]', out_dir / 'eval_report.txt')
    print('[Saved]', out_dir / 'eval_errors.csv')


if __name__ == '__main__':
    main()
