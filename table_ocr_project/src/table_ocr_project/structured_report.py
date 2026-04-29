from __future__ import annotations

from typing import Any, Dict, List, Sequence
import xml.etree.ElementTree as ET
from xml.dom import minidom

from .text_utils import normalize_text


XMLNS = 'urn:schemas-microsoft-com:office:spreadsheet'
XMLNS_O = 'urn:schemas-microsoft-com:office:office'
XMLNS_X = 'urn:schemas-microsoft-com:office:excel'
XMLNS_HTML = 'http://www.w3.org/TR/REC-html40'


def _show(value: Any, default: str = '未识别') -> str:
    if value is None:
        return default
    text = normalize_text(str(value))
    return text if text else default


def _clean_text(value: Any, default: str = '') -> str:
    if value is None:
        return default
    text = normalize_text(str(value))
    return text if text else default


def _safe_join(values: Sequence[str], sep: str = '、') -> str:
    cleaned = [normalize_text(str(v)) for v in values if normalize_text(str(v))]
    return sep.join(cleaned)


def _build_root() -> ET.Element:
    return ET.Element(
        'Workbook',
        {
            'xmlns': XMLNS,
            'xmlns:o': XMLNS_O,
            'xmlns:x': XMLNS_X,
            'xmlns:ss': XMLNS,
            'xmlns:html': XMLNS_HTML,
        },
    )


def _pretty_xml(root: ET.Element) -> str:
    rough = ET.tostring(root, encoding='utf-8')
    parsed = minidom.parseString(rough)
    pretty = parsed.toprettyxml(indent='  ', encoding='utf-8').decode('utf-8')
    marker = '<?xml version="1.0" encoding="utf-8"?>'
    if pretty.startswith(marker):
        pretty = pretty.replace(
            marker,
            marker + '\n<?mso-application progid="Excel.Sheet"?>',
            1,
        )
    return pretty


def _add_styles(root: ET.Element) -> None:
    styles = ET.SubElement(root, 'Styles')

    def add_style(
        style_id: str,
        *,
        font_name: str = '宋体',
        font_size: str = '10',
        bold: bool = False,
        horizontal: str = 'Left',
        vertical: str = 'Center',
        wrap: bool = False,
        border: bool = True,
    ) -> None:
        style = ET.SubElement(styles, 'Style', {'ss:ID': style_id})

        alignment_attrs = {'ss:Vertical': vertical}
        if horizontal:
            alignment_attrs['ss:Horizontal'] = horizontal
        if wrap:
            alignment_attrs['ss:WrapText'] = '1'
        ET.SubElement(style, 'Alignment', alignment_attrs)

        ET.SubElement(
            style,
            'Font',
            {
                'ss:FontName': font_name,
                'ss:Size': font_size,
                'ss:Bold': '1' if bold else '0',
            },
        )

        if border:
            borders = ET.SubElement(style, 'Borders')
            for pos in ['Left', 'Right', 'Top', 'Bottom']:
                ET.SubElement(
                    borders,
                    'Border',
                    {
                        'ss:Position': pos,
                        'ss:LineStyle': 'Continuous',
                        'ss:Weight': '1',
                    },
                )

    add_style('Default', horizontal='Left', vertical='Center', wrap=False, border=False)
    add_style('s_title', font_size='16', bold=True, horizontal='Center', border=False)
    add_style('s_section', font_size='11', bold=True, horizontal='Left', border=True)
    add_style('s_label', font_size='10', bold=True, horizontal='Center', border=True)
    add_style('s_value', font_size='10', bold=False, horizontal='Left', wrap=True, border=True)
    add_style('s_center', font_size='10', bold=False, horizontal='Center', wrap=True, border=True)
    add_style('s_table_header', font_size='10', bold=True, horizontal='Center', wrap=True, border=True)
    add_style('s_cell', font_size='10', bold=False, horizontal='Left', wrap=True, border=True)
    add_style('s_cell_center', font_size='10', bold=False, horizontal='Center', wrap=True, border=True)
    add_style('s_small', font_size='9', bold=False, horizontal='Left', wrap=True, border=True)


def _add_columns(table: ET.Element, widths: Sequence[int]) -> None:
    for width in widths:
        ET.SubElement(
            table,
            'Column',
            {
                'ss:AutoFitWidth': '0',
                'ss:Width': str(width),
            },
        )


def _add_row(table: ET.Element, *, height: int | None = None) -> ET.Element:
    attrs: Dict[str, str] = {}
    if height is not None:
        attrs['ss:AutoFitHeight'] = '0'
        attrs['ss:Height'] = str(height)
    return ET.SubElement(table, 'Row', attrs)


def _add_cell(
    row: ET.Element,
    value: Any = '',
    *,
    col: int | None = None,
    style_id: str = 's_cell',
    merge_across: int | None = None,
    merge_down: int | None = None,
    data_type: str = 'String',
) -> None:
    attrs: Dict[str, str] = {'ss:StyleID': style_id}
    if col is not None:
        attrs['ss:Index'] = str(col)
    if merge_across is not None and merge_across > 0:
        attrs['ss:MergeAcross'] = str(merge_across)
    if merge_down is not None and merge_down > 0:
        attrs['ss:MergeDown'] = str(merge_down)

    cell = ET.SubElement(row, 'Cell', attrs)
    text = _clean_text(value, '')
    if text != '':
        data = ET.SubElement(cell, 'Data', {'ss:Type': data_type})
        data.text = text


def _add_worksheet_options(ws: ET.Element) -> None:
    options = ET.SubElement(ws, 'WorksheetOptions', {'xmlns': XMLNS_X})
    ET.SubElement(options, 'ProtectObjects').text = 'False'
    ET.SubElement(options, 'ProtectScenarios').text = 'False'


def _flatten_events(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    main = result.get('main_table', {}) or {}
    records = main.get('structured_records', []) or []
    flat_rows: List[Dict[str, Any]] = []

    for record_index, record in enumerate(records, start=1):
        events = record.get('events', []) or []
        if not events:
            flat_rows.append({
                'record_index': str(record_index),
                'aircraft_type': _show(record.get('aircraft_type')),
                'aircraft_no': _show(record.get('aircraft_no')),
                'secondary_code': _show(record.get('secondary_code')),
                'event_index': '',
                'display_time': '',
                'pilot_code': '',
                'course_note': '',
                'extra_note': '',
            })
            continue

        for event_index, event in enumerate(events, start=1):
            flat_rows.append({
                'record_index': str(record_index),
                'aircraft_type': _show(record.get('aircraft_type')),
                'aircraft_no': _show(record.get('aircraft_no')),
                'secondary_code': _show(record.get('secondary_code')),
                'event_index': str(event_index),
                'display_time': _show(event.get('display_time')),
                'pilot_code': _show(event.get('pilot_code')),
                'course_note': _clean_text(event.get('flight_code')),
                'extra_note': _clean_text(event.get('text')),
            })

    return flat_rows


def _collect_code_column_values(result: Dict[str, Any]) -> List[str]:
    main = result.get('main_table', {}) or {}
    records = main.get('structured_records', []) or []
    values: List[str] = []
    for record in records:
        for crew_row in record.get('crew_rows', []) or []:
            code_name = _clean_text(crew_row.get('code_name'))
            if code_name:
                values.append(code_name)
    return values


def _remark_headers(remark: Dict[str, Any]) -> List[str]:
    headers = remark.get('training_headers', []) or []
    default_headers = ['序号', '参训机型', '数量', '架次', '核算架次', '时间']
    cleaned = [_clean_text(h) for h in headers if _clean_text(h)]
    return cleaned if cleaned else default_headers


def _remark_entries(remark: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = remark.get('training_entries', []) or []
    return [entry for entry in entries if isinstance(entry, dict)]


def _entry_value(entry: Dict[str, Any], keys: Sequence[str], default: str = '') -> str:
    for key in keys:
        if key in entry and _clean_text(entry.get(key)):
            return _clean_text(entry.get(key))
    return default


def _build_top_section_rows(table: ET.Element, main: Dict[str, Any]) -> None:
    row = _add_row(table, height=22)
    _add_cell(row, '顶部编组信息', col=1, style_id='s_section', merge_across=11)

    top_section = main.get('top_section', []) or []
    top_rows = max(len(top_section), 1)
    for i in range(top_rows):
        row = _add_row(table, height=20)
        if i < len(top_section):
            entry = top_section[i]
            label = _clean_text(entry.get('label'), '未识别')
            lines = entry.get('lines', []) or []
            line_text = ' | '.join([_clean_text(x) for x in lines if _clean_text(x)])
            _add_cell(row, label, col=1, style_id='s_label', merge_across=1)
            _add_cell(row, line_text or '未清晰识别', col=3, style_id='s_value', merge_across=9)
        else:
            _add_cell(row, '', col=1, style_id='s_cell', merge_across=11)


def _build_report_view_sheet(root: ET.Element, result: Dict[str, Any]) -> None:
    ws = ET.SubElement(root, 'Worksheet', {'ss:Name': '报告视图'})
    table = ET.SubElement(ws, 'Table')

    # 1-9: 飞行记录主表
    # 10: 空列
    # 11-12: 代字代号列
    # 13: 空列
    # 14-22: 备注区
    _add_columns(table, [
        55, 65, 60, 70, 60, 85, 115, 85, 190,
        20,
        80, 80,
        20,
        80, 80, 80, 55, 95, 55, 55, 85, 95
    ])

    title = result.get('title', {}) or {}
    astro = title.get('astronomical_times', {}) or {}
    remark = result.get('remark', {}) or {}
    main = result.get('main_table', {}) or {}
    bottom = result.get('bottom', {}) or {}

    # 顶部：左侧标题信息 + 中间标题 + 右侧天文时刻
    row = _add_row(table, height=24)
    _add_cell(row, '密级/标识', col=1, style_id='s_label')
    _add_cell(row, title.get('confidentiality'), col=2, style_id='s_value')
    _add_cell(row, title.get('title'), col=5, style_id='s_title', merge_across=7)
    _add_cell(row, '天亮时刻', col=15, style_id='s_label')
    _add_cell(row, astro.get('天亮时刻'), col=16, style_id='s_center')
    _add_cell(row, '天黑时刻', col=17, style_id='s_label')
    _add_cell(row, astro.get('天黑时刻'), col=18, style_id='s_center')

    row = _add_row(table, height=22)
    _add_cell(row, '批准人', col=1, style_id='s_label')
    _add_cell(row, title.get('approved_name'), col=2, style_id='s_value')
    _add_cell(row, '日出时刻', col=15, style_id='s_label')
    _add_cell(row, astro.get('日出时刻'), col=16, style_id='s_center')
    _add_cell(row, '日没时刻', col=17, style_id='s_label')
    _add_cell(row, astro.get('日没时刻'), col=18, style_id='s_center')

    row = _add_row(table, height=22)
    _add_cell(row, '日期', col=1, style_id='s_label')
    _add_cell(row, title.get('date'), col=2, style_id='s_value')
    _add_cell(row, '月出时刻', col=15, style_id='s_label')
    _add_cell(row, astro.get('月出时刻'), col=16, style_id='s_center')
    _add_cell(row, '月没时刻', col=17, style_id='s_label')
    _add_cell(row, astro.get('月没时刻'), col=18, style_id='s_center')

    _add_row(table, height=12)

    # 顶部编组信息，删除原来右上那块备注
    _build_top_section_rows(table, main)

    _add_row(table, height=12)

    # 主表区标题
    row = _add_row(table, height=22)
    _add_cell(row, '飞行记录', col=1, style_id='s_section', merge_across=8)
    _add_cell(row, '代字代号', col=11, style_id='s_section', merge_across=1)
    _add_cell(row, '备注区', col=14, style_id='s_section', merge_across=8)

    # 主表表头 + 右侧备注区的“一、参训”
    row = _add_row(table, height=24)
    headers = [
        ('序号', 1),
        ('机型', 2),
        ('机号', 3),
        ('二次代码', 4),
        ('事件序号', 5),
        ('时间', 6),
        ('驾驶代字代号', 7),
        ('课目注释', 8),
        ('附加说明', 9),
    ]
    for text, col in headers:
        _add_cell(row, text, col=col, style_id='s_table_header')

    _add_cell(row, '代字代号', col=11, style_id='s_table_header', merge_across=1)

    # 备注区第一行：一、参训
    _add_cell(row, '一、参训', col=14, style_id='s_label', merge_across=8)

    flat_rows = _flatten_events(result)
    code_column_values = _collect_code_column_values(result)
    remark_entries = _remark_entries(remark)

    training_count = max(len(remark_entries), 6)   # 参训至少保留6行
    blank_note_rows = 12                           # 占场时间下面的大块留白
    remark_rows_needed = 1 + training_count + 1 + 1 + 1 + blank_note_rows
    # 分别对应：
    # 1行：参训表头
    # training_count行：参训数据
    # 1行：合计
    # 1行：二、占场时间：
    # 1行：08:40
    # blank_note_rows行：大块空白区域

    data_row_count = max(len(flat_rows), len(code_column_values), remark_rows_needed)

    for i in range(data_row_count):
        row_height = 22
        if i >= training_count + 4:
            row_height = 28   # 下方空白区高一点，更像原图
        row = _add_row(table, height=row_height)

        # =========================
        # 左侧飞行记录
        # =========================
        if i < len(flat_rows):
            item = flat_rows[i]
            _add_cell(row, item['record_index'], col=1, style_id='s_cell_center')
            _add_cell(row, item['aircraft_type'], col=2, style_id='s_cell_center')
            _add_cell(row, item['aircraft_no'], col=3, style_id='s_cell_center')
            _add_cell(row, item['secondary_code'], col=4, style_id='s_cell_center')
            _add_cell(row, item['event_index'], col=5, style_id='s_cell_center')
            _add_cell(row, item['display_time'], col=6, style_id='s_cell_center')
            _add_cell(row, item['pilot_code'], col=7, style_id='s_cell_center')
            _add_cell(row, item['course_note'], col=8, style_id='s_cell_center')
            _add_cell(row, item['extra_note'], col=9, style_id='s_cell')
        else:
            for col in range(1, 10):
                _add_cell(row, '', col=col, style_id='s_cell')

        # =========================
        # 中间代字代号列
        # =========================
        if i < len(code_column_values):
            _add_cell(row, code_column_values[i], col=11, style_id='s_cell_center', merge_across=1)
        else:
            _add_cell(row, '', col=11, style_id='s_cell', merge_across=1)

        # =========================
        # 右侧备注区
        # 结构：
        # 0            -> 参训表头
        # 1..N         -> 参训数据
        # N+1          -> 合计
        # N+2          -> 二、占场时间：
        # N+3          -> 08:40
        # N+4 以后     -> 大块空白区
        # =========================

        # 参训表头
        if i == 0:
            _add_cell(row, '序号', col=14, style_id='s_table_header')
            _add_cell(row, '参训机型', col=15, style_id='s_table_header', merge_across=1)
            _add_cell(row, '数量', col=17, style_id='s_table_header')
            _add_cell(row, '架次', col=18, style_id='s_table_header')
            _add_cell(row, '核算架次', col=19, style_id='s_table_header', merge_across=1)
            _add_cell(row, '时间', col=21, style_id='s_table_header', merge_across=1)

        # 参训数据
        elif 1 <= i <= training_count:
            entry = remark_entries[i - 1] if (i - 1) < len(remark_entries) else {}

            seq_val = _entry_value(entry, ['序号'], str(i))
            type_val = _entry_value(entry, ['参训机型', '机型'])
            qty_val = _entry_value(entry, ['数量'])
            sorties_val = _entry_value(entry, ['架次'])
            calc_sorties_val = _entry_value(entry, ['核算架次'])
            time_val = _entry_value(entry, ['时间'])

            _add_cell(row, seq_val, col=14, style_id='s_cell_center')
            _add_cell(row, type_val, col=15, style_id='s_cell', merge_across=1)
            _add_cell(row, qty_val, col=17, style_id='s_cell_center')
            _add_cell(row, sorties_val, col=18, style_id='s_cell_center')
            _add_cell(row, calc_sorties_val, col=19, style_id='s_cell_center', merge_across=1)
            _add_cell(row, time_val, col=21, style_id='s_cell_center', merge_across=1)

        # 合计行
        elif i == training_count + 1:
            _add_cell(row, '', col=14, style_id='s_cell')
            _add_cell(row, '合计', col=15, style_id='s_cell_center', merge_across=1)
            _add_cell(row, '', col=17, style_id='s_cell')
            _add_cell(row, '', col=18, style_id='s_cell')
            _add_cell(row, '', col=19, style_id='s_cell', merge_across=1)
            _add_cell(row, '', col=21, style_id='s_cell', merge_across=1)

        # 二、占场时间：
        elif i == training_count + 2:
            _add_cell(row, '二、占场时间：', col=14, style_id='s_label', merge_across=8)

        # 占场时间值
        elif i == training_count + 3:
            _add_cell(row, _show(remark.get('occupancy_time')), col=14, style_id='s_value', merge_across=8)

        # 下方整块空白书写区
        else:
            _add_cell(row, '', col=14, style_id='s_cell', merge_across=8)

    row = _add_row(table, height=34)
    _add_cell(
        row,
        '说明：主表中部小字较密，以上为按固定模板整理后的结构化结果，个别时间和代字代号建议人工复核。',
        col=1,
        style_id='s_small',
        merge_across=21,
    )

    _add_row(table, height=12)

    line1 = bottom.get('line1', {}) or {}
    line2 = bottom.get('line2', {}) or {}

    row = _add_row(table, height=22)
    _add_cell(row, '第一行队长', col=1, style_id='s_label')
    _add_cell(row, line1.get('队长'), col=2, style_id='s_value', merge_across=3)
    _add_cell(row, '第一行政治委员', col=6, style_id='s_label')
    _add_cell(row, line1.get('政治委员'), col=7, style_id='s_value', merge_across=3)

    row = _add_row(table, height=22)
    _add_cell(row, '第二行队长', col=1, style_id='s_label')
    _add_cell(row, line2.get('队长'), col=2, style_id='s_value', merge_across=3)
    _add_cell(row, '第二行政治委员', col=6, style_id='s_label')
    _add_cell(row, line2.get('政治委员'), col=7, style_id='s_value', merge_across=3)

    _add_worksheet_options(ws)


def _build_data_sheet(root: ET.Element, result: Dict[str, Any]) -> None:
    ws = ET.SubElement(root, 'Worksheet', {'ss:Name': '结构化数据'})
    table = ET.SubElement(ws, 'Table')
    _add_columns(table, [60, 60, 60, 70, 70, 90, 110, 90, 180, 120])

    row = _add_row(table, height=24)
    headers = [
        '记录序号', '机型', '机号', '二次代码', '事件序号',
        '时间', '驾驶代字代号', '课目注释', '附加说明', '来源图片'
    ]
    for idx, h in enumerate(headers, start=1):
        _add_cell(row, h, col=idx, style_id='s_table_header')

    flat_rows = _flatten_events(result)
    input_image = _clean_text(result.get('input_image'))

    for item in flat_rows:
        row = _add_row(table, height=22)
        _add_cell(row, item['record_index'], col=1, style_id='s_cell_center')
        _add_cell(row, item['aircraft_type'], col=2, style_id='s_cell_center')
        _add_cell(row, item['aircraft_no'], col=3, style_id='s_cell_center')
        _add_cell(row, item['secondary_code'], col=4, style_id='s_cell_center')
        _add_cell(row, item['event_index'], col=5, style_id='s_cell_center')
        _add_cell(row, item['display_time'], col=6, style_id='s_cell_center')
        _add_cell(row, item['pilot_code'], col=7, style_id='s_cell_center')
        _add_cell(row, item['course_note'], col=8, style_id='s_cell_center')
        _add_cell(row, item['extra_note'], col=9, style_id='s_cell')
        _add_cell(row, input_image, col=10, style_id='s_small')

    _add_worksheet_options(ws)


def render_structured_report_xml(result: Dict[str, Any]) -> str:
    root = _build_root()
    _add_styles(root)
    _build_report_view_sheet(root, result)
    _build_data_sheet(root, result)
    return _pretty_xml(root)


__all__ = ['render_structured_report_xml']