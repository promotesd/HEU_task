from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .text_utils import normalize_text


def _show(value: Any, default: str = '未识别') -> str:
    if value is None:
        return default
    text = normalize_text(str(value))
    return text if text else default


def _join_nonempty(parts: Sequence[str], sep: str = '，') -> str:
    return sep.join([p for p in parts if normalize_text(p)])


def render_structured_report(result: Dict[str, Any]) -> str:
    lines: List[str] = []

    title = result.get('title', {}) or {}
    if title:
        lines.append('【标题信息】')
        title_parts = []
        if normalize_text(str(title.get('confidentiality', '') or '')):
            title_parts.append(f"密级/标识：{normalize_text(str(title.get('confidentiality', '') or ''))}")
        if normalize_text(str(title.get('date', '') or '')):
            title_parts.append(f"日期：{normalize_text(str(title.get('date', '') or ''))}")
        if normalize_text(str(title.get('title', '') or '')):
            title_parts.append(f"标题：{normalize_text(str(title.get('title', '') or ''))}")
        title_parts.append(f"批准人：{_show(title.get('approved_name'))}")
        lines.append(_join_nonempty(title_parts))

        astro = title.get('astronomical_times', {}) or {}
        ordered_keys = ['天亮时刻', '天黑时刻', '日出时刻', '日没时刻', '月出时刻', '月没时刻']
        lines.append('时刻信息：' + '，'.join([f"{key}{_show(astro.get(key))}" for key in ordered_keys]))
        lines.append('')

    remark = result.get('remark', {}) or {}
    if remark:
        lines.append('【备注区】')
        lines.append(f"占场时间：{_show(remark.get('occupancy_time'))}")
        headers = remark.get('training_headers', []) or ['序号', '参训机型', '数量', '架次', '核算架次', '时间']
        lines.append('参训表表头：' + '，'.join(headers))
        entries = remark.get('training_entries', []) or []
        if entries:
            lines.append('参训表内容：')
            for idx, entry in enumerate(entries, start=1):
                row_text = _join_nonempty([f"{key}:{value}" for key, value in entry.items() if normalize_text(str(value))])
                if row_text:
                    lines.append(f"  {idx}. {row_text}")
        else:
            lines.append('参训表内容：当前为空')
        lines.append('')

    main = result.get('main_table', {}) or {}
    if main:
        lines.append('【主表】')
        top_section = main.get('top_section', []) or []
        if top_section:
            lines.append('顶部编组信息：')
            for entry in top_section:
                label = normalize_text(str(entry.get('label', '') or ''))
                line_texts = [normalize_text(str(item)) for item in entry.get('lines', []) if normalize_text(str(item))]
                if not line_texts:
                    lines.append(f"  {label}：未清晰识别")
                    continue
                lines.append(f"  {label}：")
                for idx, text in enumerate(line_texts, start=1):
                    lines.append(f"    第{idx}行：{text}")

        records = main.get('structured_records', []) or main.get('body_rows', []) or []
        if records:
            lines.append('飞行记录：')
            visible_index = 0
            for record in records:
                if not isinstance(record, dict):
                    continue
                base_parts = [
                    f"机型：{_show(record.get('aircraft_type'))}",
                    f"机号：{_show(record.get('aircraft_no'))}",
                    f"二次代码：{_show(record.get('secondary_code'))}",
                ]
                crew_codes = [normalize_text(str(code)) for code in record.get('crew_codes', []) if normalize_text(str(code))]
                if crew_codes:
                    base_parts.append('关联代字代号：' + '、'.join(crew_codes))
                visible_index += 1
                lines.append(f"  {visible_index}. " + '，'.join(base_parts))

                events = record.get('events', []) or []
                if events:
                    lines.append('     飞行片段：')
                    event_index = 0
                    for event in events:
                        event_desc: List[str] = []
                        time_text = normalize_text(str(event.get('display_time', '') or ''))
                        if time_text:
                            event_desc.append(f"时间:{time_text}")
                        pilot_code = normalize_text(str(event.get('pilot_code', '') or ''))
                        if pilot_code:
                            event_desc.append(f"驾驶代字代号:{pilot_code}")
                        flight_code = normalize_text(str(event.get('flight_code', '') or ''))
                        if flight_code:
                            event_desc.append(f"标记:{flight_code}")
                        text = normalize_text(str(event.get('text', '') or ''))
                        if text:
                            event_desc.append(f"备注:{text}")
                        if event_desc:
                            event_index += 1
                            lines.append(f"       {event_index}. " + '，'.join(event_desc))
                    if event_index == 0:
                        lines.append('       未清晰识别')
                else:
                    lines.append('     飞行片段：未清晰识别')
        else:
            lines.append('飞行记录：未识别到有效内容')

        lines.append('主表说明：主表中部小字较密，以上为按固定模板整理后的结构化结果，个别时间和代字代号建议人工复核。')
        lines.append('')

    bottom = result.get('bottom', {}) or {}
    if bottom:
        lines.append('【底部签名】')
        line1 = bottom.get('line1', {}) or {}
        line2 = bottom.get('line2', {}) or {}
        lines.append(_join_nonempty([
            f"第一行队长：{_show(line1.get('队长'))}",
            f"第一行政治委员：{_show(line1.get('政治委员'))}",
        ]))
        lines.append(_join_nonempty([
            f"第二行队长：{_show(line2.get('队长'))}",
            f"第二行政治委员：{_show(line2.get('政治委员'))}",
        ]))
        lines.append('')

    return '\n'.join(lines)


__all__ = ['render_structured_report']
