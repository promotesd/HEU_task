from __future__ import annotations

from typing import Any, Dict, List

from .text_utils import normalize_text


def _join_nonempty(parts: List[str], sep: str = '，') -> str:
    return sep.join(p for p in parts if p)


def _show(value: Any, default: str = '未识别') -> str:
    if value is None:
        return default
    value = normalize_text(str(value))
    return value if value else default


def render_report(result: Dict[str, Any]) -> str:
    lines: List[str] = []

    title = result.get('title', {})
    if title:
        lines.append('【标题信息】')

        title_parts: List[str] = []
        confidentiality = normalize_text(str(title.get('confidentiality', '') or ''))
        date_value = normalize_text(str(title.get('date', '') or ''))
        title_text = normalize_text(str(title.get('title', '') or ''))
        approved_name = normalize_text(str(title.get('approved_name', '') or ''))

        if confidentiality:
            title_parts.append(f"密级/标识：{confidentiality}")

        # 这里故意不强制输出日期；如果为空就不打印
        if date_value:
            title_parts.append(f"日期：{date_value}")

        if title_text:
            title_parts.append(f"标题：{title_text}")

        title_parts.append(f"批准人：{approved_name if approved_name else '未识别'}")

        lines.append(_join_nonempty(title_parts))

        astro = title.get('astronomical_times', {}) or {}
        ordered_keys = ['天亮时刻', '天黑时刻', '日出时刻', '日没时刻', '月出时刻', '月没时刻']
        astro_text = '，'.join([f"{k}{_show(astro.get(k))}" for k in ordered_keys])
        lines.append(f"时刻信息：{astro_text}")
        lines.append('')

    remark = result.get('remark', {})
    if remark:
        lines.append('【备注区】')
        lines.append(f"占场时间：{_show(remark.get('occupancy_time'))}")

        headers = remark.get('training_headers', []) or ['序号', '参训机型', '数量', '架次', '核算架次', '时间']
        lines.append('参训表表头：' + '，'.join(headers))

        entries = remark.get('training_entries', []) or []
        if entries:
            lines.append('参训表内容：')
            idx = 0
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                row_text = _join_nonempty([f"{k}:{v}" for k, v in entry.items() if v], sep='，')
                if not row_text:
                    continue
                idx += 1
                lines.append(f"  {idx}. {row_text}")
            if idx == 0:
                lines.append('  当前为空')
        else:
            lines.append('参训表内容：当前为空')

        lines.append('')

    main = result.get('main_table', {})
    if main:
        lines.append('【主表】')
        body_rows = main.get('body_rows', []) or []
        if body_rows:
            lines.append('主体记录：')
            count = 0
            for row in body_rows:
                if not isinstance(row, dict):
                    continue
                prefix = _join_nonempty([
                    normalize_text(str(row.get('aircraft_type', '') or '')),
                    normalize_text(str(row.get('aircraft_no', '') or '')),
                    normalize_text(str(row.get('secondary_code', '') or '')),
                    normalize_text(str(row.get('name', '') or '')),
                    normalize_text(str(row.get('code_name', '') or '')),
                ])
                events = row.get('events', []) or []
                event_texts = []
                for ev in events:
                    if not isinstance(ev, dict):
                        continue
                    ev_text = normalize_text(str(ev.get('text', '') or ''))
                    if not ev_text:
                        continue
                    start_time = normalize_text(str(ev.get('start_time', '') or ''))
                    end_time = normalize_text(str(ev.get('end_time', '') or ''))
                    if start_time and end_time:
                        event_texts.append(f"{start_time}~{end_time}，{ev_text}")
                    elif start_time:
                        event_texts.append(f"{start_time}，{ev_text}")
                    else:
                        event_texts.append(ev_text)

                item = _join_nonempty([prefix, '；'.join(event_texts)])
                if item:
                    count += 1
                    lines.append(f"  {count}. {item}")

            if count == 0:
                lines.append('主体记录：未识别到有效内容')
        else:
            lines.append('主体记录：未识别到有效内容')

        lines.append('')

    bottom = result.get('bottom', {})
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

    return '\n'.join(line for line in lines if line is not None)