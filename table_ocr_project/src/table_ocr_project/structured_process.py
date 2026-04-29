from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config_utils import dump_json, load_json
from .ocr_engine import PaddleOCREngine
from .pipeline import process_image_with_fixed_template, read_image
from .semantic_extractors import (
    extract_bottom_fields,
    extract_remark_fields,
    extract_title_fields,
)
from .structured_main_table import extract_structured_main_table
from .structured_report import render_structured_report_xml
from .text_utils import normalize_text


def default_lexicon_path() -> Path:
    return Path(__file__).resolve().parents[2] / 'config' / 'domain_lexicon_demo.json'


def _resolve_lexicon_path(lexicon_path: str | Path | None) -> Path | None:
    candidate = Path(lexicon_path) if lexicon_path else default_lexicon_path()
    return candidate if candidate.exists() else None


def _load_lexicon(lexicon_path: str | Path | None) -> Dict[str, List[str]]:
    resolved = _resolve_lexicon_path(lexicon_path)
    if not resolved:
        return {}
    return load_json(resolved)


def _repair_title_times(title: Dict[str, Any]) -> Dict[str, Any]:
    subregion_text = normalize_text(
        str(((title.get('subregion_texts') or {}).get('astronomical_times')) or '')
    )
    if not subregion_text:
        return title

    astro = dict(title.get('astronomical_times', {}) or {})
    alias_map = {
        '天亮时刻': ['天亮时刻'],
        '天黑时刻': ['天黑时刻', '天风时刻', '天墨时刻', '天嘿时刻'],
        '日出时刻': ['日出时刻'],
        '日没时刻': ['日没时刻'],
        '月出时刻': ['月出时刻'],
        '月没时刻': ['月没时刻'],
    }
    for key, aliases in alias_map.items():
        for alias in aliases:
            match = re.search(re.escape(alias) + r'\s*[:：]?\s*(\d{1,2}[:：]?\d{2})', subregion_text)
            if match:
                raw = match.group(1).replace('：', ':')
                digits = ''.join(ch for ch in raw if ch.isdigit())
                if len(digits) == 4:
                    astro[key] = f'{digits[:2]}:{digits[2:]}'
                    break

    title['astronomical_times'] = astro
    return title


def run_structured_ocr(
    input_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    lexicon_path: str | Path | None = None,
    lang: str = 'ch',
) -> Dict[str, Any]:
    config = load_json(config_path)
    output_dir = Path(output_dir)
    aligned = read_image(output_dir / 'aligned.png')
    lexicon = _load_lexicon(lexicon_path)

    engine = PaddleOCREngine(lang=lang)
    title = _repair_title_times(extract_title_fields(aligned, config, engine, lexicon))
    remark = extract_remark_fields(aligned, config, engine, lexicon)
    bottom = extract_bottom_fields(aligned, config, engine, lexicon)
    main_table = extract_structured_main_table(output_dir / 'main_table.png', config, engine, lexicon)

    result = {
        'input_image': str(Path(input_path).resolve()),
        'title': title,
        'remark': remark,
        'main_table': main_table,
        'bottom': bottom,
    }

    dump_json(result, output_dir / 'ocr_result.json')
    (output_dir / 'report.xml').write_text(
        render_structured_report_xml(result),
        encoding='utf-8'
    )
    return result


def run_process_form_workflow(
    image_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    lexicon_path: str | Path | None = None,
    lang: str = 'ch',
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta = process_image_with_fixed_template(
        image_path=image_path,
        config_path=config_path,
        output_dir=output_dir,
    )
    result = run_structured_ocr(
        input_path=image_path,
        config_path=config_path,
        output_dir=output_dir,
        lexicon_path=lexicon_path,
        lang=lang,
    )
    return meta, result


__all__ = [
    'default_lexicon_path',
    'run_process_form_workflow',
    'run_structured_ocr',
]