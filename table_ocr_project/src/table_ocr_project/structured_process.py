from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Tuple

import numpy as np

from .config_utils import dump_json, load_json
from .ocr_engine import PaddleOCREngine
from .pipeline import process_image_with_fixed_template_in_memory, read_image
from .semantic_extractors import (
    DEFAULT_OCR_CANDIDATE_MODE,
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


def _add_profile_time(profile: MutableMapping[str, float] | None, key: str, start: float) -> None:
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + (time.perf_counter() - start)


def _add_profile_count(profile: MutableMapping[str, float] | None, key: str, value: int) -> None:
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + float(value)


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
    ordered_keys = ['天亮时刻', '天黑时刻', '日出时刻', '日没时刻', '月出时刻', '月没时刻']
    canonical_astro_text = ' '.join(
        f'{key}:{astro.get(key)}'
        for key in ordered_keys
        if normalize_text(str(astro.get(key, '') or ''))
    )
    if canonical_astro_text:
        subregion_texts = dict(title.get('subregion_texts', {}) or {})
        subregion_texts['astronomical_times'] = canonical_astro_text
        title['subregion_texts'] = subregion_texts
    return title


def run_structured_ocr(
    input_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    lexicon_path: str | Path | None = None,
    lang: str = 'ch',
    aligned: np.ndarray | None = None,
    main_table_img: np.ndarray | None = None,
    config: Dict[str, Any] | None = None,
    profile: MutableMapping[str, float] | None = None,
    ocr_candidate_mode: str = DEFAULT_OCR_CANDIDATE_MODE,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if config is None:
        config = load_json(config_path)
    output_dir = Path(output_dir)
    if aligned is None:
        aligned = read_image(output_dir / 'aligned.png')
    lexicon = _load_lexicon(lexicon_path)
    _add_profile_time(profile, 'prepare_ocr_inputs', t0)

    t0 = time.perf_counter()
    engine = PaddleOCREngine(lang=lang)
    _add_profile_time(profile, 'init_ocr_engine', t0)

    t0 = time.perf_counter()
    calls_before = int(getattr(engine, 'ocr_call_count', 0))
    title = _repair_title_times(extract_title_fields(aligned, config, engine, lexicon, candidate_mode=ocr_candidate_mode))
    _add_profile_time(profile, 'ocr_title', t0)
    _add_profile_count(profile, 'ocr_calls_title', int(getattr(engine, 'ocr_call_count', 0)) - calls_before)

    t0 = time.perf_counter()
    calls_before = int(getattr(engine, 'ocr_call_count', 0))
    remark = extract_remark_fields(aligned, config, engine, lexicon, candidate_mode=ocr_candidate_mode)
    _add_profile_time(profile, 'ocr_remark', t0)
    _add_profile_count(profile, 'ocr_calls_remark', int(getattr(engine, 'ocr_call_count', 0)) - calls_before)

    t0 = time.perf_counter()
    calls_before = int(getattr(engine, 'ocr_call_count', 0))
    bottom = extract_bottom_fields(aligned, config, engine, lexicon, candidate_mode=ocr_candidate_mode)
    _add_profile_time(profile, 'ocr_bottom', t0)
    _add_profile_count(profile, 'ocr_calls_bottom', int(getattr(engine, 'ocr_call_count', 0)) - calls_before)

    t0 = time.perf_counter()
    calls_before = int(getattr(engine, 'ocr_call_count', 0))
    main_table_source = main_table_img if main_table_img is not None else output_dir / 'main_table.png'
    main_table = extract_structured_main_table(main_table_source, config, engine, lexicon)
    _add_profile_time(profile, 'ocr_main_table', t0)
    _add_profile_count(profile, 'ocr_calls_main_table', int(getattr(engine, 'ocr_call_count', 0)) - calls_before)
    _add_profile_count(profile, 'ocr_calls_total', int(getattr(engine, 'ocr_call_count', 0)))

    result = {
        'input_image': str(Path(input_path).resolve()),
        'title': title,
        'remark': remark,
        'main_table': main_table,
        'bottom': bottom,
    }

    t0 = time.perf_counter()
    dump_json(result, output_dir / 'ocr_result.json')
    _add_profile_time(profile, 'write_json', t0)

    t0 = time.perf_counter()
    (output_dir / 'report.xml').write_text(
        render_structured_report_xml(result),
        encoding='utf-8'
    )
    _add_profile_time(profile, 'write_report', t0)
    return result


def run_process_form_workflow(
    image_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    lexicon_path: str | Path | None = None,
    lang: str = 'ch',
    debug_output: bool = False,
    save_cells: bool = False,
    profile: MutableMapping[str, float] | None = None,
    ocr_candidate_mode: str = DEFAULT_OCR_CANDIDATE_MODE,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    artifacts = process_image_with_fixed_template_in_memory(
        image_path=image_path,
        config_path=config_path,
        output_dir=output_dir,
        save_debug_outputs=debug_output,
        save_cells=save_cells,
        profile=profile,
    )
    meta = artifacts['metadata']
    result = run_structured_ocr(
        input_path=image_path,
        config_path=config_path,
        output_dir=output_dir,
        lexicon_path=lexicon_path,
        lang=lang,
        aligned=artifacts['aligned'],
        main_table_img=artifacts['crops']['main_table'],
        config=artifacts['config'],
        profile=profile,
        ocr_candidate_mode=ocr_candidate_mode,
    )
    return meta, result


__all__ = [
    'default_lexicon_path',
    'run_process_form_workflow',
    'run_structured_ocr',
]
