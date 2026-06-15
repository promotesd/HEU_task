from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_START = time.perf_counter()

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from table_ocr_project.structured_process import run_process_form_workflow
from table_ocr_project.semantic_extractors import (
    DEFAULT_OCR_CANDIDATE_MODE,
    STRICT_OCR_CANDIDATE_MODE,
    available_ocr_candidate_modes,
)

IMPORTS_DONE = time.perf_counter()


def main() -> None:
    main_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description='Align image, crop fixed regions, and export a structured main-table XML report.'
    )
    parser.add_argument('--input', required=True, help='Input image.')
    parser.add_argument('--config', required=True, help='Template config JSON.')
    parser.add_argument('--output-dir', required=True, help='Output directory.')
    parser.add_argument('--lexicon', default=None, help='Optional domain lexicon JSON.')
    parser.add_argument('--lang', default='ch', help='PaddleOCR language, default: ch')
    parser.add_argument(
        '--debug-output',
        action='store_true',
        help='Save aligned/cropped/debug images. Disabled by default for speed.',
    )
    parser.add_argument(
        '--save-cells',
        action='store_true',
        help='Save every main-table cell image. Requires --debug-output.',
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Print coarse timing information for the structured workflow.',
    )
    parser.add_argument(
        '--strict-ocr-candidates',
        action='store_true',
        help='Use the legacy multi-candidate OCR strategy for title/remark/bottom.',
    )
    parser.add_argument(
        '--ocr-candidate-mode',
        choices=available_ocr_candidate_modes(),
        default=None,
        help='Internal OCR candidate strategy override. Defaults to the validated fast mode.',
    )
    parser.add_argument(
        '--enable-hardcoded-rules',
        action='store_true',
        help='Enable hardcoded/prior report normalization rules for comparison. Disabled by default for baseline testing.',
    )
    args = parser.parse_args()
    args_parsed = time.perf_counter()
    if args.save_cells and not args.debug_output:
        parser.error('--save-cells requires --debug-output')
    if args.strict_ocr_candidates and args.ocr_candidate_mode:
        parser.error('--strict-ocr-candidates cannot be used with --ocr-candidate-mode')

    profile = {} if args.profile else None
    ocr_candidate_mode = (
        STRICT_OCR_CANDIDATE_MODE
        if args.strict_ocr_candidates
        else (args.ocr_candidate_mode or DEFAULT_OCR_CANDIDATE_MODE)
    )

    workflow_start = time.perf_counter()
    meta, result = run_process_form_workflow(
        image_path=args.input,
        config_path=args.config,
        output_dir=args.output_dir,
        lexicon_path=args.lexicon,
        lang=args.lang,
        debug_output=args.debug_output,
        save_cells=args.save_cells,
        profile=profile,
        ocr_candidate_mode=ocr_candidate_mode,
        enable_hardcoded_rules=args.enable_hardcoded_rules,
    )
    workflow_done = time.perf_counter()

    print('Process finished.')
    print(f"Rows: {meta['grid']['num_rows']}, Cols: {meta['grid']['num_cols']}")
    print(f"Cells: {len(meta['cells'])}")
    print(f"Debug images: {'enabled' if args.debug_output else 'disabled'}")
    print(f"Cell images: {'enabled' if args.save_cells else 'disabled'}")
    print(f"OCR candidate mode: {ocr_candidate_mode}")
    print(f"Hardcoded rules: {'enabled' if args.enable_hardcoded_rules else 'disabled'}")
    print(f"JSON: {Path(args.output_dir) / 'ocr_result.json'}")
    print(f"XML report: {Path(args.output_dir) / 'report.xml'}")
    print(f"Main records: {len(result.get('main_table', {}).get('structured_records', []))}")
    if profile is not None:
        stage_total = sum(value for key, value in profile.items() if not key.startswith('ocr_calls_'))
        workflow_total = workflow_done - workflow_start
        script_total_before_profile = time.perf_counter() - SCRIPT_START
        print('Profile:')
        print(f"  startup_imports: {IMPORTS_DONE - SCRIPT_START:.3f}s")
        print(f"  parse_args: {args_parsed - main_start:.3f}s")
        print(f"  workflow_total: {workflow_total:.3f}s")
        print(f"  workflow_profiled_stages_total: {stage_total:.3f}s")
        print(f"  workflow_unprofiled: {max(0.0, workflow_total - stage_total):.3f}s")
        for key, value in profile.items():
            if key.startswith('ocr_calls_'):
                print(f"  {key}: {int(value)}")
            else:
                print(f"  {key}: {value:.3f}s")
        print(f"  script_total_before_profile_print: {script_total_before_profile:.3f}s")


if __name__ == '__main__':
    main()
