from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from table_ocr_project.structured_process import run_process_form_workflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Align image, crop fixed regions, and export a structured main-table report.'
    )
    parser.add_argument('--input', required=True, help='Input image.')
    parser.add_argument('--config', required=True, help='Template config JSON.')
    parser.add_argument('--output-dir', required=True, help='Output directory.')
    parser.add_argument('--lexicon', default=None, help='Optional domain lexicon JSON.')
    parser.add_argument('--lang', default='ch', help='PaddleOCR language, default: ch')
    args = parser.parse_args()

    meta, result = run_process_form_workflow(
        image_path=args.input,
        config_path=args.config,
        output_dir=args.output_dir,
        lexicon_path=args.lexicon,
        lang=args.lang,
    )

    print('Process finished.')
    print(f"Rows: {meta['grid']['num_rows']}, Cols: {meta['grid']['num_cols']}")
    print(f"Cells: {len(meta['cells'])}")
    print(f"JSON: {Path(args.output_dir) / 'ocr_result.json'}")
    print(f"Text report: {Path(args.output_dir) / 'report.txt'}")
    print(f"Main records: {len(result.get('main_table', {}).get('structured_records', []))}")


if __name__ == '__main__':
    main()
