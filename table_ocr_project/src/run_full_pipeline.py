from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from table_ocr_project.pipeline import run_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run the full template-based OCR pipeline and export JSON + text report.'
    )
    parser.add_argument('--input', required=True, help='Input image path.')
    parser.add_argument('--config', required=True, help='Template config JSON.')
    parser.add_argument('--output-dir', required=True, help='Output directory.')
    parser.add_argument('--lexicon', default=None, help='Optional domain lexicon JSON.')
    parser.add_argument('--lang', default='ch', help='PaddleOCR language, default: ch')
    args = parser.parse_args()

    result = run_full_pipeline(
        image_path=args.input,
        config_path=args.config,
        output_dir=args.output_dir,
        lexicon_path=args.lexicon,
        lang=args.lang,
    )
    print('Full OCR pipeline finished.')
    print(f"JSON: {Path(args.output_dir) / 'ocr_result.json'}")
    print(f"Text report: {Path(args.output_dir) / 'report.txt'}")
    print(f"Title: {result.get('title', {}).get('title', '')}")


if __name__ == '__main__':
    main()


