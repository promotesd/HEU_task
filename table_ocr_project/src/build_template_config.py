from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from table_ocr_project.pipeline import bootstrap_template_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build template config from one clear template image.'
    )
    parser.add_argument('--template', required=True, help='Path to template image.')
    parser.add_argument('--output-config', required=True, help='Path to output template JSON.')
    parser.add_argument('--output-debug-dir', required=True, help='Debug image directory.')
    args = parser.parse_args()

    config = bootstrap_template_config(
        template_path=args.template,
        output_config_path=args.output_config,
        output_debug_dir=args.output_debug_dir,
    )
    print('Template config generated.')
    print(f"Config: {args.output_config}")
    print(f"Image size: {config['template']['image_size']}")
    print(f"Grid cols: {len(config['grid']['x_lines']) - 1}, rows: {len(config['grid']['y_lines']) - 1}")


if __name__ == '__main__':
    main()
