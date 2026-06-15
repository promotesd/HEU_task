#!/usr/bin/env bash
set -euo pipefail

cd /share/zhangyudong6-nfs/AAAZLYH/code/HEU_task/table_ocr_project

input_image="${1:-data/input/sample.jpeg}"

python src/process_form.py \
  --input "$input_image" \
  --config config/template_config.json \
  --output-dir data/output/process_structured \
  --lexicon config/domain_lexicon_demo.json \
  --debug-output
