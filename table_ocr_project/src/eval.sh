#!/usr/bin/env bash
set -e

# 用法：
#   bash src/eval.sh
# 或：
#   bash src/eval.sh /path/to/report.xml /path/to/report_gt.xml /path/to/output_eval_dir

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SRC_DIR}/.." && pwd)"

PRED_XML="${1:-${PROJECT_ROOT}/data/output/process_structured/report.xml}"
GT_XML="${2:-${PROJECT_ROOT}/data/input/report_gt.xml}"
OUT_DIR="${3:-${PROJECT_ROOT}/data/output/eval_test}"

echo "Project root: ${PROJECT_ROOT}"
echo "Pred XML    : ${PRED_XML}"
echo "GT XML      : ${GT_XML}"
echo "Out dir     : ${OUT_DIR}"

if [ ! -f "${PRED_XML}" ]; then
  echo "[ERROR] Pred XML not found: ${PRED_XML}"
  exit 1
fi

if [ ! -f "${GT_XML}" ]; then
  echo "[ERROR] GT XML not found: ${GT_XML}"
  echo "请先复制一份预测 XML 到 data/input/report_gt.xml，然后人工校正："
  echo "cp \"${PRED_XML}\" \"${GT_XML}\""
  exit 1
fi

PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}" \
python "${SRC_DIR}/eval_test/evaluate_report_xml.py" \
  --pred "${PRED_XML}" \
  --gt "${GT_XML}" \
  --out-dir "${OUT_DIR}"
