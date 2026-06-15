# Table OCR Project

固定模板计划板 OCR 项目。流程会把输入图片配准到模板，裁切标题区、主表、备注区和底部署名区，识别表格内容，并输出结构化 JSON 与可用 WPS/Excel 打开的 `report.xml`。

当前推荐入口是 `src/process_form.py`。默认模式关闭硬编码报告校正规则，但启用词典先验，用于查看真实 OCR 与自动抽取结果；需要测试完全无词典先验的基线时，可以显式关闭词典先验；需要和人工答案对齐做对照时，可以显式开启硬编码规则。

## 目录结构

```text
table_ocr_project/
├── config/
│   ├── domain_lexicon_demo.json
│   └── template_config.json
├── data/
│   ├── input/
│   │   └── sample.jpeg
│   └── output/
├── src/
│   ├── process_form.py
│   ├── eval.sh
│   ├── build_template_config.py
│   └── table_ocr_project/
│       ├── pipeline.py
│       ├── semantic_extractors.py
│       ├── structured_main_table.py
│       ├── structured_process.py
│       └── structured_report.py
└── README.md
```

## 环境

服务器上当前使用的 conda 环境：

```bash
source /share/zhangyudong6-nfs/miniconda3/etc/profile.d/conda.sh
conda activate tableocr
```

如果在新环境部署，先安装依赖：

```bash
pip install -r requirements.txt
```

## 快速运行

先进入项目目录。项目路径使用绝对路径，输入图片等参数继续使用相对路径：

```bash
cd /share/zhangyudong6-nfs/AAAZLYH/code/HEU_task/table_ocr_project
```

无词典先验基线：关闭外部词表和内置代字代号候选，只保留图像结构、坐标、OCR 原文和通用正则规则：

```bash
python src/process_form.py \
  --input data/input/sample.jpeg \
  --config config/template_config.json \
  --output-dir data/output/process_no_priors \
  --profile \
  --disable-lexicon-priors
```

默认自动抽取：启用词典先验，关闭硬编码报告校正规则：

```bash
python src/process_form.py \
  --input data/input/sample.jpeg \
  --config config/template_config.json \
  --output-dir data/output/process_no_hardcoded \
  --lexicon config/domain_lexicon_demo.json \
  --profile
```

开启硬编码报告校正规则，用于和答案对齐做对照：

```bash
python src/process_form.py \
  --input data/input/sample.jpeg \
  --config config/template_config.json \
  --output-dir data/output/process_hardcoded \
  --lexicon config/domain_lexicon_demo.json \
  --profile \
  --enable-hardcoded-rules
```

输出中会打印当前状态：

```text
Hardcoded rules: disabled
Lexicon priors: enabled
```

或：

```text
Hardcoded rules: enabled
Lexicon priors: enabled
```

无词典先验时会显示：

```text
Hardcoded rules: disabled
Lexicon priors: disabled
```

也可以用 `src/run.sh` 运行默认流程。脚本会先切到绝对项目目录，第一个参数是图片相对路径；不传时默认使用 `data/input/sample.jpeg`：

```bash
bash src/run.sh
bash src/run.sh data/input/sample.jpeg
```

## 验证

默认验证 `data/output/process_structured/report.xml`：

```bash
bash src/eval.sh
```

验证指定输出：

```bash
bash src/eval.sh \
  data/output/process_no_priors/report.xml \
  data/input/report_gt.xml \
  data/output/eval_no_priors
```

```bash
bash src/eval.sh \
  data/output/process_no_hardcoded/report.xml \
  data/input/report_gt.xml \
  data/output/eval_no_hardcoded
```

硬编码规则对照验证：

```bash
bash src/eval.sh \
  data/output/process_hardcoded/report.xml \
  data/input/report_gt.xml \
  data/output/eval_hardcoded
```

评估脚本会输出整体单元格准确率、报告视图准确率、结构化数据准确率、CER、错误样例，并保存：

- `eval_result.json`
- `eval_report.txt`
- `eval_errors.csv`

当前 `data/input/sample.jpeg` 的三组对照结果：

| 模式 | 总体准确率 | 报告视图 | 结构化数据 | workflow_total | ocr_calls_total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无词典先验 `process_no_priors` | 44.99% | 46.35% | 42.95% | 17.261s | 12 |
| 自动抽取 + 词典先验 `process_no_hardcoded` | 92.29% | 87.55% | 99.36% | 14.108s | 12 |
| 硬编码报告校正 `process_hardcoded` | 99.74% | 100.00% | 99.36% | 14.054s | 12 |

## 输出文件

每次运行的输出目录包含：

- `metadata.json`：图像配准、区域框、网格、单元格索引。
- `ocr_result.json`：标题区、备注区、底部署名区、主表结构化 OCR 结果。
- `report.xml`：Spreadsheet XML，可用 WPS/Excel 打开。

注意：`report.xml` 是 XML Spreadsheet 格式，不是 `.xlsx`。如果 WPS/Excel 显示异常，优先检查是否存在合并单元格覆盖、列宽过小或空白单元格覆盖文本。

## 调试参数

保存配准图、区域裁剪图和主表网格调试图：

```bash
python src/process_form.py \
  --input data/input/sample.jpeg \
  --config config/template_config.json \
  --output-dir data/output/process_debug \
  --lexicon config/domain_lexicon_demo.json \
  --debug-output
```

保存全部主表单元格图像：

```bash
python src/process_form.py \
  --input data/input/sample.jpeg \
  --config config/template_config.json \
  --output-dir data/output/process_debug_cells \
  --lexicon config/domain_lexicon_demo.json \
  --debug-output \
  --save-cells
```

恢复旧版多候选 OCR 路径进行复核：

```bash
python src/process_form.py \
  --input data/input/sample.jpeg \
  --config config/template_config.json \
  --output-dir data/output/process_strict \
  --lexicon config/domain_lexicon_demo.json \
  --strict-ocr-candidates \
  --profile
```

## 主要模块

- `pipeline.py`：模板配准、区域裁切、网格 metadata 生成。
- `semantic_extractors.py`：标题区、备注区、底部署名区 OCR。
- `structured_main_table.py`：主表 OCR 行列归属、机型/机号/二次代码、代字代号和事件抽取。
- `structured_process.py`：串联完整结构化 OCR 流程。
- `structured_report.py`：把结构化结果写成 Spreadsheet XML 报告。

## 模板配置

重新生成模板配置：

```bash
python src/build_template_config.py \
  --template data/input/sample.jpeg \
  --output-config config/template_config.json \
  --output-debug-dir data/output/template_debug
```

会生成：

- `config/template_config.json`
- `data/output/template_debug/template_layout_debug.png`
- `data/output/template_debug/template_main_table_grid_debug.png`

## 词表

默认词表：

```text
config/domain_lexicon_demo.json
```

词表用于 OCR 后处理和近似匹配，建议维护：

- 姓名
- 代字代号
- 飞行代码
- 机型/机号/二次代码
- 常见备注缩写
