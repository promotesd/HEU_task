# Table OCR Project

这个项目用于处理固定模板的军用计划板图像。当前代码已经拆分为“模板对齐与切分”和“结构化主表识别”两层，适合对同一类版式图片进行批量 OCR、表格结构化和中文报告输出。

## 项目目标

给定一张固定模板的表单图片，完成以下流程：

1. 以清晰样本图生成模板配置。
2. 将待识别图片与模板自动配准。
3. 按固定区域裁切 `title`、`main_table`、`remark`、`bottom`。
4. 对主表网格、标题区、备注区、底部签名区做 OCR。
5. 输出结构化 JSON 和可直接阅读的 `report.txt`。

当前特别针对主表做了额外结构化整理，重点抽取：

- 主表上半区的编组/人员信息
- 飞机起降时间
- 驾驶代字代号
- 机型、机号、二次代码
- 飞行片段及备注文本

## 目录结构

```text
table_ocr_project/
├── README.md
├── requirements.txt
├── config/
│   ├── domain_lexicon_demo.json
│   ├── example_paths.json
│   └── template_config.json
├── data/
│   ├── input/
│   └── output/
└── src/
    ├── build_template_config.py
    ├── debug_semantic_boxes.py
    ├── process_form.py
    ├── run_full_pipeline.py
    └── table_ocr_project/
        ├── __init__.py
        ├── alignment.py
        ├── config_utils.py
        ├── grid.py
        ├── layout.py
        ├── narrative.py
        ├── ocr_engine.py
        ├── pipeline.py
        ├── preprocess.py
        ├── semantic_extractors.py
        ├── structured_main_table.py
        ├── structured_process.py
        ├── structured_report.py
        └── text_utils.py
```

## 模块说明

- `src/build_template_config.py`
  从一张清晰模板图自动生成 `template_config.json`。
- `src/process_form.py`
  当前推荐入口。先做对齐/切分，再输出面向主表结构化的 `ocr_result.json` 和 `report.txt`。
- `src/run_full_pipeline.py`
  保留原始完整流程，输出传统语义抽取结果和文字报告。
- `src/debug_semantic_boxes.py`
  调试各语义区域 OCR 的辅助脚本。

包内核心模块说明：

- `pipeline.py`
  模板配置生成、图像对齐、区域裁切、单元格切分、完整流程调度。
- `structured_process.py`
  结构化流程总入口，串联主表结构化识别和结构化报告输出。
- `structured_main_table.py`
  主表上半区和飞行记录的结构化整理逻辑。
- `structured_report.py`
  将结构化结果写成易读的 `report.txt`。
- `semantic_extractors.py`
  标题区、备注区、底部签名区和旧版主表抽取逻辑。

## 环境准备

建议在项目根目录安装依赖：

```bash
pip install -r requirements.txt
```

如果 `PaddleOCR` 相关环境尚未准备好，请先按你的 Python/CUDA 环境安装对应版本依赖。

## 使用流程

### 1. 生成模板配置

选择一张最清晰、最标准的同版式图片作为模板：

```bash
python src/build_template_config.py \
  --template data/input/sample.jpeg \
  --output-config config/template_config.json \
  --output-debug-dir data/output/template_debug
```

输出内容：

- `config/template_config.json`
- `data/output/template_debug/template_layout_debug.png`
- `data/output/template_debug/template_main_table_grid_debug.png`

### 2. 总流程入口：结构化主表处理

这个入口最适合你当前的主表识别任务：

```bash
python src/process_form.py \
  --input data/input/sample.jpeg \
  --config config/template_config.json \
  --output-dir data/output/process_structured \
  --lexicon config/domain_lexicon_demo.json
```

这个流程会先执行：

1. 图像与模板配准
2. 固定区域裁切
3. 主表切格
4. 标题区、备注区、底部签名区 OCR
5. 主表上半区和飞行记录结构化整理
6. 生成 `ocr_result.json` 和结构化 `report.txt`



## 输出文件说明

`process_form.py`，输出目录通常都会包含以下内容：

- `aligned.png`
  与模板配准后的整图。
- `title.png`
  标题区域裁剪图。
- `main_table.png`
  主表区域裁剪图。
- `remark.png`
  备注区域裁剪图。
- `bottom.png`
  底部签名区域裁剪图。
- `main_table_grid_debug.png`
  主表网格调试图。
- `cells/`
  主表按固定网格切出的单元格图像。
- `metadata.json`
  对齐结果、区域框、网格信息、单元格文件索引。
- `ocr_result.json`
  结构化 OCR 结果。
- `report.txt`
  便于人工查看的文字报告。

## `process_form.py` 的结果重点

`process_form.py` 输出的 `report.txt` 目前重点整理以下内容：

- 标题信息
- 备注区信息
- 主表顶部编组/人员信息
- 主表飞行记录
- 底部签名信息

其中主表飞行记录会尽量按记录项输出：

- `机型`
- `机号`
- `二次代码`
- `关联代字代号`
- `飞行片段`
- `飞行片段中的时间、驾驶代字代号、标记、备注`

由于主表中部字体很小、密度很高，个别时间和代字代号仍可能需要人工复核。

## 词表与识别修正

项目支持通过词表对 OCR 结果进行近似匹配修正。推荐将以下内容维护到词表 JSON 中：

- 姓名
- 代字代号
- 飞行代码
- 机型
- 机号
- 常见备注缩写

默认示例词表位于：

```text
config/domain_lexicon_demo.json
```

你可以在运行 `process_form.py` 或 `run_full_pipeline.py` 时通过 `--lexicon` 指定自己的词表文件。

## 配置文件说明

模板配置文件 `config/template_config.json` 中最重要的部分包括：

- `template`
  模板图路径和尺寸。
- `alignment`
  ORB 特征点配准参数。
- `regions`
  标题区、主表区、备注区、底部区的固定框。
- `grid`
  主表网格线位置。
- `semantic`
  标题区、备注区、底部区和主表结构化所需的语义配置。

如果识别区域略有偏差，优先微调：

- `regions`
- `grid`
- `semantic.main_table_schema`

## 调试建议

当识别不理想时，建议按以下顺序检查：

1. 先看 `aligned.png` 是否对齐正确。
2. 再看 `main_table.png` 是否完整覆盖主表。
3. 再看 `main_table_grid_debug.png` 的网格线是否贴合。
4. 如果主表细粒度内容不稳定，检查词表是否足够完整。
5. 如需排查语义框 OCR，可使用 `src/debug_semantic_boxes.py`。

## 当前建议

- 同版式批量处理时，优先使用 `process_form.py`。
- 如果你修改了主表结构化逻辑，请同步检查 `structured_main_table.py` 和 `structured_report.py`。
- 如果你调整了模板或网格，请重新生成 `template_config.json`。
