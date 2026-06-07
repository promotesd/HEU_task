# Table OCR Project

这个项目用于处理固定模板的军用计划板图像。当前代码已经拆分为“模板对齐与切分”和“结构化主表识别”两层，适合对同一类版式图片进行批量 OCR、表格结构化和中文报告输出。

## 项目目标

给定一张固定模板的表单图片，完成以下流程：

1. 以清晰样本图生成模板配置。
2. 将待识别图片与模板自动配准。
3. 按固定区域裁切 `title`、`main_table`、`remark`、`bottom`。
4. 对主表网格、标题区、备注区、底部签名区做 OCR。
5. 输出结构化 JSON 和可用 Excel/WPS 打开的 `report.xml`。

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
    ├── ablation_profile.py
    ├── process_form.py
    ├── run.sh
    └── table_ocr_project/
        ├── __init__.py
        ├── alignment.py
        ├── config_utils.py
        ├── grid.py
        ├── layout.py
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
- `src/ablation_profile.py`
  对语义区 OCR 候选策略做保守消融，比较耗时和输出一致性。
- `src/process_form.py`
  当前推荐入口。默认走快速模式，跳过中间图片落盘，只输出 `metadata.json`、`ocr_result.json` 和 `report.xml`。
- `src/run.sh`
  `process_form.py` 的示例运行脚本。

包内核心模块说明：

- `pipeline.py`
  模板配置生成、图像对齐、区域裁切、网格 metadata 生成和可选调试图输出。
- `structured_process.py`
  结构化流程总入口，串联主表结构化识别和结构化报告输出。
- `structured_main_table.py`
  主表上半区和飞行记录的结构化整理逻辑。
- `structured_report.py`
  将结构化结果写成可用 Excel/WPS 打开的 `report.xml`。
- `semantic_extractors.py`
  标题区、备注区、底部签名区的 OCR 抽取逻辑。

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
3. 主表网格 metadata 生成
4. 标题区、备注区、底部签名区 OCR
5. 主表上半区和飞行记录结构化整理
6. 生成 `metadata.json`、`ocr_result.json` 和结构化 `report.xml`

默认快速模式不会保存 `aligned.png`、区域裁剪图、网格调试图和 `cells/` 单元格图片，以减少磁盘 I/O。需要调试图片时可以显式开启：

```bash
python src/process_form.py \
  --input data/input/sample.jpeg \
  --config config/template_config.json \
  --output-dir data/output/process_structured \
  --lexicon config/domain_lexicon_demo.json \
  --debug-output
```

如果确实需要保存全部 2625 张主表单元格图片，再加 `--save-cells`：

```bash
python src/process_form.py \
  --input data/input/sample.jpeg \
  --config config/template_config.json \
  --output-dir data/output/process_structured \
  --lexicon config/domain_lexicon_demo.json \
  --debug-output \
  --save-cells
```

需要查看耗时拆分时，可以加 `--profile`：

```bash
python src/process_form.py \
  --input data/input/sample.jpeg \
  --config config/template_config.json \
  --output-dir data/output/process_structured \
  --lexicon config/domain_lexicon_demo.json \
  --profile
```

`--profile` 会输出启动/import、参数解析、workflow 总耗时、已统计阶段合计、workflow 未覆盖耗时，以及对齐、裁切、OCR、报告写入等阶段耗时。

默认语义区 OCR 使用已在 `sample.jpeg` 上消融验证通过的 `fast` 候选策略，减少重复 OCR 调用。该策略已验证与旧版多候选模式的 `ocr_result.json` 和 `report.xml` 输出一致；语义区 OCR 调用数从 197 次降到 50 次，整张图总 OCR 调用数为 51 次。

当前服务器环境下，`sample.jpeg` 默认模式连续 3 次独立进程运行平均约 `18.4s`，最慢约 `18.6s`。如果需要恢复旧版多候选 OCR 路径进行复核，可以加：

```bash
python src/process_form.py \
  --input data/input/sample.jpeg \
  --config config/template_config.json \
  --output-dir data/output/process_structured \
  --lexicon config/domain_lexicon_demo.json \
  --strict-ocr-candidates \
  --profile
```

如需重新做候选策略消融，可以运行：

```bash
python src/ablation_profile.py \
  --input data/input/sample.jpeg \
  --config config/template_config.json \
  --output-root /tmp/table_ocr_ablation \
  --lexicon config/domain_lexicon_demo.json
```



## 输出文件说明

`process_form.py` 默认快速模式只保留最终结果和必要 metadata：

- `metadata.json`
  对齐结果、区域框、网格信息、单元格 row/col/bbox 索引。
- `ocr_result.json`
  结构化 OCR 结果。
- `report.xml`
  便于人工查看和表格软件打开的结构化报告。

开启 `--debug-output` 后，会额外输出：

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

开启 `--debug-output --save-cells` 后，会额外输出：

- `cells/`
  主表按固定网格切出的单元格图像。

## `process_form.py` 的结果重点

`process_form.py` 输出的 `report.xml` 目前重点整理以下内容：

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

你可以在运行 `process_form.py` 时通过 `--lexicon` 指定自己的词表文件。

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

1. 如果默认快速模式识别不理想，先用 `--debug-output` 重新跑一遍。
2. 看 `aligned.png` 是否对齐正确。
3. 再看 `main_table.png` 是否完整覆盖主表。
4. 再看 `main_table_grid_debug.png` 的网格线是否贴合。
5. 如果主表细粒度内容不稳定，检查词表是否足够完整。

## 当前建议

- 同版式批量处理时，优先使用 `process_form.py`。
- 如果你修改了主表结构化逻辑，请同步检查 `structured_main_table.py` 和 `structured_report.py`。
- 如果你调整了模板或网格，请重新生成 `template_config.json`。
