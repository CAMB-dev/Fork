# PCAP-first SCE 恶意流量检测

本项目是一个毕设工程，目标是构建一套面向原始网络流量的恶意流量检测系统。当前阶段先完成可复现、可审计、可展示的 CICIDS2017 baseline；论文核心创新 SCE（Semantic Conversion Encoder，语义转换编码器）作为下一阶段主线，在稳定的数据和评估链路上继续实现。

## 当前目标

- PCAP-first：优先从原始 PCAP 重组 flow，而不是依赖已经整理好的序列数据集。
- Level-0/1/2 分类：良性/恶性、攻击家族、攻击子类逐层输出。
- 方案1 baseline：使用 byte/header token 化 + Byte-BERT，验证完整训练、评估和推理链路。
- 方案2 SCE：将 bytes、header fields 和 flow pattern 转换为更稳定的语义 token/codebook，作为论文创新点。
- 方案3 泛化增强：在 SCE 或 Byte-BERT 基础上做自监督预训练，并在 IoT-23、CTU-13 等数据集上做后续泛化验证。
- 展示系统：第一版支持上传 PCAP/PCAPNG，自动解析 flow 并输出 flow 级预测和整体风险汇总。

## 当前阶段边界

- 主数据集：CICIDS2017 全量 Monday-Friday PCAP + 官方 labelled-flow CSV。
- 当前可信 baseline split：`all_masked_header_split_submode_stratified_group_cap32_notcpclose`。
- `time_block`：只作为时间外推和子形态外推压力测试，不作为当前可训练主线。
- USTC/Payload-Byte：仅作为 sanity、诊断或辅助预训练候选，不作为正式主结论。
- IoT-23/CTU-13：保留为后续泛化验证，不阻塞当前阶段交付。

## 核心链路

```text
PCAP/label
  -> flow 重组与标签对齐
  -> payload/full/masked-header 输入视图
  -> leakage/coverage/split audit
  -> 方案1 Byte-BERT baseline
  -> macro/per-class/关键类 recall 验收
  -> PCAP 上传推理 demo
  -> 后续 SCE 语义 token/codebook
```

## 关键文档

- [docs/project_decisions.md](docs/project_decisions.md)：当前最高层技术决策和阶段目标。
- [docs/model_design.md](docs/model_design.md)：Byte-BERT baseline、SCE 定位和展示系统推理形态。
- [docs/experiment_plan.md](docs/experiment_plan.md)：当前实验、验收指标和后续泛化路线。
- [docs/dataset_prepare.md](docs/dataset_prepare.md)：CICIDS2017/USTC 数据下载、预处理、gate 和训练入口。
- [docs/data_pipeline.md](docs/data_pipeline.md)：PCAP-first 数据管线、schema 和 audit 规则。
- [docs/current_data_status.md](docs/current_data_status.md)：当前本地/远端数据状态和已知问题。
- [docs/current_goal_audit.md](docs/current_goal_audit.md)：当前目标完成证据、缺口和下一步。

## 常用命令

```powershell
uv run traffic-bert version
uv run traffic-bert vocab write artifacts/vocab.txt
uv run traffic-bert data build --input-path data/raw/sample.pcap --output-path data/processed/train.parquet --source-dataset custom --label-source filename
uv run traffic-bert predict pcap data/raw/sample.pcap \
  --checkpoint artifacts/classifier/classifier.best.pt \
  --summary-output artifacts/demo/sample_prediction.json
```

Linux CUDA 服务器上的 CICIDS2017 baseline 链路：

```bash
HF_ENDPOINT=https://hf-mirror.com DATASET=cicids2017-all bash scripts/download_datasets.sh
SKIP_DOWNLOAD=1 FORCE=1 bash scripts/prepare_cicids2017_all_parallel.sh
PLAN=bert-supervised USE_CONNECTION_TOKENS=1 MAJOR_CLASS_WEIGHTING=sqrt_balanced bash scripts/train_cicids_formal_cuda.sh
```

当前 cap32/notcpclose 产物的完整 `TRAIN_PATH`、`CICIDS_PROCESSED_DIR`、coverage 和 audit 环境变量见 [docs/dataset_prepare.md](docs/dataset_prepare.md)。

PCAP 上传 demo：

```bash
uv run python scripts/pcap_demo_server.py \
  --checkpoint artifacts/cicids2017_masked_header_submode_group_formal_cuda_v16_cap32_notcpclose_conn_b128_w4_sqrt_weighted/classifier/classifier.best.pt \
  --port 7860 \
  --device auto
```

服务器 GPU + 本地端口转发的启动方式见 [docs/dataset_prepare.md](docs/dataset_prepare.md)。

SCE v0 codebook 样例：

```bash
uv run traffic-bert sce build-codebook \
  --input-path data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/train.parquet \
  --output-path artifacts/sce/cicids_v0_frequency_codebook_sample.json \
  --view masked_header_packet \
  --chunk-size 4 \
  --max-entries 64 \
  --min-count 5 \
  --max-rows 20000
```

SCE v0 对照 run 可通过 `SEMANTIC_CODEBOOK_PATH=...` 接入现有 formal 训练脚本。当前还没有把 SCE v0 指标写成正式结论。

label-lift codebook 支持按标签 reservoir 抽样：

```bash
uv run traffic-bert sce build-codebook \
  --input-path data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/train.parquet \
  --output-path artifacts/sce/cicids_label_lift_bot_codebook.json \
  --view masked_header_packet \
  --ranking label_lift \
  --target-label botnet_malware \
  --chunk-size 4 \
  --max-entries 64 \
  --min-count 20 \
  --max-rows-per-label 6000
```

当前 v14 高频 codebook 和 v15 Bot label-lift codebook 都低于当前 Byte-BERT baseline，不能作为正式提升结论。

Windows 入口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/prepare_datasets.ps1 `
  -Dataset cicids2017-all `
  -ForceBuild
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/train_cicids_formal.ps1
```

## 结果解释原则

- Accuracy 和 weighted-F1 不能单独作为通过标准。
- Support 足够的攻击类必须报告 per-class precision/recall/F1。
- Bot 等关键类 recall/F1 不达标时，run 必须标记为 failed 或 needs review。
- 当前 v16 flow-level baseline 已让 Bot 达到毕设可交付口径；host-window 诊断只作为展示/告警层，不覆盖 flow-level 指标。
- Heartbleed/Infiltration 等极低样本类保留在数据和报告中，但作为 low-support reported-only，不作为当前可训练主类别验收。
- `time_block` 压力测试失败时，应解释为时间/子形态外推困难，不等同于 baseline 数据链路无效。

## 环境约定

本仓库固定使用 Python 3.11 和 `uv` 管理环境。大文件、原始数据、处理后 parquet、模型权重和实验产物不进入 Git。

```powershell
uv python install 3.11
uv sync
uv run pytest
```
