# 数据准备与训练入口

本文档是当前数据准备、gate 和训练入口的 source of truth。当前阶段只要求 CICIDS2017 baseline 和 PCAP 推理 demo 稳定，不处理 IoT-23/CTU-13。

## 1. CICIDS2017 baseline

正式 baseline 使用 CICIDS2017 全量 Monday-Friday working-hour PCAP 和官方 labelled-flow CSV。任何 Friday-only、smoke、payload-only 旧产物都不能作为正式训练输入。

正式预处理要求：

- 扫完整 PCAP，`WINDOW_SCOPE=pcap`。
- 使用 `masked_header_packet`。
- 保留空 payload flow。
- `flow_timeout_seconds=120`。
- CSV timestamp offset 使用当前已验证的 `+3h`。
- 每个攻击标签通过 attack coverage gate。
- Split 必须通过 bytes/window/nearby five-tuple/submode coverage audit。

当前可训练主 split：

```text
data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/
```

压力测试 split 在需要时用 `--include-nonformal-splits` 额外生成：

```text
data/processed/cicids2017/all_masked_header_split_time_block_cap32_notcpclose/
```

## 2. 从零准备 CICIDS2017

Linux CUDA 服务器：

```bash
HF_ENDPOINT=https://hf-mirror.com DATASET=cicids2017-all bash scripts/download_datasets.sh
SKIP_DOWNLOAD=1 FORCE=1 bash scripts/prepare_cicids2017_all_parallel.sh
```

Windows：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/prepare_datasets.ps1 `
  -Dataset cicids2017-all `
  -ForceBuild
```

关键产物：

```text
data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose/flows_all.parquet
data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/train.parquet
data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/val.parquet
data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/test.parquet
artifacts/cicids2017_coverage_masked_header_cap32_notcpclose/attack_coverage.json
artifacts/cicids2017_all_masked_header_cap32_notcpclose/submode_stratified_group_split_audit.json
artifacts/cicids2017_all_masked_header_cap32_notcpclose/formal_dataset_gate.json
```

## 3. Gate 规则

训练前必须通过：

- 五个 PCAP 均存在且非空。
- `flows_all.parquet` 存在。
- 每个 shard 的 build metadata 包含 `flow_timeout_seconds=120`、`close_on_tcp_flags=false`、`tcp_close_policy_version=3`、`connection_type_version=1`、`window_scope=pcap`、`csv_time_offset_hours=3.0`。TCP FIN/RST 信息仍保留在 packet/connection_type 中，但不再把关闭/拒绝包拆成独立训练 flow。
- `attack_coverage.json.formal_eligible=true`。
- split audit `formal_eligible=true` 且 `blocking_warnings=[]`。
- 训练入口不得使用 `--skip-audit`。

正式 gate 命令由训练脚本自动调用，也可单独运行：

```bash
uv run python scripts/verify_formal_dataset.py \
  --train-path data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/train.parquet \
  --val-path data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/val.parquet \
  --test-path data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/test.parquet \
  --processed-dir data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose \
  --merged-path data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose/flows_all.parquet \
  --attack-coverage-path artifacts/cicids2017_coverage_masked_header_cap32_notcpclose/attack_coverage.json \
  --required-split-parent data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose \
  --audit-path artifacts/cicids2017_all_masked_header_cap32_notcpclose/submode_stratified_group_split_audit.json \
  --required-view masked_header_packet \
  --required-keep-empty-payload true
```

## 4. 训练入口

Linux CUDA：

```bash
PLAN=bert-supervised \
OUTPUT_ROOT=artifacts/cicids2017_masked_header_submode_group_formal_cuda_v16_cap32_notcpclose_conn_b128_w4_sqrt_weighted \
TRAIN_PATH=data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/train.parquet \
VAL_PATH=data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/val.parquet \
TEST_PATH=data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/test.parquet \
CICIDS_PROCESSED_DIR=data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose \
CICIDS_MERGED_PATH=data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose/flows_all.parquet \
CICIDS_ATTACK_COVERAGE_PATH=artifacts/cicids2017_coverage_masked_header_cap32_notcpclose/attack_coverage.json \
CICIDS_REQUIRED_SPLIT_PARENT=data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose \
CICIDS_SPLIT_AUDIT_PATH=artifacts/cicids2017_all_masked_header_cap32_notcpclose/submode_stratified_group_split_audit.json \
CLASSIFIER_EPOCHS=3 \
CLASSIFIER_BATCH_SIZE=128 \
CLASSIFIER_NUM_WORKERS=4 \
MAJOR_CLASS_WEIGHTING=sqrt_balanced \
USE_CONNECTION_TOKENS=1 \
MAX_LENGTH=512 \
MAX_WINDOWS=2 \
DEVICE=auto \
bash scripts/train_cicids_formal_cuda.sh
```

Windows：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/train_cicids_formal.ps1
```

训练结束应写出：

```text
formal_dataset_gate.json
split_audit.json
classifier/history/history.json
classifier/eval/metrics.json
classifier/eval/classification_report.json
classifier/eval/confusion_matrix.csv
formal_run_summary.json
```

当前任何 high accuracy 结果都必须结合 macro-F1、per-class recall/F1 和 confusion matrix 解释。

当前最强 flow-level 结果来自 `connection tokens + sqrt_balanced` 的 v16 run。该 run 使用 no-close 数据口径，Bot test precision/recall/F1 约 `0.615/0.995/0.760`，达到当前毕设可交付口径。`context tokens`、`balanced` loss、numeric context side-channel、SCE v0 frequency 和 Bot label-lift codebook 均已因 Bot 误报过多或低于 baseline 而降级。

后续如果还要提升 Bot，优先考虑更明确的 host/session-window 辅助任务或真正语义化的 SCE token；继续盲调 class weight、context token 或 Bot 阈值的收益已经被当前实验否定。

## 5. PCAP 推理 demo 后端

已有 CLI 可作为展示系统后端原型：

```bash
uv run traffic-bert predict pcap sample.pcap \
  --checkpoint artifacts/classifier/classifier.best.pt \
  --view masked_header_packet \
  --max-windows 2 \
  --summary-output artifacts/demo/sample_prediction.json
```

该命令 stdout 和 `--summary-output` 都会写出 `{summary, flows}`，其中 `summary` 包含 flow 总数、预测攻击 flow 数、各 major label 计数、connection type 计数、高风险 flow 列表和 top host risk 汇总。`--output` 仍可额外写出 flow-level JSONL。

最小上传页面由标准库脚本提供，不引入额外 Web 框架：

```bash
uv run python scripts/pcap_demo_server.py \
  --checkpoint artifacts/cicids2017_masked_header_submode_group_formal_cuda_v16_cap32_notcpclose_conn_b128_w4_sqrt_weighted/classifier/classifier.best.pt \
  --host 127.0.0.1 \
  --port 7860 \
  --device auto
```

在 5090 服务器上运行并本地转发：

```bash
# remote: /root/Fork
.venv/bin/python scripts/pcap_demo_server.py \
  --checkpoint artifacts/cicids2017_masked_header_submode_group_formal_cuda_v16_cap32_notcpclose_conn_b128_w4_sqrt_weighted/classifier/classifier.best.pt \
  --host 127.0.0.1 \
  --port 7861 \
  --device auto \
  --risk-threshold 0.9

# local
ssh -i ./<ssh_key> -N -L 7861:127.0.0.1:7861 -p <ssh_port> <user>@<server>
```

浏览器打开 `http://127.0.0.1:7861/` 后上传 `.pcap` 或 `.pcapng`。当前实现会在进程内复用模型，避免每次上传都重新加载 checkpoint。v16 flow-level Bot 已通过；host-window 只保留为可选风险汇总，不作为替代 flow-level 指标。远端 GPU 冒烟已用一个小 PCAP 验证上传接口。

第一版展示系统只承诺 PCAP/PCAPNG 上传，输出 flow 级预测和文件级风险汇总。Web UI 不应绕过 `PcapFlowExtractor` 或重新实现一套解析逻辑。

## 6. SCE v0 codebook

当前 SCE v0 可以构建 codebook，并可接入 classifier train/eval/predict；尚未作为正式结论：

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

默认 `--drop-zero-chunks` 会跳过全零 chunk，避免 masked-header 的零填充成为最高频 token。该 codebook 只是 SCE 最小骨架，不代表最终 SCE 效果。

SCE v0 对照训练入口：

```bash
SEMANTIC_CODEBOOK_PATH=artifacts/sce/cicids_v0_frequency_codebook_sample.json \
PLAN=bert-supervised \
USE_CONNECTION_TOKENS=1 \
MAJOR_CLASS_WEIGHTING=sqrt_balanced \
bash scripts/train_cicids_formal_cuda.sh
```

checkpoint 会嵌入 codebook 元数据，因此后续 `traffic-bert eval classifier` 和 `traffic-bert predict pcap` 不显式传 `--semantic-codebook-path` 也可以从 checkpoint 自动恢复。

## 7. USTC 和 Payload-Byte

USTC-TFC2016：

```bash
DATASET=ustc_tfc2016 bash scripts/download_datasets.sh
DATASET=ustc_tfc2016 bash scripts/prepare_datasets.sh
```

USTC 只用于辅助诊断：

- `split_source_file_major_balanced`：benign/malware 大类诊断。
- `split_source_file`：严格文件级 unseen-family 诊断。
- `split_label_stratified`：sanity/upper-bound，不作为正式泛化结论。

Payload-Byte 是单包 payload 行级数据，不含完整五元组和双向 flow 边界。当前不混入 CICIDS 监督主训练，可作为后续自监督或 sanity 候选。

## 8. 当前不处理的数据集

IoT-23 和 CTU-13 是后续泛化验证数据。当前阶段不下载、不预处理、不训练，避免分散 CICIDS baseline、SCE 设计和 demo 交付。

## 9. 运行后必须检查

- `split_audit.json` 是否 `formal_eligible=true`。
- `formal_dataset_gate.json` 是否 `formal_eligible=true`。
- `metrics.json` 中 macro-F1、balanced accuracy、per-class recall 是否合理。
- Bot 等关键类是否被大量判成 benign。
- Heartbleed/Infiltration 是否按 low-support reported-only 解释。
