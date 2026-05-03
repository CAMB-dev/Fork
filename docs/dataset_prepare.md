# 数据集下载与处理总控脚本

本项目提供 Windows PowerShell 和 Linux Bash 两个总控入口，用于串联下载、预处理、划分、校验和 split audit。

## CICIDS2017 Friday Smoke

默认会通过 `HF_ENDPOINT=https://hf-mirror.com` 下载 Friday PCAP 和 `GeneratedLabelledFlows.zip`，只处理小窗口 smoke 数据：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/prepare_datasets.ps1 `
  -Dataset cicids2017-friday-smoke
```

Linux/ROCm 服务器：

```bash
HF_ENDPOINT=https://hf-mirror.com DATASET=cicids2017-friday-smoke bash scripts/prepare_datasets.sh
```

常用参数：

```bash
MAX_PACKETS_TO_READ=250000
MAX_PACKETS_TO_SKIP=0
MAX_PACKETS_PER_FLOW=16
MAX_PER_MAJOR=2000
LABEL_FILE_CONTAINS=DDos
START_TIME=2017-07-07T12:34:00Z
END_TIME=2017-07-07T12:45:00Z
RUN_SMOKE_TRAIN=1
```

`LABEL_FILE_CONTAINS` 用于只选择 Friday 的某个官方 labelled flow CSV，例如 `Morning`、`PortScan`、`DDos`。如果 smoke 结果只有 `benign`，先不要解读模型效果，应换用攻击时段或增加 `MAX_PACKETS_TO_SKIP/MAX_PACKETS_TO_READ` 重新生成。

输出：

```text
data/raw/CICIDS2017/
data/processed/cicids2017/friday_payload_only/
data/processed/cicids2017/smoke/
artifacts/cicids2017_smoke/
```

## USTC-TFC2016

Linux Bash 脚本会默认从公开 GitHub archive 下载 USTC，并尝试自动解压其中的 `.7z` PCAP 包。服务器需要安装 `unzip`，并建议安装 `7zip` 或 `p7zip-full`：

```bash
sudo apt-get update
sudo apt-get install -y unzip p7zip-full
```

Windows PowerShell 脚本仍主要负责处理已解压目录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/prepare_datasets.ps1 -Dataset ustc
```

Linux/ROCm 服务器：

```bash
DATASET=ustc_tfc2016 bash scripts/prepare_datasets.sh
```

如果解压目录不同：

```bash
USTC_SOURCE_ROOT=/data/USTC-TFC2016-master DATASET=ustc_tfc2016 bash scripts/prepare_datasets.sh
```

输出：

```text
data/processed/ustc_tfc2016/files/
data/processed/ustc_tfc2016/merged/payload_only.parquet
data/processed/ustc_tfc2016/split_label_stratified/
data/processed/ustc_tfc2016/split_source_file/
artifacts/ustc_tfc2016/
```

`split_label_stratified` 适合主训练 sanity，但存在同源风险；`split_source_file` 用于文件级泛化对照，可能出现 val/test 类别缺失。每次运行都会通过 `scripts/audit_processed_split.py` 输出 audit 报告。

## 5090/CUDA 从零运行

Linux CUDA 服务器上先安装系统解压工具，再下载和预处理 USTC-TFC2016：

```bash
sudo apt-get update
sudo apt-get install -y unzip p7zip-full

DATASET=ustc_tfc2016 bash scripts/download_datasets.sh
DATASET=ustc_tfc2016 bash scripts/prepare_datasets.sh
```

严格 source-file split 的 supervised BERT 训练：

```bash
CUDA_VISIBLE_DEVICES=0 \
PLAN=bert-supervised \
TRAIN_PATH=data/processed/ustc_tfc2016/split_source_file/train.parquet \
VAL_PATH=data/processed/ustc_tfc2016/split_source_file/val.parquet \
TEST_PATH=data/processed/ustc_tfc2016/split_source_file/test.parquet \
OUTPUT_ROOT=artifacts/ustc_bert_source_file_cuda \
CLASSIFIER_EPOCHS=3 \
CLASSIFIER_BATCH_SIZE=16 \
MAX_LENGTH=512 \
MAX_WINDOWS=2 \
DEVICE=auto \
bash scripts/train_ustc_formal_cuda.sh
```

## 全部处理

Windows：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/prepare_datasets.ps1 `
  -Dataset all
```

Linux：

```bash
HF_ENDPOINT=https://hf-mirror.com DATASET=all bash scripts/prepare_datasets.sh
```

## 跳过下载或强制重建

Windows：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/prepare_datasets.ps1 `
  -Dataset cicids2017-friday-smoke `
  -SkipDownload `
  -ForceBuild
```

Linux：

```bash
DATASET=cicids2017-friday-smoke SKIP_DOWNLOAD=1 FORCE_BUILD=1 bash scripts/prepare_datasets.sh
```

## 运行后必须检查

- `*.validate.json` 或 `traffic-bert data validate` 输出无严重错误。
- `split_audit.json` 中是否存在 `source_file overlap` 或类别缺失 warning。
- 训练前确认使用的是目标 split，而不是误用旧的同源 sanity split。
