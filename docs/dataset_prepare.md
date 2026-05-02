# 数据集下载与处理总控脚本

本项目提供 Windows PowerShell 和 Linux Bash 两个总控入口，用于串联下载、预处理、划分、校验和 split audit。

## CICIDS2017 Friday Smoke

默认会通过 HuggingFace mirror 下载 Friday PCAP 和 `GeneratedLabelledFlows.zip`，只处理小窗口 smoke 数据：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/prepare_datasets.ps1 `
  -Dataset cicids2017-friday-smoke `
  -Proxy http://127.0.0.1:7890
```

Linux/ROCm 服务器：

```bash
PROXY=http://127.0.0.1:7890 DATASET=cicids2017-friday-smoke bash scripts/prepare_datasets.sh
```

常用参数：

```bash
MAX_PACKETS_TO_READ=250000
MAX_PACKETS_TO_SKIP=0
MAX_PACKETS_PER_FLOW=16
MAX_PER_MAJOR=2000
START_TIME=2017-07-07T12:34:00Z
END_TIME=2017-07-07T12:45:00Z
RUN_SMOKE_TRAIN=1
```

输出：

```text
data/raw/CICIDS2017/
data/processed/cicids2017/friday_payload_only/
data/processed/cicids2017/smoke/
artifacts/cicids2017_smoke/
```

## USTC-TFC2016

USTC 目前没有在脚本中绑定下载直链。先手动把原始数据解压到：

```text
data/raw/USTC-TFC2016/extracted/USTC-TFC2016-master/
```

然后运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/prepare_datasets.ps1 -Dataset ustc
```

Linux/ROCm 服务器：

```bash
DATASET=ustc bash scripts/prepare_datasets.sh
```

如果解压目录不同：

```bash
USTC_SOURCE_ROOT=/data/USTC-TFC2016-master DATASET=ustc bash scripts/prepare_datasets.sh
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

## 全部处理

Windows：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/prepare_datasets.ps1 `
  -Dataset all `
  -Proxy http://127.0.0.1:7890
```

Linux：

```bash
DATASET=all PROXY=http://127.0.0.1:7890 bash scripts/prepare_datasets.sh
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
