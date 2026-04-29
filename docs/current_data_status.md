# 当前数据处理状态

本文档记录截至 2026-04-29 已完成的数据下载、预处理、标签修复、合并划分和下一步训练决策。大文件产物均位于 `data/`，不进入 Git。

## 1. 已处理数据集

### USTC-TFC2016

当前作为主监督训练数据。

- 原始位置：`data/raw/USTC-TFC2016/`
- 处理后单文件：`data/processed/ustc_tfc2016/files/`
- 重标后单文件：`data/processed/ustc_tfc2016/relabelled_files/`
- 合并文件：`data/processed/ustc_tfc2016/merged/all.parquet`
- 主训练划分：`data/processed/ustc_tfc2016/split_label_stratified/`
- 文件级泛化对照：`data/processed/ustc_tfc2016/split_source_file/`
- 摘要：`data/processed/ustc_tfc2016/merge_split.summary.json`

处理方式：

- 从 PCAP 顺序解析 flow。
- 当前只生成 `payload_only` 视图。
- 每个 flow 最多保留 16 个 packet。
- 丢弃空 payload flow。
- 使用低优先级、低线程、单进程方式处理，避免占满 CPU。

最终合并结果：

```text
rows: 407594
flows: 407594
view: payload_only
major_labels:
  benign: 309887
  botnet_malware: 97707
```

细分类：

```text
cridex: 8196
geodo: 6707
htbot: 6322
miuref: 4952
neris: 11011
nsis: 6065
shifu: 9581
tinba: 8503
virut: 25593
zeus: 10777
```

主训练划分采用 `source_label` 分层、`flow_id` 稳定划分。这样每个恶意家族都进入 train/val/test：

```text
split   rows
train   285310
val      61134
test     61150
```

文件级泛化对照采用 `source_file` 划分。这个划分能避免同一 PCAP 跨集合，但 USTC 每个恶意家族基本只有一个 PCAP，因此很多细类不会同时出现在 train/val/test。它不作为主训练划分，只作为泛化对照。

### Payload-Byte

当前作为 MLM post-train 的候选数据，不作为第一阶段主监督训练数据。

- 原始位置：`data/raw/Payload-Byte/`
- 处理后文件：`data/processed/payload_byte/`
- 综合划分：`data/processed/payload_byte/split/`

合并结果：

```text
rows: 1490132
flows: 1490132
view: payload_only
packet_count:
  min: 1
  median: 1
  max: 1
```

大类分布：

```text
benign: 383104
botnet_malware: 4963
dos_ddos: 824563
bruteforce: 80008
heartbleed: 13486
infiltration: 115007
scan: 8392
web_attack: 15107
other_attack: 45502
```

注意：Payload-Byte 是单包 payload 行级数据，不含完整五元组和双向 flow 边界。它与 USTC 的 flow 级样本语义不同，因此不直接混入第一阶段监督训练集。

## 2. 标签修复和分类决策

统一标签字段：

```text
source_label -> major_label -> minor_labels
```

大类是单选，细类是 multi-hot。当前 USTC 中良性样本没有细类：

```text
minor_labels=[]: 309887
benign 中无细类: 309887
botnet_malware 中有细类: 97707
```

已修复的关键问题：

- 之前 `Htbot` 会被通用 `Bot|Botnet` 规则误映射为 `botnet`。
- 现在 `Htbot -> botnet_malware / htbot`。
- 通用 botnet 规则已收紧为单词边界匹配，避免误伤包含 `bot` 子串的家族名。

当前 USTC 输出逻辑：

```text
benign:
  minor_labels = []

botnet_malware:
  minor_labels in [cridex, geodo, htbot, miuref, neris, nsis, shifu, tinba, virut, zeus]
```

## 3. 当前不做综合监督集的原因

当前不直接把 USTC 和 Payload-Byte 合成一个监督训练集。

原因：

- USTC 是 flow 级 PCAP 重组样本。
- Payload-Byte 是单包 payload 行级样本。
- 直接混合监督训练时，模型容易学习数据集来源差异，而不是攻击语义。
- Payload-Byte 类别更丰富，但与 USTC 的样本粒度不一致，应先用于自监督 post-train。

当前推荐路线：

```text
1. MLM post-train:
   USTC + Payload-Byte 的 payload_only 字节序列

2. 监督 fine-tune:
   USTC split_label_stratified

3. 辅助监督融合:
   在主链路稳定后，再尝试加入 Payload-Byte，并控制 source_dataset 权重
```

## 4. 后续数据集优先级

### 优先级 1：CICIDS2017 原始 PCAP + CSV 标签

目的：

- 补齐 `dos_ddos`、`bruteforce`、`web_attack`、`scan`、`infiltration`、`heartbleed` 等大类。
- 与 Payload-Byte 版 CICIDS2017 相互验证。

主要难点：

- PCAP flow 与官方 CSV flow 标签对齐。
- 时间戳、五元组、协议字段格式需要严格统一。

### 优先级 2：CIC-DDoS2019

目的：

- 扩展 DDoS 相关细类。
- 增强 `dos_ddos` 下的分类能力。

风险：

- 数据较大，处理时间和磁盘占用更高。

### 优先级 3：CSE-CIC-IDS2018

目的：

- 类别丰富，适合后期扩展和 post-train。

风险：

- 规模很大，完整 raw PCAP 处理成本高。

### 优先级 4：CTU-13

目的：

- 可作为 botnet 外部泛化测试。

风险：

- normal/background 与 botnet PCAP 的 payload 完整性不完全一致，不能直接作为主监督训练集。

## 5. 验证状态

已完成验证：

```text
uv run pytest
32 passed
```

数据校验：

- `data/processed/ustc_tfc2016/merged/all.parquet` 通过。
- `split_label_stratified/train.parquet` 通过。
- `split_label_stratified/val.parquet` 通过。
- `split_label_stratified/test.parquet` 通过。
- `split_source_file/train.parquet` 通过。
- `split_source_file/val.parquet` 通过。
- `split_source_file/test.parquet` 通过。

最新相关代码提交：

```text
42c3c19 feat: add stratified traffic splits
1c714e0 chore: add low impact ustc preprocessing script
c7521f9 build: default training device to cuda
```
