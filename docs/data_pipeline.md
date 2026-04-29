# 数据预处理流水线设计

数据预处理是本项目的第一大工程模块。它的目标不是简单格式转换，而是把不同来源、不同标签体系、不同原始格式的流量数据统一为可训练、可评估、可复现的 flow 级数据。

## 1. 输入数据

第一阶段数据源：

- USTC-TFC2016：按 benign/malware 和应用或恶意家族组织的 PCAP 数据。
- CICIDS2017：官方 PCAP、flow 标签和 CSV 标注。
- Payload-Byte：可作为 sanity check 或备用来源，不作为主线。

后续候选：

- CSE-CIC-IDS2018。
- CIC-DDoS2019。

## 2. 原始数据目录

本地目录约定：

```text
data/raw/
├── USTC-TFC2016/
├── CICIDS2017/
├── CSE-CIC-IDS2018/
└── CIC-DDoS2019/
```

每个数据集应有一个 manifest，记录：

- 数据来源 URL。
- 原始文件名。
- 标签文件名。
- 下载时间。
- checksum。
- 数据集版本。
- 处理状态。

manifest 进入 Git，原始大文件不进入 Git。

## 3. PCAP 解析

解析目标：

- 读取每个 packet 的时间戳。
- 解析 Ethernet/IP/TCP/UDP 层。
- 提取五元组。
- 提取方向。
- 提取 payload bytes。
- 提取完整 packet bytes。
- 构建 masked-header bytes。

如果本机安装 `tshark`，后续可以增加高性能解析路径。当前环境未检测到 `tshark` 或 `dumpcap`，第一版默认使用 Python 解析库，例如 Scapy。

## 4. Flow 重组

flow key 使用双向归并后的五元组：

```text
min(src_ip, dst_ip), max(src_ip, dst_ip),
min(src_port, dst_port), max(src_port, dst_port),
protocol
```

同时记录第一个 packet 的方向作为 canonical forward direction。后续 packet 根据是否与 canonical direction 一致标为：

```text
[PKT_FWD]
[PKT_BWD]
```

每个 packet 结束后追加：

```text
[PKT_END]
```

这样模型可以感知请求/响应方向和包边界。

## 5. 三种输入视图

每个 flow 同时生成三种字节视图。

### payload_only

只保留 TCP/UDP application payload。

适用目的：

- 严格验证 payload 内容本身是否足以检测攻击。
- 避免模型过度依赖 IP、端口等捷径。

### full_packet

保留完整 packet bytes。

适用目的：

- 验证包头和 payload 的完整组合是否能提升检测效果。
- 作为上限对比，但需要警惕数据集偏差。

### masked_header_packet

保留包头结构，但脱敏容易泄漏的字段。

默认脱敏字段：

- MAC 地址。
- IP 地址。
- TCP/UDP 端口。
- checksum。

保留字段：

- 协议结构。
- flags。
- 长度字段。
- 序列中 packet 的方向和边界。

适用目的：

- 判断模型是否真正利用协议结构，而不是记住地址、端口或数据集环境。

## 6. 标签对齐

不同数据集的标签来源不同。

### CIC 系列

CIC 系列通常提供 PCAP 和 flow/CSV 标签。对齐时使用：

- 时间戳。
- 源/目的 IP。
- 源/目的端口。
- 协议。
- 官方标签。

如果重建 flow 与官方标签无法可靠匹配：

- 不进入监督分类数据。
- 可以保留到无标签 MLM 预训练数据。

### USTC-TFC2016

USTC 数据可从目录名或文件名获得原始类别：

```text
Benign/<app>.pcap
Malware/<family>.pcap
```

映射为：

- benign app -> `major_label=benign`
- malware family -> `major_label=botnet_malware`，`minor_labels=[family]`

本地完整批处理使用仓库脚本顺序解析每个 PCAP，默认只生成 `payload_only` 视图，并限制常见数值库线程数和进程 CPU 亲和性，避免预处理时占满 CPU：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/preprocess_ustc_tfc2016.ps1 -Workspace .
```

输出约定：

- 单文件 parquet：`data/processed/ustc_tfc2016/files/*.parquet`
- 单文件 stdout/stderr：`data/processed/ustc_tfc2016/logs/*.log`
- 批处理进度：`data/processed/ustc_tfc2016/preprocess_progress.jsonl`

### Payload-Byte CSV

Payload-Byte 文件已经把每条样本整理成固定长度的 packet payload byte 列：

```text
payload_byte_1 ... payload_byte_1500, ttl, total_len, protocol, t_delta, label
```

这类数据不含完整五元组和双向 flow 边界，因此不能作为 flow 级主实验的严格替代。当前处理策略是：

- 每一行转换为一个单包 `payload_only` 样本。
- 去掉尾部零填充，保留 payload 内部的零字节。
- 默认丢弃全零 payload 行。
- `flow_id` 使用数据集名和原 CSV 行号生成，保证可复现。
- 标签仍通过统一 `configs/label_map.yaml` 映射到大类和细类。

对应命令：

```powershell
uv run traffic-bert data build-payload-csv --input-path data/raw/Payload-Byte/Payload_data_CICIDS2017.csv --output-path data/processed/payload_byte/cicids2017.parquet --source-dataset payload-byte-cicids2017
uv run traffic-bert data split --input-path data/processed/payload_byte/cicids2017.parquet --output-dir data/processed/payload_byte/cicids2017_split --group-column flow_id
uv run traffic-bert data merge --input-path data/processed/payload_byte/cicids2017_split/train.parquet --input-path data/processed/payload_byte/unsw_split/train.parquet --output-path data/processed/payload_byte/split/train.parquet
```

## 7. 标签统一

统一标签分两层：

```text
source_label -> major_label -> minor_labels
```

大类单选，细类 multi-hot。

输出字段：

```text
source_dataset
source_label
major_label
minor_labels
```

标签映射通过配置文件维护，不硬编码到解析逻辑中。

## 8. 空 payload 和短 flow

空 payload 或很短 payload 不直接全量删除。

处理策略：

- `full_packet` 和 `masked_header_packet` 视图保留。
- `payload_only` 视图允许通过配置决定是否保留。
- 记录 `has_payload`、`payload_byte_length`、`packet_count`。

这样避免丢掉扫描、DDoS 等可能没有明显应用层 payload 的样本。

## 9. 长度控制和滑窗

flow 级 token 序列可能很长。预处理或 Dataset 层进行滑窗：

```text
max_length = 512
stride = 384
```

一个 flow 可生成多个窗口：

```text
flow_001/window_000
flow_001/window_001
flow_001/window_002
```

窗口继承同一 flow 标签。模型训练时窗口级编码，最终聚合为 flow 级表示。

## 10. 数据划分

主实验原则上采用文件或时间划分：

```text
train: 若干 PCAP 文件或时间段
val:   不同 PCAP 文件或时间段
test:  独立 PCAP 文件或时间段
```

不将同一 PCAP 或同一时间段中的相邻 flow 随机打散到不同集合。

当前 USTC-TFC2016 有一个特殊情况：每个恶意家族基本只对应一个 PCAP 文件。若严格按 `source_file` 划分，很多恶意细类不会同时出现在 train/val/test，导致细分类无法正常训练和评估。因此当前保留两套划分：

- 主训练划分：`source_label` 分层 + `flow_id` 稳定划分，路径为 `data/processed/ustc_tfc2016/split_label_stratified/`。
- 泛化对照划分：严格 `source_file` 划分，路径为 `data/processed/ustc_tfc2016/split_source_file/`。

主实验训练优先使用 `split_label_stratified`，论文中需要说明这是为了保证细分类标签覆盖；`split_source_file` 用于额外评估未见 PCAP 文件的泛化能力。

## 11. 类别均衡

处理不均衡的方式：

- 统计每个大类和细类数量。
- 训练集可构建 class-balanced subset。
- 训练时支持 weighted sampler。
- loss 支持 class weights。

验证集和测试集默认保持原始分布，以反映真实评估难度。

## 12. 输出数据

处理后目录：

```text
data/processed/
├── manifest.json
├── label_map.yaml
├── train.parquet
├── val.parquet
├── test.parquet
├── stats.json
└── class_distribution.csv
```

Parquet schema 初版字段：

```text
flow_id: string
source_dataset: string
source_file: string
source_label: string
major_label: string
minor_labels: list<string>
split: string
view: string
packet_count: int
payload_byte_length: int
packet_byte_length: int
has_payload: bool
start_time: timestamp
end_time: timestamp
bytes: binary
```

后续如需同时存储三种视图，可以采用两种方式：

1. 每个 view 一行。
2. 一行包含 `payload_only_bytes`、`full_packet_bytes`、`masked_header_packet_bytes`。

第一版推荐每个 view 一行，方便训练配置按 view 过滤。

## 13. 预处理质量检查

每次构建数据后生成统计信息：

- flow 数量。
- packet 数量。
- 各数据集样本数量。
- 各大类和细类数量。
- payload 长度分布。
- packet 数量分布。
- 无标签 flow 数量。
- 空 payload flow 数量。
- train/val/test 分布。

这些统计既用于调试，也可作为论文数据集章节的表格来源。
