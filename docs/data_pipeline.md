# 数据管线

本文档描述 PCAP-first 数据管线、schema 和审计规则。当前阶段以 CICIDS2017 baseline 为主，SCE 和跨数据集泛化后续接入。

## 1. 总体流程

```text
raw PCAP / official labels
  -> flow extraction
  -> label alignment
  -> input views
  -> processed parquet
  -> split generation
  -> validation / audit / gate
  -> training / evaluation / PCAP prediction
```

数据管线的目标是让训练、评估和展示系统都复用同一套 flow extraction 与 tokenization 逻辑。

## 2. Flow 粒度

样本单位是 flow/session，而不是单包。

当前 PCAP 重组规则：

- 使用有向首包方向记录 initiator/responder。
- 使用双向五元组聚合同一 session。
- `flow_timeout_seconds=120`，同五元组超过 idle timeout 后切新 session。
- `flow_id` 包含 session 信息，避免一天内同五元组合并成一个样本。
- 单个 flow 可限制 `max_packets_per_flow`，但原始 packet count 和截断标记必须保留。

这样可以保留 PortScan、DDoS、Bot 短连接、reset/refused 等可能主要体现在包头和交互形态中的行为。

## 3. 输入视图

当前支持：

- `payload_only`：只保留 L4 payload，适合消融和部分诊断。
- `full_packet`：保留完整包字节，容易学习 IP/端口等捷径，不能直接作为主结论。
- `masked_header_packet`：保留包头结构但脱敏 MAC/IP/端口/checksum 等字段，是当前 CICIDS baseline 主输入。

SCE 后续会在这些基础信号上生成语义 token/codebook，但不会改变 PCAP-first 的数据入口。

## 4. 标签对齐

CICIDS2017 官方 labelled flows 依赖 timestamp、源/目的 IP、源/目的端口、协议和攻击标签。正式构建必须：

- 全量扫描 PCAP，不用攻击窗口裁剪训练样本。
- 使用 CSV timestamp offset `+3h`。
- 优先按有向五元组匹配，必要时记录 fallback。
- 使用 flow start time 在重复五元组候选中选最近 label。
- 写出 `label_match_mode` 和 `label_time_delta_seconds`。

无标签 flow 不进入监督训练，可作为后续自监督候选。

## 5. Processed Schema

核心字段：

```text
flow_id: string
source_dataset: string
source_file: string
source_label: string
major_label: string
minor_labels: list<string>
split: string
view: string
protocol: string
connection_type: string
endpoint_a: string
endpoint_b: string
initiator_endpoint: string
responder_endpoint: string
label_match_mode: string
label_time_delta_seconds: float
packet_count: int
observed_packet_count: int
was_packet_truncated: bool
payload_byte_length: int
packet_byte_length: int
has_payload: bool
start_time: float
end_time: float
packet_directions: list<string>
packet_lengths: list<int>
bytes: binary
```

`connection_type` 是审计和可选输入字段，例如 `tcp_payload`、`tcp_handshake_only`、`tcp_reset_or_refused`、`tcp_control_only`、`udp_payload`。当前最强 flow-level baseline 使用 connection token；它能改善 Bot 这类短 TCP reset/control flow 的可解释性和可分性。

## 6. Split 口径

当前 CICIDS baseline 主 split：

```text
all_masked_header_split_submode_stratified_group_cap32_notcpclose
```

设计目标：

- 保持正式 group，不拆 bytes/window 重复、nearby five-tuple 等高泄漏风险样本。
- 按 source label 和可见 packet submode 分层，避免验证集中出现训练集中完全没有的关键攻击形态。
- 仍然通过 source file overlap 记录数据来源风险，但不把 CICIDS 全量 split 简化成单一文件级评估。

压力测试 split：

```text
all_masked_header_split_time_block
```

time-block 用于评估时间和子形态外推。它可能导致某些攻击形态在训练集中不存在，因此不作为当前可训练主线。

Random-flow 或 label-stratified split 只能作为 upper-bound/sanity。

## 7. Audit 与 Gate

Split audit 必须检查：

- source file overlap。
- major/source label 覆盖。
- low-support label。
- flow_id leakage。
- nearby five-tuple leakage。
- raw bytes hash overlap。
- first-window token hash overlap。
- label submode train coverage。

Formal dataset gate 必须检查：

- 五个 CICIDS PCAP 存在且非空。
- processed shard build metadata 新鲜。
- attack coverage 合格。
- split audit 合格。
- 不允许 smoke、Friday-only、payload-only 旧产物作为正式输入。

## 8. 展示系统复用

PCAP 上传展示系统应复用同一数据管线：

```text
uploaded.pcap
  -> PcapFlowExtractor
  -> masked_header_packet flow records
  -> checkpoint inference
  -> flow-level predictions
  -> file-level risk summary
```

展示系统不需要官方 label，也不需要 split；它只需要和训练时一致的 flow extraction、view 构建和 tokenizer。

## 9. USTC 与 Payload-Byte

USTC-TFC2016：

- 可用于 sanity 和大类诊断。
- 每个恶意家族基本单 PCAP，严格 source-file split 下 minor/family 是 unseen-family 诊断。
- `split_label_stratified` 会产生同源风险，只能作为 upper-bound。

Payload-Byte：

- 是单包 payload 行级数据。
- 不含完整五元组和双向 flow 边界。
- 不直接混入 CICIDS 当前监督主训练。

## 10. SCE 接入点

SCE 最小版应接在 view 构建之后、Byte-BERT tokenization 之前：

```text
packet bytes / header fields / flow pattern
  -> SCE semantic tokens
  -> encoder
```

SCE 输出必须仍能写入 processed parquet 或一个可复用的派生字段，确保训练、评估、demo 使用同一输入语义。
