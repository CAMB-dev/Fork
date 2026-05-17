# 模型设计

本文档描述当前模型路线。Byte-BERT 是方案1 baseline；SCE 是后续方案2 的核心创新模块。

## 1. 当前 baseline 结构

```text
flow bytes / masked packet bytes
  -> byte tokenizer
  -> sliding windows
  -> Byte-BERT encoder
  -> window pooling
  -> flow representation
  -> hierarchical classifier
```

该 baseline 的作用是建立可信工程链路：PCAP 解析、flow 重组、token 化、训练、评估、推理和展示系统后端。

## 2. Byte tokenizer

方案1 使用固定 byte token：

```text
0x00 -> b_00
...
0xff -> b_ff
```

特殊 token：

```text
[PAD] [UNK] [CLS] [SEP] [MASK]
[PKT_FWD] [PKT_BWD] [PKT_END]
```

该 tokenizer 可逆、稳定、便于快速训练，但它仍然直接暴露原始 byte 模式。它不是最终 SCE，只是 baseline 和 SCE 的输入基座。

## 3. SCE 定义

SCE 全称 Semantic Conversion Encoder，即语义转换编码器。它是本项目定义的核心模块，不是现成库或固定论文名词。

SCE 的目标是将以下原始信号转换成更稳定、更可学习的语义 token：

- payload bytes。
- header fields。
- packet direction。
- packet length pattern。
- flow-level interaction pattern。

理想效果是让模型少记偶然 byte、IP/端口或数据集特征，多学习类似 `HTTP_POST_FORM`、`CREDENTIAL_SUBMISSION`、`SHORT_RESET_SEQUENCE` 这样的行为模式。SCE 后续可以用离散 codebook、聚类、可学习 tokenizer 或规则引导的语义片段实现。

## 4. SCE v0：频率 codebook 骨架

当前已加入一个最小 SCE/codebook 骨架：

```bash
traffic-bert sce build-codebook \
  --input-path data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/train.parquet \
  --output-path artifacts/sce/cicids_v0_frequency_codebook_sample.json \
  --view masked_header_packet \
  --chunk-size 4 \
  --max-entries 64 \
  --min-count 5 \
  --max-rows 20000
```

实现范围：

- `src/traffic_bert/sce.py` 提供 `SemanticCodebook`、`learn_frequency_codebook` 和 label-lift codebook。
- codebook 将 byte chunk 转成 `[SCE_0000]` 形式的离散 token。
- `traffic-bert sce build-codebook` 支持 `frequency` 和 `label_lift` ranking；label-lift 可用 `--max-rows-per-label` 做确定性 reservoir 抽样，避免按 parquet head 抽样误导 Bot token。
- unmatched chunk 会 fallback 为原始 byte token，避免丢失信息。
- 默认跳过全零 chunk，避免 masked-header 的零填充主导 codebook。
- `ByteTokenizer(extra_tokens=codebook.tokens)` 可承载 SCE token，但默认 tokenizer 不改变。
- `FlowWindowDataset(..., semantic_codebook=...)` 可在样本编码时使用 SCE token。
- `traffic-bert train/eval classifier`、`eval calibrate-thresholds`、`predict hex/pcap` 已支持 `--semantic-codebook-path`。
- checkpoint 会保存 codebook 元数据，评估和预测可从 checkpoint 自动恢复 codebook。

当前限制：

- v0 只是确定性频率 codebook，不是最终语义聚类或可学习 tokenizer。
- v14 对照显示朴素频率 codebook 会让 Bot false positive 变多，不能作为最终 SCE 方案。
- v15 label-lift Bot codebook 也未超过当前 Byte-BERT baseline，说明简单监督 byte chunk 仍不足以稳定提升 Bot 这类短连接形态。
- 在 masked-header 视图中仍会学到部分 header/protocol 片段，后续需要进一步过滤低信息 token 或引入更有语义的 chunk 特征。

## 5. SCE 分阶段原则

当前不把已有 byte tokenizer 硬包装成 SCE。文档和论文中应明确：

- 方案1：Byte-BERT baseline，证明 PCAP-first 链路成立。
- 方案2：SCE/codebook，作为核心创新实现。
- 方案3：SCE/Byte-BERT 自监督预训练和跨数据集泛化。

这样可以避免在 baseline 尚未稳定时过早引入复杂模块，也避免论文创新点被写成简单 byte token 化。

## 6. Byte-BERT Encoder

当前 baseline 使用小型 BERT 配置：

```text
base vocab_size = 318
SCE vocab_size = 318 + codebook entries
max_position_embeddings = 512
hidden_size = 256
num_hidden_layers = 4
num_attention_heads = 4
intermediate_size = 1024
```

当前训练以云端 CUDA 为主，本地只做轻量验证。模型规模后续可扩大，但必须先保证数据 gate、split audit 和 per-class 结果可信。

## 7. 长 flow 处理

单个 flow 可能超过 512 tokens，因此使用滑窗：

```text
window_0 -> encoder -> h_0
window_1 -> encoder -> h_1
window_2 -> encoder -> h_2
```

当前 baseline 聚合方式为稳定的窗口 pooling。后续可以比较 mean、max、attention pooling，但不能用更复杂 pooling 掩盖数据泄漏或类别覆盖问题。

## 8. 层级分类头

分类口径对应 Level-0/1/2：

- Level-0：benign vs attack，可由 `major_label != benign` 派生。
- Level-1：`major_label` softmax 单选。
- Level-2：`minor_labels` 或 `source_label` 多标签/细类报告。

训练主 loss 当前以 major classification 为核心，minor 输出用于细类诊断和阈值校准。论文报告需要区分 Level-0 detection、Level-1 family classification 和 Level-2 subtype analysis。

## 9. MLM 预训练

MLM 是方案3 的增强方向：

- 对 byte/SCE token 做 mask prediction。
- 可使用有标签 flow、未进入监督训练的可解析 flow 或辅助数据。
- 当前不把 Payload-Byte/USTC 混入正式监督训练，但可以作为后续自监督候选。

## 10. 推理与展示系统

第一版展示系统只要求 PCAP/PCAPNG 上传：

```text
PCAP upload
  -> PcapFlowExtractor
  -> masked-header packet view
  -> Byte-BERT checkpoint
  -> flow predictions
  -> file-level risk summary
```

已有 `traffic-bert predict pcap` 可作为后端原型。Web UI 不改变模型输入语义，只是封装上传、解析、推理和结果展示。

## 11. 后续模型对比

论文对比不应只比较 accuracy。可选 baseline：

- byte n-gram/TF-IDF + Logistic Regression。
- 1D-CNN。
- GRU/BiLSTM。
- Transformer Encoder。
- Byte-BERT baseline。
- SCE + Byte-BERT。

所有对比必须使用同一 processed schema、同一 split、同一 audit 口径。
