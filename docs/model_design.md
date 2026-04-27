# 模型设计

本项目模型主线是 Byte-BERT：使用 BERT 架构，但输入不是自然语言句子，而是由网络流量字节序列文本化得到的 token 序列。

## 1. 总体结构

```text
flow bytes
  -> byte textual tokenizer
  -> sliding windows
  -> Byte-BERT encoder
  -> window pooling
  -> flow representation
  -> hierarchical classifier
```

## 2. 字节文本化 tokenizer

每个 byte 映射为固定 token：

```text
0x48 0x45 0x4c 0x4c 0x4f
-> b_48 b_45 b_4c b_4c b_4f
```

特殊 token：

```text
[PAD] [UNK] [CLS] [SEP] [MASK]
[PKT_FWD] [PKT_BWD] [PKT_END]
```

词表规模约为：

```text
256 byte tokens + 8 special tokens = 264
```

这个 tokenizer 不依赖中文或英文预训练词表，因为 payload 的语义单位不是自然语言词，而是协议字段、命令片段、二进制结构和攻击 payload 模式。

## 3. Byte-BERT Encoder

第一版采用小型 BERT 配置：

```text
vocab_size = 264
max_position_embeddings = 512
hidden_size = 256
num_hidden_layers = 4
num_attention_heads = 4
intermediate_size = 1024
```

选择小模型的原因：

- 本机 GPU 为 RTX 3050 Laptop，显存约 4GB。
- flow 级输入需要滑窗，batch 内可能包含多个窗口。
- 第一阶段优先保证训练可跑通，再逐步扩大模型。

## 4. MLM 预训练

预训练任务使用 masked language modeling。

输入 token 随机 mask：

- 15% token 参与 MLM。
- 其中 80% 替换为 `[MASK]`。
- 10% 替换为随机 byte token。
- 10% 保持不变。

MLM 目标是预测原始 byte token。

预训练数据：

- 有标签 flow。
- 无标签但可解析的 flow。
- 不进入分类训练的 unmatched flow。

这样可以利用更多原始流量学习字节序列模式。

## 5. 长 flow 滑窗与聚合

单个 flow 可能切成多个窗口：

```text
window_0 -> BERT -> h_0
window_1 -> BERT -> h_1
window_2 -> BERT -> h_2
```

窗口表示初版取 `[CLS]` hidden state。

flow 聚合候选：

- mean pooling：简单稳定，作为默认实现。
- attention pooling：学习每个窗口的重要性，作为增强实现。

第一版可先实现 mean pooling，保留 attention pooling 配置位。

## 6. 层级分类头

分类分为两级。

### 大类分类

大类是单选：

```text
major_logits = Linear(flow_repr, num_major_classes)
major_probs = softmax(major_logits)
```

loss：

```text
CrossEntropyLoss
```

### 子类分类

子类是大类内部多标签：

```text
minor_logits = Linear(flow_repr, num_minor_classes)
minor_probs = sigmoid(minor_logits)
```

训练时使用 multi-hot 标签。

loss：

```text
BCEWithLogitsLoss
```

可以通过 mask 让每条样本只对其大类下的子类计算 loss。

## 7. 总损失

分类微调总损失：

```text
loss = major_loss + lambda_minor * minor_loss
```

默认：

```text
lambda_minor = 1.0
```

如果子类标签噪声较大，后续可降低 `lambda_minor`。

## 8. 推理逻辑

推理步骤：

1. 对输入 PCAP/hex/bytes 构建 flow。
2. 转换为 byte token 序列。
3. 滑窗输入 Byte-BERT。
4. 聚合窗口表示。
5. 预测大类。
6. 只在预测大类对应的子类集合内判断阈值。

输出规则：

- `major_label` 始终输出。
- 没有子类过阈值时，只输出大类。
- 多个子类过阈值时：
  - 最高概率子类为 `primary_minor_label`。
  - 所有过阈值子类为 `activated_minor_labels`。

示例：

```json
{
  "major_label": "web_attack",
  "major_prob": 0.91,
  "primary_minor_label": "sql_injection",
  "primary_minor_prob": 0.77,
  "activated_minor_labels": [
    {"label": "sql_injection", "prob": 0.77},
    {"label": "xss", "prob": 0.61}
  ]
}
```

如果无子类过阈值：

```json
{
  "major_label": "web_attack",
  "major_prob": 0.88,
  "primary_minor_label": null,
  "primary_minor_prob": null,
  "activated_minor_labels": []
}
```

## 9. 阈值校准

子类阈值默认从 0.5 开始。

训练完成后，在验证集上为每个子类搜索阈值：

- 优化 F1。
- 或在指定 FPR 约束下最大化 recall。

最终阈值保存到配置或 checkpoint metadata 中。

## 10. Baseline

为了支撑论文对比，计划实现：

- byte n-gram/TF-IDF + Logistic Regression。
- byte n-gram/TF-IDF + Linear SVM。
- 1D-CNN。
- BiLSTM/GRU。
- Transformer Encoder。
- Byte-BERT。

主模型和 baseline 都读取同一套 `data/processed/` 数据，保证对比公平。

