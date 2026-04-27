# 项目决策记录

本文档记录当前已经讨论并确定的技术路线，后续实现和论文写作应优先遵循这里的决策。

## 1. 项目定位

项目目标是构建一套基于原始网络流量字节序列的恶意流量检测模型。输入可以来自 PCAP 中的 payload，也可以来自完整 packet bytes 或十六进制 payload。模型输出恶意流量的大类和可能的细分类结果。

该项目不是单纯使用 CICFlowMeter 统计特征做传统机器学习，而是尽量从原始字节序列中学习模式，突出深度学习和 NLP 表示学习路线。

## 2. 数据集策略

采用“先小后大”的数据集策略：

- 第一阶段优先接入 USTC-TFC2016 和 CICIDS2017 相关数据。
- CICIDS2017 以官方 PCAP 和标签为主，Payload-Byte 可作为快速 sanity check 或备用数据源。
- 后续扩展 CSE-CIC-IDS2018、CIC-DDoS2019 等数据集。

数据主线以原始 PCAP 为主，因为题目关注原始 payload/packet bytes。预处理后的公开数据可以辅助验证，但不作为唯一来源。

## 3. 样本粒度

模型主实验采用 flow 级样本，而不是单包样本。

原因：

- 攻击行为往往体现在多个包的交互过程里。
- flow 级样本更适合表达请求/响应、方向、包边界和长时序特征。
- 后续论文中可以更自然地说明流量重组和序列建模的必要性。

flow 由五元组归并，并按时间顺序追加每个 packet 的字节序列。

## 4. 字节到文本 token 的映射

题目希望体现“映射到自然语言/NLP 输入”的思想。当前确定的表达方式是：不把 payload 翻译成人类语言，而是将字节流转换为 NLP 风格的离散文本 token 序列。

映射方式：

```text
0x00 -> b_00
0x01 -> b_01
...
0xff -> b_ff
```

并加入特殊 token：

```text
[PAD] [UNK] [CLS] [SEP] [MASK] [PKT_FWD] [PKT_BWD] [PKT_END]
```

这样可以同时满足：

- BERT 输入需要离散 token 的要求。
- 原始字节信息可逆，不丢失。
- 避免强行映射到中文或英文带来的伪语义。
- 论文中可以将其描述为 payload 的文本化表示和字节语言建模。

## 5. 输入视图

为了验证是否应该保留包头信息，预处理阶段同时构建三种视图：

1. `payload_only`
   - 只保留 TCP/UDP application payload。
   - 最符合“payload 检测”的定义。

2. `full_packet`
   - 保留完整 packet bytes，包括链路层、网络层、传输层头部和 payload。
   - 可能获得更高指标，但也可能学习到 IP、端口等数据集捷径。

3. `masked_header_packet`
   - 保留包头结构，但脱敏 MAC、IP、端口、checksum 等容易泄漏数据集身份或攻击脚本特征的字段。
   - 用于判断包头结构本身是否对检测有效。

论文中通过三种视图的消融实验说明最终输入选择的合理性。

## 6. 长 flow 处理

BERT 输入长度有限，flow 可能远超 512 tokens。当前决策：

- 使用滑窗切分长 flow。
- 默认窗口长度为 512 tokens。
- 默认 stride 为 384 tokens。
- 每个窗口通过 BERT 编码。
- 窗口表示再聚合为 flow 级表示。

聚合方式第一版优先实现 mean pooling 或 attention pooling。

## 7. 模型路线

模型采用 Byte-BERT：

- 使用 BERT 架构，但词表是固定字节 token 词表。
- 先在无标签或有标签 flow 的字节文本序列上做 MLM 预训练。
- 再在统一标签后的监督数据上做分类微调。
- 初始模型配置保持较小，以适配本机 RTX 3050 Laptop 4GB 显存。

默认模型规模：

```text
hidden_size = 256
num_hidden_layers = 4
num_attention_heads = 4
max_position_embeddings = 512
```

## 8. 分类策略

分类采用层级多标签方案：

1. 大类分类
   - 大类是单选。
   - 使用 softmax 和 cross entropy。

2. 大类内子类分类
   - 子类允许多个同时激活。
   - 使用 sigmoid 和 BCE loss。
   - 只在预测或真实大类对应的子类集合内判断阈值。

推理输出规则：

- 先输出 `major_label`。
- 如果大类下没有子类超过阈值，则只输出大类。
- 如果一个或多个子类超过阈值：
  - 概率最高的子类作为 `primary_minor_label`。
  - 其他过阈值子类进入 `activated_minor_labels`。

这样既保留多标签能力，又能给用户一个清晰的主结论。

## 9. 标签体系

统一标签体系以攻击行为为主，而不是完全保留各数据集原始标签。

初始大类建议：

```text
benign
dos_ddos
bruteforce
web_attack
botnet_malware
scan
infiltration
heartbleed
other_attack
```

原始标签映射示例：

```yaml
"DDoS":
  major_label: dos_ddos
  minor_labels: ["ddos"]

"DoS Hulk":
  major_label: dos_ddos
  minor_labels: ["dos_hulk"]

"Web Attack - SQL Injection":
  major_label: web_attack
  minor_labels: ["sql_injection"]

"Neris":
  major_label: botnet_malware
  minor_labels: ["neris"]
```

## 10. 数据划分

训练集、验证集、测试集优先按文件或时间划分，而不是随机按 flow 划分。

原因：

- 随机 flow 划分容易让同一攻击会话的相邻 flow 同时进入训练和测试。
- 文件/时间划分更严格，指标更可信。
- 论文中更容易说明没有明显数据泄漏。

如果后续需要，也可以额外报告随机划分结果作为上限参考，但主结果应使用文件/时间划分。

## 11. 无标签和空 payload 处理

无标签 flow：

- 不进入监督分类训练。
- 可以进入 MLM 预训练。

空 payload 或极短 payload：

- 不直接全部删除。
- 在 `full_packet` 和 `masked_header_packet` 视图中保留。
- 在 `payload_only` 视图中可配置过滤或保留。

这样可以避免丢失扫描、DDoS 等可能主要体现在包头或交互行为中的样本。

## 12. 类别不均衡

第一版采用采样和加权 loss 结合：

- 构建 class-balanced train subset 或 weighted sampler。
- 对大类和子类 loss 引入 class weights。
- 指标报告重点看 Macro-F1 和 per-class 结果。

## 13. 评估指标

主指标使用 Macro-F1，同时报告更多维度：

- Accuracy
- Macro Precision
- Macro Recall
- Macro-F1
- Weighted-F1
- per-class Precision/Recall/F1
- Confusion matrix
- 攻击检测视角下的 TPR、FPR、FNR
- 子类阈值校准曲线或 PR 曲线

这样既能反映整体效果，也能避免类别不均衡导致 accuracy 虚高。

## 14. Baseline 和消融

论文实验应包含完整对比：

- 传统方法：n-gram/TF-IDF + Logistic Regression 或 Linear SVM。
- 深度模型：1D-CNN、BiLSTM/GRU、普通 Transformer Encoder。
- 主模型：Byte-BERT。

消融实验：

- `payload_only` vs `full_packet` vs `masked_header_packet`。
- 有 MLM 预训练 vs 无 MLM 预训练。
- 不同 max length 或滑窗策略。
- 不同 pooling 策略。

## 15. 推理入口

第一版以 CLI 为主。

支持输入：

- PCAP 文件。
- 十六进制 payload 字符串。
- bytes 文件。

输出 JSON，包含：

```json
{
  "major_label": "dos_ddos",
  "major_prob": 0.93,
  "primary_minor_label": "ddos",
  "primary_minor_prob": 0.81,
  "activated_minor_labels": [
    {"label": "ddos", "prob": 0.81}
  ]
}
```

Web 演示不是第一阶段重点。

## 16. Git 和工程管理

- 当前工作区需要初始化为本地 Git 仓库。
- 后续每个里程碑都做本地 commit。
- 大型数据、模型权重和实验产物不提交。
- GitHub 上传时只包含代码、配置、文档和小型 fixtures。

