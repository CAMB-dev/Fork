# 实验计划

本文档用于规划毕设实验、消融实验和论文图表。

## 1. 实验目标

实验需要回答以下问题：

1. 基于原始字节序列的 Byte-BERT 是否能完成恶意流量分类。
2. flow 级建模是否能有效利用 payload 和包序列信息。
3. 包头信息到底是有效特征，还是数据集偏差。
4. MLM 预训练是否提升分类性能。
5. Byte-BERT 相比传统方法、CNN/RNN/Transformer 是否有优势。

## 2. 数据集组合

第一阶段：

- USTC-TFC2016。
- CICIDS2017。

后续扩展：

- CSE-CIC-IDS2018。
- CIC-DDoS2019。

每个数据集需要记录：

- 原始样本数量。
- 成功解析 flow 数量。
- 有标签 flow 数量。
- 无标签 flow 数量。
- 各大类/细类数量。
- train/val/test 划分比例。

## 3. 主实验

主实验使用：

- 输入视图：根据消融结果选择，预计优先比较 `payload_only` 和 `masked_header_packet`。
- 模型：Byte-BERT。
- 训练流程：MLM 预训练 + 分类微调。
- 数据划分：文件或时间划分。
- 主指标：Macro-F1。

报告指标：

- Accuracy。
- Macro Precision。
- Macro Recall。
- Macro-F1。
- Weighted-F1。
- per-class Precision/Recall/F1。
- Confusion matrix。
- 恶意检测 TPR、FPR、FNR。

## 4. 消融实验

### 输入视图消融

比较：

```text
payload_only
full_packet
masked_header_packet
```

目的：

- 判断包头是否有帮助。
- 判断完整包指标是否来自 IP/端口等捷径。
- 为论文中最终输入选择提供证据。

### MLM 预训练消融

比较：

```text
Byte-BERT without MLM
Byte-BERT with MLM
```

目的：

- 验证自监督预训练对 payload 字节建模的帮助。

### 长度和滑窗消融

比较：

```text
max_len = 256
max_len = 512
不同 stride
截断 vs 滑窗
```

目的：

- 证明长 flow 不应简单截断。
- 找到显存和效果之间的平衡点。

### 聚合方式消融

比较：

```text
mean pooling
attention pooling
max pooling
```

目的：

- 判断多个窗口如何聚合最合适。

## 5. Baseline 对比

计划对比模型：

1. n-gram/TF-IDF + Logistic Regression。
2. n-gram/TF-IDF + Linear SVM。
3. 1D-CNN。
4. BiLSTM 或 GRU。
5. Transformer Encoder。
6. Byte-BERT without MLM。
7. Byte-BERT with MLM。

所有模型应尽量使用相同的 train/val/test 划分和标签体系。

## 6. 分类输出评估

大类评估：

- 按单标签多分类评估。
- 使用 confusion matrix 和 per-class 指标。

子类评估：

- 按多标签评估。
- 报告 micro/macro F1。
- 报告每个子类阈值。
- 分析无子类过阈值时只输出大类的比例。

输出逻辑评估：

- 大类正确但子类未命中。
- 大类正确且主子类正确。
- 大类错误导致子类不评估。

## 7. 论文图表建议

建议生成以下图表：

- 数据集处理流程图。
- Byte-BERT 模型结构图。
- 层级分类输出示意图。
- 各数据集类别分布表。
- 三种输入视图消融结果表。
- baseline 对比表。
- confusion matrix。
- per-class F1 柱状图。
- 子类阈值校准曲线。
- flow 长度分布图。

## 8. 成功标准

第一阶段工程成功标准：

- 能从 PCAP 构建 flow 级 processed 数据。
- 能完成字节 token 化和滑窗。
- 能跑通 Byte-BERT MLM 预训练。
- 能跑通分类微调。
- 能输出大类和子类阈值结果。
- 能在小样本上完成端到端 CLI 推理。

论文成功标准：

- 数据处理流程可解释。
- 模型设计和题目要求一致。
- 实验对比和消融足够支撑结论。
- 指标不只依赖 accuracy，能体现类别不均衡下的真实性能。

