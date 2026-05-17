# 项目决策记录

本文档是当前最高层决策记录。若旧文档或历史实验记录与本文冲突，以本文、`experiment_plan.md` 和 `dataset_prepare.md` 为准。

## 1. 项目定位

项目定位为 **PCAP-first 的 SCE 恶意流量检测系统**。核心问题不是使用现成统计特征做传统分类，而是把原始网络流量转换成模型可学习的序列表示，并完成恶意流量检测、分类和可展示推理。

当前阶段不再扩展新数据集，先把 CICIDS2017 baseline、审计链路和 PCAP 上传推理 demo 做成可交付闭环。

## 2. 三阶段技术路线

方案1是工程 baseline：

- 从 PCAP 重组 flow。
- 将 payload、packet header 或 masked header packet 转成 byte token。
- 使用 Byte-BERT 完成 MLM、监督分类和 PCAP 推理。
- 目标是证明训练、评估、审计和展示链路可信。

方案2是论文核心创新：

- SCE（Semantic Conversion Encoder）把 bytes、header fields、packet direction 和 flow pattern 转换成更稳定的语义 token 或 codebook。
- SCE 不是现成论文名词，是本项目对“网络流量语义转换层”的定义。
- SCE 的目标是降低原始 byte 噪声、减少数据集捷径、提升变种攻击和跨数据集泛化。

方案3是泛化增强：

- 在 SCE 或 Byte-BERT 表示上做自监督预训练。
- 后续接入 IoT-23、CTU-13 等泛化验证数据。
- 不在当前阶段阻塞 CICIDS2017 baseline 和 demo 交付。

## 3. 分类目标

分类采用 Level-0/1/2 口径：

- Level-0：良性 / 恶性。
- Level-1：攻击家族或大类，例如 dos_ddos、scan、web_attack、botnet_malware。
- Level-2：攻击子类或原始标签细分，例如 DDoS、PortScan、Bot、XSS。

当前实现中，Level-0 可由 benign vs non-benign 派生，Level-1 对应 `major_label`，Level-2 对应 `minor_labels` 或 `source_label` 报告。文档和论文中需要明确这个映射，避免把所有层级混成一个平面多分类问题。

## 4. 数据集策略

当前阶段：

- 主数据集：CICIDS2017 全量 Monday-Friday working-hour PCAP + 官方 labelled-flow CSV。
- 诊断数据：USTC-TFC2016，只用于 sanity、source-file 诊断或辅助说明，不作为正式主结论。
- 辅助数据：Payload-Byte 可作为快速 sanity 或自监督候选，不直接混入当前监督主训练。

后续阶段：

- IoT-23：优先作为泛化验证集。
- CTU-13：作为 botnet 外部泛化补充。
- 其他 CIC 系列数据只进入 roadmap，不阻塞当前交付。

## 5. CICIDS2017 当前正式口径

CICIDS2017 baseline 必须使用：

- 全量 Monday-Friday PCAP，不允许 Friday/smoke/旧 parquet 冒充正式数据。
- `masked_header_packet` 作为当前正式输入视图，保留空 payload flow。
- `flow_timeout_seconds=120`，避免同五元组整天合并。
- full-PCAP scan + 官方 CSV 近邻标签匹配，不用攻击时间窗口裁剪训练数据。
- coverage gate、split audit 和 formal dataset gate。

当前可训练主 split 是 `all_masked_header_split_submode_stratified_group_cap32_notcpclose`。它用于保证关键攻击子形态在训练集中有覆盖，并通过 bytes/window/nearby five-tuple 等泄漏检查。

`time_block` 只作为时间外推压力测试。它的失败应解释为时间或子形态外推困难，不应反过来把当前可训练 baseline 否定掉。

## 6. 模型与展示系统

Byte-BERT 是方案1 baseline encoder，不是最终 SCE 本体。它负责提供可运行、可审计的基线。

展示系统第一版只承诺 PCAP/PCAPNG 上传：

```text
上传 PCAP
  -> 解析 flow
  -> masked-header packet token 化
  -> 加载 checkpoint 推理
  -> flow 级预测
  -> 整体风险汇总
```

现有 `traffic-bert predict pcap` 是后端基础能力。Web/UI 可以在该 CLI 能力稳定后封装，不需要先实现 IoT-23/CTU-13。

## 7. 评估与通过标准

正式报告必须包含：

- Accuracy。
- Macro Precision/Recall/F1。
- Weighted-F1。
- Per-class precision/recall/F1。
- Confusion matrix。
- Level-0 attack detection TPR/FPR/FNR/F1。
- Split audit、attack coverage 和 formal gate 文件。

Accuracy 和 weighted-F1 不能单独作为通过标准。Bot、web_attack、scan、dos_ddos 等 support 足够的关键类必须检查 recall 和 F1。Bot recall/F1 不达标时，run 必须标记为 failed 或 needs review。

Heartbleed、Infiltration 等极低样本类保留在数据报告中，但当前作为 low-support reported-only，不作为可训练主类别验收。

## 8. 当前优先级

1. 收束文档和目标口径。
2. 完成 CICIDS2017 baseline 数据 gate、训练 gate 和训练后验收 gate。
3. 保留 Bot 复盘结论：正式 flow 不按 FIN/RST 再拆分，TCP close/reset 信息留在 packet 和 connection_type 中。
4. 封装 PCAP 上传推理 demo。
5. 在 baseline 稳定后继续改进 SCE；当前已完成频率 codebook 骨架和样例构建，但还不能写成有效提升。
6. 最后再做 IoT-23/CTU-13 泛化验证。
