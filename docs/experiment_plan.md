# 实验计划

本文档规划当前阶段实验、验收标准和后续扩展。当前阶段目标是 CICIDS2017 baseline + PCAP 推理 demo，不处理 IoT-23/CTU-13。

## 1. 当前阶段问题

当前阶段需要回答：

1. PCAP-first 数据链路能否稳定构建 flow 级训练样本。
2. Byte-BERT baseline 能否在审计合格的 CICIDS2017 split 上完成恶意流量分类。
3. 结果是否经得起 macro/per-class 指标检查，而不是只依赖 accuracy。
4. PCAP 上传推理 demo 是否能复用同一 flow extraction 和模型推理链路。

SCE 是下一阶段核心创新，不在当前阶段强行实现。

## 2. 数据集组合

当前阶段：

- CICIDS2017 原始 PCAP + CSV：正式 baseline 数据集。
- USTC-TFC2016：辅助诊断和 sanity，不作为严格正式主结论。
- Payload-Byte：辅助 sanity 或自监督候选，不进入当前监督主训练。

后续阶段：

- IoT-23：跨场景泛化验证。
- CTU-13：botnet 泛化补充。

当前训练路线：

```text
方案1 baseline:
  CICIDS2017 all_masked_header_split_submode_stratified_group_cap32_notcpclose

压力测试:
  CICIDS2017 all_masked_header_split_time_block

辅助诊断:
  USTC split_source_file_major_balanced

sanity / upper-bound:
  USTC split_label_stratified
```

## 3. CICIDS2017 baseline 实验

正式 baseline 使用：

- 全量 Monday-Friday PCAP。
- `masked_header_packet` 输入视图。
- 保留空 payload flow。
- `flow_timeout_seconds=120`。
- coverage gate + split audit + formal dataset gate。
- `all_masked_header_split_submode_stratified_group_cap32_notcpclose` 作为可训练主 split。

`time_block` 只作为压力测试，报告它是否暴露时间外推或子形态外推失败。

## 4. 训练验收标准

正式 run 必须报告：

- Accuracy。
- Macro Precision/Recall/F1。
- Weighted-F1。
- Per-class precision/recall/F1。
- Confusion matrix。
- Level-0 attack detection precision/recall/F1、FPR、FNR。
- Split audit 和 formal gate。

通过判断不能只看 accuracy 或 weighted-F1。最低要求：

- `formal_eligible=true`。
- `blocking_warnings=[]`。
- support 足够的关键类必须有可接受的 recall/F1。
- Bot 等关键类明显漏判时，run 标记为 failed 或 needs review。
- Heartbleed/Infiltration 等极低样本类标记为 low-support reported-only，不作为当前可训练主类别验收。

当前已知结果状态：

- 当前可信数据版本是 cap32 + no-close 的 full-PCAP build：
  `/root/Fork/data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose`。
- 当前最强 flow-level baseline 是 v16：
  `/root/Fork/artifacts/cicids2017_masked_header_submode_group_formal_cuda_v16_cap32_notcpclose_conn_b128_w4_sqrt_weighted`。
- v16 使用 `connection tokens + sqrt_balanced`，test Bot precision/recall/F1 约 `0.615/0.995/0.760`，达到当前毕设可交付口径。
- v11/v12/v13 分别尝试 context tokens、balanced loss、numeric context side-channel，均因 Bot 误报过多而停止。
- Bot 阈值扫描没有找到满足 `recall >= 0.8` 且 `F1 >= 0.75` 的阈值。
- `scripts/analyze_bot_host_windows.py` 已用 v10/v16 预测做 host-window 诊断。
- v16 flow-level Bot 已通过；host-window 风险层只作为展示/告警辅助，不覆盖 flow-level 指标。
- `formal_run_summary.json` 中 `acceptance.formal_eligible=true`；Infiltration 仍是 low-support reported-only，BruteForce 截断比例需要在报告中说明。
- Bot 复盘显示旧 `close_on_tcp_flags=true` 会把同一连接切成大量无 payload 关闭片段；v16 改为 `false` 后保留 TCP close/reset 信息但不单独切训练 flow。
- Infiltration support 极低，不能作为主类别可训练性结论。
- 后续应保留 v16 作为 flow-level Byte-BERT baseline；若继续提升 Bot，应设计 host/session-window 辅助判别或更语义化的 SCE，而不是继续盲调 class weight 或 context token。

## 5. SCE 实验路线

SCE 不阻塞当前 baseline，但论文路线应保留三阶段：

1. Byte-BERT baseline：固定 byte token 和 masked header packet。
2. SCE 最小版：将 byte/header/flow pattern 转成语义 token 或 codebook。
3. SCE + 自监督：在更多未标注或辅助数据上做预训练，再验证泛化。

SCE 实现前需要明确：

- 语义 token 的构造规则或学习方式。
- 是否保留原始 byte token 作为 fallback。
- SCE 输出如何接入现有 `FlowWindowDataset`。
- 与 Byte-BERT baseline 使用同一 split 和同一验收指标。

当前 SCE v0 状态：

- 已实现频率 byte-chunk codebook 骨架：`src/traffic_bert/sce.py`。
- 已提供 CLI：`traffic-bert sce build-codebook`。
- 已接入 `FlowWindowDataset`、classifier train/eval、threshold calibration 和 PCAP/hex prediction。
- classifier checkpoint 会嵌入 codebook 元数据，评估/预测可以自动恢复。
- 已在远端 CICIDS train 抽样 `20000` 行上生成样例：
  `/root/Fork/artifacts/sce/cicids_v0_frequency_codebook_sample.json`。
- v0 默认跳过全零 chunk，避免 masked-header 的零填充主导 codebook。
- v14 已跑 `connection + SCE v0 frequency codebook 64 tokens + sqrt_balanced` 对照，第 1 epoch 后因 Bot 误报过多停止。
- v14 test 指标：macro-F1 约 `0.780`，Bot precision/recall/F1 约 `0.452/0.761/0.567`。
- 已补充 label-lift codebook 和按标签 reservoir 抽样，v15 使用 Bot 专用 64-token codebook 做 1 epoch probe。
- v15 test 指标：macro-F1 约 `0.783`，Bot precision/recall/F1 约 `0.462/0.777/0.579`；30/60/300 秒 host-window 均未通过。
- 当前结论：朴素高频 byte-chunk 和简单 label-lift byte-chunk 都不优于当前 Byte-BERT baseline；下一版 SCE 必须转向更有语义的 chunk/cluster/行为 token。

## 6. PCAP 展示系统

第一版 demo 只支持 PCAP/PCAPNG 上传：

```text
上传 PCAP
  -> flow 解析
  -> masked-header packet view
  -> checkpoint 推理
  -> flow 级结果
  -> 文件级风险汇总
```

展示系统验收：

- 能处理一个小型 PCAP。
- 能列出 flow 数、攻击/良性预测数量和高风险 flow。
- 能展示每个 flow 的 major prediction、confidence 和可选 minor labels。
- 能展示 host/session-window 风险汇总，尤其用于解释 Bot 这类单 flow 难以稳定区分的行为。
- 后端复用 `traffic-bert predict pcap` 的能力，不另造一套解析逻辑。
- 当前 CLI 后端已输出 `{summary, flows}`，UI 上传页可以直接调用该能力。

## 7. 后续泛化实验

IoT-23 和 CTU-13 暂不进入当前工程目标。它们的进入条件：

- CICIDS baseline 和 demo 已稳定。
- 训练后验收 gate 已经能阻断 high-accuracy/low-key-class-recall 的 run。
- SCE 最小版有明确输入输出。

泛化实验要单独报告 dataset shift，不与 CICIDS 当前主结果混成一个指标。

## 8. 图表与论文材料

当前阶段优先生成：

- 数据处理流程图。
- Byte-BERT baseline 结构图。
- SCE 概念图。
- CICIDS2017 类别分布表。
- Coverage/audit 摘要表。
- Confusion matrix。
- Per-class F1/recall 柱状图。
- PCAP demo 流程截图或接口示意。
