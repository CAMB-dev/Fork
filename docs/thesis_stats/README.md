# CICIDS2017 Byte-BERT v16 论文结果表

本目录只整理当前正式 v16 run 的论文结果表，不再展开原始 PCAP 或 processed 数据的大统计。现在写论文时，重点使用 macro-F1、攻击检测指标、per-class 指标和混淆矩阵，不把 accuracy 单独作为效果结论。

## 来源文件

- 训练摘要：`artifacts/cloud_logs/v16_cap32_notcpclose/formal_run_summary.json`
- 测试指标：`artifacts/cloud_logs/v16_cap32_notcpclose/test_metrics.json`
- 训练过程：`artifacts/cloud_logs/v16_cap32_notcpclose/history/train.csv`
- 已同步的测试混淆矩阵：`artifacts/cloud_logs/v16_cap32_notcpclose/test_eval/confusion_matrix.csv`
- 服务器正式 run：`/root/Fork/artifacts/cicids2017_masked_header_submode_group_formal_cuda_v16_cap32_notcpclose_conn_b128_w4_sqrt_weighted`

## 表格用途

| 文件 | 论文用途 |
| --- | --- |
| `01_training_history.*` | 训练过程表；第 2 epoch 的 validation macro-F1 最好。 |
| `02_formal_run_summary.*` | gate 和 checkpoint 选择证据；说明 v16 是正式 baseline。 |
| `03_test_summary_metrics.*` | test 总指标；优先报告 macro-F1，weighted-F1 作为补充。 |
| `04_attack_detection_metrics.*` | benign-vs-attack 二分类检测指标；适合系统检测能力讨论。 |
| `05_per_class_major_metrics.*` | 最重要的分类结果表；用它支撑各攻击类分析。 |
| `06_confusion_matrix.*` | 错分分布；用于解释 false positive / false negative。 |
| `07_acceptance_warnings.*` | 论文 caveat，必须在实验分析或局限性中说明。 |

## 可直接写进论文的表述

建议写法：

> 在 CICIDS2017 全量 PCAP 正式划分上，Byte-BERT baseline 的 test macro-F1 为 0.8117，weighted-F1 为 0.9978。在 benign-vs-attack 二分类检测口径下，attack precision 为 0.9852，attack recall 为 0.9993，attack F1 为 0.9922，FPR 为 0.0025，FNR 为 0.0007。

不要把 accuracy 单独写成效果结论。当前 accuracy 为 0.9978，但它明显受 benign 大类占比影响；更可信的主结论应来自 macro-F1、攻击检测指标和 per-class 指标。

## 必须保留的 caveat

- Bot 可以保留为当前阶段可交付类别：no-close flow 口径修复后，test precision 0.6146，recall 0.9948，F1 0.7598，support 194。
- Infiltration 是 low-support reported-only：test support 2，不能作为主类训练成功来验收。
- Heartbleed 在 test 中 support 为 0，不能用于 class-level 验收。
- BruteForce 指标通过，但 packet truncation ratio 约 0.42，需要作为输入窗口截断 caveat 说明。
- SCE v14/v15 只能写成失败探索或后续方向，不能写成优于 Byte-BERT baseline 的有效提升。
