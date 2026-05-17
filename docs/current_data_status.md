# 当前数据状态

本文档只记录当前阶段需要知道的数据事实和已知问题。历史实验记录不作为当前正式结论。

## 1. 当前阶段结论

- 当前工程主线是 CICIDS2017 全量 PCAP 的 Byte-BERT baseline。
- 当前可训练主 split 是 `all_masked_header_split_submode_stratified_group_cap32_notcpclose`。
- `time_block` 只作为时间/子形态外推压力测试。
- USTC/Payload-Byte 只作为 sanity、诊断或辅助预训练候选。
- IoT-23/CTU-13 暂不处理，后续用于泛化验证。

## 2. 本地 CICIDS2017 状态

本地 `data/raw/CICIDS2017` 已发现五天 PCAP 和官方 label zip：

```text
Monday-WorkingHours.pcap
Tuesday-WorkingHours.pcap
Wednesday-workingHours.pcap
Thursday-WorkingHours.pcap
Friday-WorkingHours.pcap
GeneratedLabelledFlows.zip
```

本地 `data/processed/cicids2017` 目前主要是历史 `all_payload_only`、Friday/smoke 产物。这些产物不能作为当前正式 baseline 训练输入。

本地若要重建正式 processed 数据，应运行：

```bash
SKIP_DOWNLOAD=1 FORCE=1 bash scripts/prepare_cicids2017_all_parallel.sh
```

正式本地产物应包含：

```text
data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose/flows_all.parquet
data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/
artifacts/cicids2017_coverage_masked_header_cap32_notcpclose/attack_coverage.json
artifacts/cicids2017_all_masked_header_cap32_notcpclose/submode_stratified_group_split_audit.json
```

## 3. 云端 CICIDS2017 状态

云端已完成当前可信版本的 full-PCAP masked-header no-close 数据构建、gate 验证和 Byte-BERT baseline 训练：

- raw PCAP：Monday-Friday 五天均存在。
- processed dir：`/root/Fork/data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose`
- merged flows：`1,841,658` 行。
- 主 split：`/root/Fork/data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose`
- split support：train/val/test 分别为 `1,286,920 / 274,768 / 279,970` 行。
- Bot support：train/val/test 分别为 `856 / 178 / 194` 行。
- attack coverage：通过。
- submode-stratified group split audit：通过。
- formal dataset gate：通过。

当前最强 flow-level Byte-BERT baseline run 是：

```text
/root/Fork/artifacts/cicids2017_masked_header_submode_group_formal_cuda_v16_cap32_notcpclose_conn_b128_w4_sqrt_weighted
```

关键 test 指标：

- accuracy：`0.9978`。
- macro-F1：`0.8117`。
- weighted-F1：`0.9978`。
- Bot：precision `0.6146`，recall `0.9948`，F1 `0.7598`，test support `194`。
- acceptance：`formal_eligible=true`，关键类 Bot/BruteForce/DoS-DDoS/Scan/Web Attack 均通过当前验收线。
- overfit signal：`train_val_accuracy_gap` 约 `0.00008`，无过拟合 warning。
- warnings：Infiltration test support 只有 `2`，仅 low-support reported-only；BruteForce 截断比例约 `0.42`，需要在报告中说明 `max_packets_per_flow=32` 的截断口径。

已证伪或降级的路线：

- v11：`connection + context tokens`，Bot val F1 约 `0.580`，误报过多，已停止。
- v12：`connection + balanced class weighting`，Bot val F1 约 `0.596`，误报过多，已停止。
- v13：`connection + numeric context side-channel`，Bot val F1 约 `0.599`，误报过多，已停止。
- v14：`connection + SCE v0 frequency codebook 64 tokens + sqrt_balanced`，第 1 epoch 后停止；test Bot precision `0.452`、recall `0.761`、F1 `0.567`，误报过多，低于当前 Byte-BERT baseline。
- v15：`connection + SCE label-lift Bot codebook 64 tokens + sqrt_balanced`，1 epoch probe；test Bot precision `0.462`、recall `0.777`、F1 `0.579`，host-window 也未通过，低于当前 Byte-BERT baseline。
- Bot threshold scan 未找到同时满足 `recall >= 0.8` 且 `F1 >= 0.75` 的阈值。

Bot 复盘结论：

- 当前问题不像时间 offset 或数据漏处理。
- Bot TP、Bot FN 和 benign->Bot FP 都集中在短 TCP reset/control flow：通常 `1-2` 包、payload 为 `0`、duration 极短。
- 2026-05-15 本地复核确认：在当前 initiator/responder 匹配逻辑下，`csv_time_offset_hours=3.0` 能匹配 Bot；`+4h` 只匹配到 BENIGN，匹配不到 Bot。
- 2026-05-15 本地复核还显示：旧 formal 口径 `close_on_tcp_flags=true` 会把 Bot 从 `1228` 个连接样本切成 `2915` 个 flow，其中 `2177` 个 Bot flow 无 payload；`close_on_tcp_flags=false` 下仍能匹配 Bot `1228` 条，其中 payload Bot `738` 条、empty/control Bot `490` 条。
- 因此 v10 视为旧 TCP-close 切分口径 baseline；v16 已按 `close_on_tcp_flags=false` 重建数据并重新训练，Bot flow-level 指标已达到毕设可交付口径。
- 新增诊断脚本：`scripts/analyze_bot_host_windows.py`。
- v16 预测诊断输出：
  `/root/Fork/artifacts/cicids2017_masked_header_submode_group_formal_cuda_v16_cap32_notcpclose_conn_b128_w4_sqrt_weighted/bot_host_window_analysis.json`
- v16 flow-level Bot 已通过验收，host-window 不再是弥补 Bot 失败的必要项。
- v16 host-window policy 没有启用项：30s test 可过，但 val F1 略低于阈值；60s/300s 均未同时通过 val/test。展示系统可以保留 host-risk 汇总，但不需要把它写成正式指标。

低支持类口径：

- Infiltration test support 只有 `2`，当前不适合作为可训练主类别验收，只能 low-support reported-only。

## 4. USTC 状态

USTC-TFC2016 当前只作为辅助诊断数据。

已知口径：

- `split_label_stratified`：source label 分层，适合 sanity/upper-bound，但同一源 PCAP 会跨 train/val/test。
- `split_source_file`：严格文件级诊断，但 malware family 基本单 PCAP，minor/family 会变成 unseen-family 评估。
- `split_source_file_major_balanced`：用于 benign/malware 大类诊断。

USTC minor/family 结果不作为严格正式主结论。

## 5. Payload-Byte 状态

Payload-Byte 是单包 payload 行级数据，不含完整五元组和双向 flow 边界。它不直接混入当前 CICIDS 监督训练。

后续可以作为：

- 快速 sanity。
- 自监督预训练候选。
- 非 PCAP 输入对照。

## 6. 当前风险清单

- High accuracy 仍然容易掩盖低支持类失败，报告时必须列 per-class 指标。
- Heartbleed/Infiltration 样本极少，不能作为当前可训练主类别验收。
- 本地旧 payload/smoke parquet 很容易被误用，训练前必须依赖 formal gate。
- 当前 cap32/notcpclose 数据链路已优先于旧 cap16、cap32/tcpstate 结果；历史 run 只能作参考，不能作为当前正式结论。
- `time_block` 压力测试可能产生训练集中没有某攻击子形态的情况，结果应单独解释。
- SCE v0 codebook 骨架已实现、接入编码和训练/评估入口，并生成远端样例；当前没有优于 v16 Byte-BERT baseline 的 SCE 对照结果，已完成可采用结果仍是方案1 Byte-BERT baseline。
- SCE label-lift codebook 已支持按标签 reservoir 抽样；当前 Bot 专用 label-lift probe 不优于 Byte-BERT baseline，不能继续加长训练当作修复方向。

## 7. 当前通过标准

一个 run 想进入正式报告，至少需要：

- 数据 gate 通过。
- Split audit 通过。
- Attack coverage 通过。
- Per-class 指标完整。
- Bot 等 support 足够的关键类 recall/F1 达标；当前 v16 已达标。
- Low-support 类单独标记 reported-only。
- 不把 accuracy 或 weighted-F1 作为单独通过依据。
