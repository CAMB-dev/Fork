# 当前 Goal Audit

本文档记录当前阶段目标的完成证据和缺口。它不是论文结论，只用于防止把局部通过误判为整体完成。

## 1. 目标拆解

当前目标：

- CICIDS2017 全量 PCAP 的 Byte-BERT baseline。
- 可复现、可审计的数据处理链路。
- 正式训练与评估 gate。
- PCAP/PCAPNG 上传推理 demo。
- SCE/codebook 作为下一阶段核心创新路线。
- `time_block` 仅作为时间/子形态外推压力测试。
- USTC/Payload-Byte 仅作为 sanity、诊断或辅助预训练来源。
- IoT-23/CTU-13 暂缓到后续泛化验证。

## 2. 已完成并有证据

### CICIDS2017 数据链路

当前可信数据版本：

```text
/root/Fork/data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose
/root/Fork/data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose
```

已验证事实：

- Monday-Friday 五天 PCAP 均存在。
- `flows_all.parquet` 为 full-PCAP processed 产物。
- attack coverage gate 通过。
- split audit 通过。
- formal dataset gate 通过。
- 旧 Friday/smoke/payload-only 产物不作为正式输入。

### Flow-Level Byte-BERT Baseline

当前最强 flow-level baseline：

```text
/root/Fork/artifacts/cicids2017_masked_header_submode_group_formal_cuda_v16_cap32_notcpclose_conn_b128_w4_sqrt_weighted
```

证据：

- 使用 `connection tokens + sqrt_balanced`。
- no-close 数据口径：`close_on_tcp_flags=false`，保留 TCP close/reset 信息但不把关闭包单独切成训练 flow。
- best validation epoch 是第 2 轮，`val_macro_f1=0.8111`。
- test macro-F1 `0.8117`，weighted-F1 `0.9978`。
- test Bot precision/recall/F1 约 `0.615 / 0.995 / 0.760`。
- BruteForce、DoS/DDoS、Scan、Web Attack、Bot 等关键类通过当前验收阈值。
- Infiltration 属于 low-support reported-only。

当前判断：

- v16 是当前最强可交付 baseline，可以作为当前阶段 Byte-BERT baseline 主结果。
- `formal_run_summary.json` 中 `acceptance.formal_eligible=true`。
- 仍需在报告中说明：Infiltration 低支持、BruteForce 截断比例较高、SCE 尚未形成有效提升。

### 已证伪的 Bot 修补路线

以下路线已降级，不应继续盲目重复：

- v11：`connection + context tokens`，Bot val F1 约 `0.580`。
- v12：`connection + balanced loss`，Bot val F1 约 `0.596`。
- v13：`connection + numeric context side-channel`，Bot val F1 约 `0.599`。
- Bot 阈值扫描没有找到同时满足 `recall >= 0.8` 且 `F1 >= 0.75` 的阈值。

Bot 复盘结论：

- 当前问题不像时间 offset 或漏处理；本地复核确认 `csv_time_offset_hours=3.0` 能匹配 Bot，`+4h` 在当前 initiator/responder 匹配逻辑下匹配不到 Bot。
- Bot TP、Bot FN 和 benign->Bot FP 都集中在短 TCP reset/control flow。
- 典型形态是 `1-2` 包、payload 为 `0`、duration 极短。
- Bot 所在 host-minute 窗口混有大量官方 benign flow，host-risk 可以用于展示/告警，但不能直接替代 flow-level 标签。
- 旧 formal build 使用 `close_on_tcp_flags=true`，把 Bot 从 `1228` 个连接样本切为 `2915` 个 flow，其中 `2177` 个无 payload；v16 已使用 `close_on_tcp_flags=false` 重建，Bot test F1 提升到 `0.7598`。

### Host-Window Bot 诊断

已实现并在远端 v10/v16 预测上运行：

```text
scripts/analyze_bot_host_windows.py
/root/Fork/artifacts/cicids2017_masked_header_submode_group_formal_cuda_v16_cap32_notcpclose_conn_b128_w4_sqrt_weighted/bot_host_window_analysis.json
```

关键结论：

- 旧 v10 的 host-window 能缓解 Bot 展示层风险，但不能替代 flow-level 指标。
- v16 flow-level Bot 已通过，因此 host-window 不再是必要补丁。
- v16 host-window policies 均为 `enabled=false`，因为没有窗口同时满足 val/test 要求。
- `bot_host_window_analysis.json` 仍保留 `policies` 字段，demo/CLI 默认只加载 enabled policy。

当前判断：

- host-window 仍可保留为展示系统风险汇总，但不覆盖 flow-level 输出。

### PCAP/PCAPNG 上传推理 Demo

已实现：

```text
scripts/pcap_demo_server.py
```

能力：

- 支持 `.pcap` / `.pcapng` 上传。
- 进程内复用 classifier checkpoint。
- 复用 `PcapFlowExtractor`、Byte tokenizer 和 classifier 推理逻辑。
- 返回 `{summary, flows}`。
- `summary` 包含 flow 统计、major label 计数、connection type 计数、高风险 flow、top host risk 和 enabled host-window policy 风险。

当前可用地址：

```text
http://127.0.0.1:7861/
```

运行方式：

- 远端 5090 服务器运行 demo server。
- 本地 SSH tunnel 转发 `7861:127.0.0.1:7861`。
- 小 PCAP 上传冒烟已通过；从 Monday 原始 PCAP 抽取的 40 包样本返回 2 个 benign flow，高风险 flow 为 0。

### 测试

当前完整测试：

```text
102 passed, 1 warning
```

已知 warning：

- `scripts/pcap_demo_server.py` 使用标准库 `cgi`，Python 3.13 标记 deprecated。
- 项目当前 Python 版本约束是 `<3.12`，该 warning 不阻塞当前 demo。

## 3. 完成度清单

| 目标项 | 当前状态 | 证据 | 是否阻塞 complete |
| --- | --- | --- | --- |
| CICIDS2017 全量 PCAP processed | 已完成 | `flows_all.parquet`、Monday-Friday PCAP、coverage/gate 通过 | 否 |
| 可审计 split 与 formal gate | 已完成 | split audit、formal dataset gate、training preflight | 否 |
| Byte-BERT flow-level baseline | 已完成 | v16 no-close run，关键类通过，Bot test F1 `0.7598` | 否 |
| PCAP/PCAPNG 上传 demo | 已完成 | `scripts/pcap_demo_server.py`，远端 demo + 本地 tunnel，小 PCAP API 冒烟 | 否 |
| Host-window Bot 风险层 | 已完成为辅助层 | v16 flow-level 已通过；host-window 可作为展示汇总但无 enabled policy | 否 |
| SCE/codebook 下一阶段路线 | 已接入骨架但无有效提升 | v14/v15 probe 均低于 Byte-BERT baseline；作为下一阶段路线保留 | 否，当前阶段不阻塞 |
| IoT-23/CTU-13 | 暂缓 | 文档明确后续泛化验证 | 否 |

因此当前阶段的 CICIDS Byte-BERT baseline、数据链路、gate 和 demo 能力已经可交付；SCE 作为下一阶段创新路线，尚不能写成有效提升结论。

## 4. 未完成或不能算完成

### Bot 验收状态

旧 v10 缺口：

- v10 Bot recall `0.7133 < 0.8000`。
- v10 Bot F1 `0.7126 < 0.7500`。

v16 已修复该缺口：

- v16 Bot recall `0.9948`。
- v16 Bot F1 `0.7598`。
- v16 `acceptance.formal_eligible=true`。

### SCE/codebook 已接入但尚未形成有效提升

当前代码已具备 SCE v0 codebook 骨架，并已完成两个失败 probe；它仍未形成可采用的 SCE 方案。

已完成：

- `src/traffic_bert/sce.py`：频率 byte-chunk codebook 和 label-lift codebook。
- `traffic-bert sce build-codebook`：从 processed parquet 构建 codebook JSON，支持按标签 reservoir 抽样。
- `ByteTokenizer(extra_tokens=codebook.tokens)`：为 SCE token 预留 tokenizer 接口。
- `FlowWindowDataset(..., semantic_codebook=...)`：样本编码可使用 SCE token。
- `train/eval classifier`、`eval calibrate-thresholds`、`predict hex/pcap` 支持 `--semantic-codebook-path`。
- `scripts/train_cicids_formal_cuda.sh` 和 `scripts/train_cicids_formal.ps1` 支持 SCE codebook 参数。
- classifier checkpoint 会嵌入 codebook 元数据，评估和预测可自动恢复。
- 远端样例：
  `/root/Fork/artifacts/sce/cicids_v0_frequency_codebook_sample.json`。

已完成但失败的同 split probe：

- v14：频率 codebook，Bot test F1 约 `0.567`，低于当前 Byte-BERT baseline。
- v15：Bot label-lift codebook，Bot test F1 约 `0.579`，host-window 也未通过，低于当前 Byte-BERT baseline。

### SCE v0-64 对照失败记录

远端 run：

```text
/root/Fork/artifacts/cicids2017_masked_header_submode_group_formal_cuda_v14_sce_v0_64_conn_b128_w4_sqrt_weighted
```

配置：

- `connection tokens`
- `sqrt_balanced`
- `SEMANTIC_CODEBOOK_PATH=artifacts/sce/cicids_v0_frequency_codebook_sample.json`
- codebook：4-byte chunk，64 entries，drop-zero chunks

结果：

- 第 1 epoch 后停止。
- val Bot precision/recall/F1 约 `0.495/0.779/0.605`。
- test Bot precision/recall/F1 约 `0.452/0.761/0.567`。
- `formal_run_summary.json` 中 `acceptance.formal_eligible=false`。

判断：

- 朴素高频 byte-chunk SCE v0 会提高 Bot recall 但显著增加 false positive。
- 当前不继续该 v0-64 方向训练。
- 简单 Bot label-lift byte-chunk 也不能解决 Bot；下一版 SCE 应过滤低信息 header/protocol 片段，或改用更有语义的 chunk/cluster/行为 token 规则。

### Host/Session-Window 还不是正式训练模型

当前已有 host-window 诊断和 demo host risk 汇总，但这不是训练模型，也不改变正式 flow-level 指标。

如果要继续改善 Bot，应设计一个明确的辅助任务：

- host/session-window 风险建模。
- 与 flow-level Byte-BERT 的融合方式。
- 与官方 flow label 指标分开报告。

## 5. 下一步建议

优先级从高到低：

1. 固定当前 v16 为 flow-level Byte-BERT baseline，作为当前毕设阶段可交付主结果。
2. 在 demo 中保留 flow-level 结果和 host-risk 结果两个层次，但 host-window 不覆盖 flow label。
3. 报告中明确 Infiltration low-support、BruteForce packet truncation 和 high accuracy 的解释边界。
4. SCE 后续不要继续简单 byte-chunk；转向更明确的语义 chunk/cluster/行为 token。
5. IoT-23/CTU-13 暂不进入当前工程目标。

## 6. Completion audit 2026-05-15

本节把 active goal 拆成 prompt-to-artifact checklist。状态只按真实文件、命令输出和远端 artifact 判断。

| 目标要求 | 证据 | 状态 |
| --- | --- | --- |
| 项目目标收束为 PCAP-first 的 SCE 恶意流量检测毕设系统 | `README.md`、`docs/project_decisions.md`、`docs/model_design.md` 均明确 PCAP-first、Level-0/1/2、Byte-BERT baseline、SCE 下一阶段路线 | 通过 |
| 当前阶段优先 CICIDS2017 全量 PCAP | 远端五个 PCAP 均存在且非空；`flows_all.stats.json` rows/flows 为 `1,841,658`；主 split 为 `all_masked_header_split_submode_stratified_group_cap32_notcpclose` | 通过 |
| Byte-BERT baseline 可训练并有结果 | v16 run：`/root/Fork/artifacts/cicids2017_masked_header_submode_group_formal_cuda_v16_cap32_notcpclose_conn_b128_w4_sqrt_weighted`；test macro-F1 `0.8117`、weighted-F1 `0.9978`、Bot F1 `0.7598` | 通过 |
| 数据处理链路可复现、可审计 | `scripts/prepare_data.py`、`scripts/prepare_cicids2017_all_parallel.py`、`scripts/prepare_datasets.sh`/`.ps1` 默认指向 cap32/notcpclose formal 口径；`attack_coverage.json.formal_eligible=true` | 通过 |
| 正式训练与评估 gate | `scripts/verify_formal_dataset.py` 默认直接检查当前 formal split；服务器默认运行结果 `formal_eligible=true`、`errors=[]`；训练 run 的 `formal_run_summary.json.acceptance.formal_eligible=true` | 通过 |
| PCAP/PCAPNG 上传推理 demo | `scripts/pcap_demo_server.py` 已实现并运行在远端 PID `184198`；本地 SSH tunnel PID `66740`；本地上传 `monday_head_demo_local.pcap` 返回 2 个 benign flow，高风险 0 | 通过 |
| SCE/codebook 作为下一阶段核心路线 | `src/traffic_bert/sce.py`、`traffic-bert sce build-codebook`、classifier/eval/predict/demo 的 `semantic_codebook_path` 接入已存在；`tests/test_sce_codebook.py` 为 `6 passed`；v14/v15 失败 probe 已记录为下一阶段改进依据 | 通过为骨架与路线；不写成有效提升 |
| time-block 仅压力测试 | `README.md`、`docs/experiment_plan.md`、`docs/data_pipeline.md`、`docs/dataset_prepare.md` 均标记 time-block 为压力测试，不作为当前主线 | 通过 |
| USTC/Payload-Byte 仅 sanity/诊断/辅助预训练 | `README.md`、`docs/project_decisions.md`、`docs/dataset_prepare.md`、`docs/current_data_status.md` 均降级 USTC/Payload-Byte，不作为正式主结论 | 通过 |
| IoT-23/CTU-13 暂缓 | `README.md`、`docs/project_decisions.md`、`docs/experiment_plan.md`、`docs/dataset_prepare.md` 均写明当前不处理，后续泛化验证再接入 | 通过 |
| 本地验证覆盖入口改动 | `uv run pytest -q` 为 `102 passed, 1 warning`；PowerShell parser 为 `ps1_ok`；核心脚本 `py_compile` 通过；远端 bash syntax 通过 | 通过 |

Completion audit 结论：当前 active goal 所要求的当前阶段交付物已经完成。剩余 SCE 工作属于下一阶段有效提升，不阻塞本阶段完成；文档中已明确不得把 v14/v15 写成提升结论。
