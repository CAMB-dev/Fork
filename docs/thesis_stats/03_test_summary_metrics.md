| metric | value | paper_usage | note |
| --- | --- | --- | --- |
| accuracy | 0.9978 | 辅助 | 不能单独作为效果结论，类别不均衡下易偏高 |
| macro_precision | 0.7816 | 主结果 | 衡量各 major class 平均精度 |
| macro_recall | 0.8555 | 主结果 | 衡量各 major class 平均召回 |
| macro_f1 | 0.8117 | 主结果 | 当前 Byte-BERT baseline 的核心 test 指标 |
| weighted_precision | 0.9980 | 补充 | 受 benign 大类占比影响 |
| weighted_recall | 0.9978 | 补充 | 受 benign 大类占比影响 |
| weighted_f1 | 0.9978 | 补充 | 可报告，但不能替代 macro/per-class 分析 |
| balanced_accuracy | 0.8555 | 补充 | 与 macro recall 等价或接近，适合不均衡场景 |
| eval_loss | 0.0122 | 辅助 | 训练记录用，不作为论文主效果 |
