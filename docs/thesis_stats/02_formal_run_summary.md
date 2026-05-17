| item | value | paper_usage |
| --- | --- | --- |
| formal_gate | pass | 数据与训练 gate 通过，可写入实验设置 |
| split_audit | pass | split audit 通过，可写入实验设置 |
| acceptance.formal_eligible | true | 当前 v16 baseline 可作为正式主结果 |
| best_val_epoch | 2 | 按验证集 macro-F1 选择 checkpoint |
| best_val_macro_f1 | 0.8111 | 模型选择依据，不替代 test 结论 |
| test_macro_f1 | 0.8117 | 主效果指标 |
| test_weighted_f1 | 0.9978 | 补充指标，需说明类别不均衡 |
