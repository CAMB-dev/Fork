| label | precision | recall | f1 | support | paper_usage |
| --- | --- | --- | --- | --- | --- |
| benign | 0.9999 | 0.9975 | 0.9987 | 240689 | 主结果背景类 |
| dos_ddos | 0.9699 | 0.9982 | 0.9838 | 13903 | 关键攻击类，可作为主结果 |
| bruteforce | 0.9934 | 0.9981 | 0.9957 | 1056 | 关键攻击类，可作为主结果；需说明截断比例约 0.42 |
| web_attack | 0.8934 | 1.0000 | 0.9437 | 285 | 关键攻击类，可作为主结果，但 support 较小 |
| botnet_malware | 0.6146 | 0.9948 | 0.7598 | 194 | 关键攻击类，可保留；no-close 口径修复后通过阈值 |
| scan | 0.9999 | 0.9998 | 0.9999 | 23841 | 关键攻击类，可作为主结果 |
| infiltration | 0.0000 | 0.0000 | 0.0000 | 2 | low-support reported-only，不作为主类验收 |
| heartbleed | 0.0000 | 0.0000 | 0.0000 | 0 | test support 为 0，不作为主类验收 |
| other_attack | 0.0000 | 0.0000 | 0.0000 | 0 | test support 为 0，不作为主类验收 |
