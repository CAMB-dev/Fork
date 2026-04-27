# 基于 Payload Byte-BERT 的恶意流量检测

本项目是一个毕设工程，目标是基于开源网络流量数据集，构建一套面向原始 payload/packet bytes 的恶意流量检测模型。整体路线采用 NLP 序列建模思想：将原始字节流转换为 BERT 可处理的文本化 token 序列，再通过 Byte-BERT 进行预训练和分类微调。

## 项目目标

- 从公开数据集下载、解析、清洗并统一恶意流量标签。
- 以 flow 为样本单位重组 PCAP 中的网络流量。
- 将二进制 payload 或完整 packet bytes 文本化为字节 token 序列。
- 使用 BERT 架构进行 MLM 预训练和监督分类微调。
- 输出大类攻击类型，并在大类内部按阈值激活细分类结果。
- 支持可复现实验、消融实验、baseline 对比和命令行推理。

## 核心链路

```text
原始 PCAP/标签
  -> flow 级重组
  -> payload/full/masked-header 三种输入视图
  -> 标签统一
  -> 字节文本化 tokenizer/encoder
  -> Byte-BERT MLM 预训练
  -> 层级分类微调
  -> 大类 + 细类阈值输出
```

## 已确定技术路线

- 项目管理：`uv` + Python 3.10。
- 深度学习框架：PyTorch + CUDA。
- 模型主线：Byte-BERT，小型 BERT 配置优先适配 4GB 显存。
- 输入粒度：flow 级样本，不采用单包样本作为主实验。
- 字节映射：`b_00` 到 `b_ff` 的固定字节词表，加 BERT special tokens。
- 长序列处理：滑窗切分，窗口级 BERT 表示再聚合为 flow 级表示。
- 分类策略：大类 softmax 单选，子类 sigmoid 多标签阈值激活。
- 数据划分：按文件或时间划分，避免随机 flow 划分造成数据泄漏。

详细决策见 [docs/project_decisions.md](docs/project_decisions.md)。

## 目录结构

```text
.
├── configs/                 # 训练、数据和实验配置
├── data/                    # 本地数据目录，不提交大文件
├── docs/                    # 项目文档和毕设思路沉淀
├── scripts/                 # 一次性脚本或入口封装
├── src/traffic_bert/        # 项目源码
└── tests/                   # 测试代码
```

## 数据目录约定

```text
data/
├── raw/                     # 原始 PCAP、CSV、压缩包
├── interim/                 # 解析后的中间产物
└── processed/               # 训练可直接读取的 Parquet/manifest/stats
```

`data/` 目录只保留 `.gitkeep`，原始数据、处理后数据、模型权重和实验产物都不进入 Git。

## 后续实现顺序

1. 实现数据集 manifest 和标签映射配置。
2. 实现 PCAP 到 flow 的解析与三种视图构建。
3. 实现字节 tokenizer 和滑窗数据集。
4. 实现 Byte-BERT MLM 预训练。
5. 实现层级分类模型、训练和评估。
6. 实现 baseline、消融实验和 CLI 推理。

## 当前已实现命令

```powershell
uv run traffic-bert version
uv run traffic-bert vocab write artifacts/vocab.txt
uv run traffic-bert data build --input-path data/raw/sample.pcap --output-path data/processed/train.parquet --source-dataset custom --label-source filename
uv run traffic-bert train mlm --train-path data/processed/train.parquet
uv run traffic-bert train classifier --train-path data/processed/train.parquet --val-path data/processed/val.parquet
uv run traffic-bert eval classifier --data-path data/processed/test.parquet --checkpoint artifacts/classifier/classifier.pt
uv run traffic-bert predict hex "474554202f20485454502f312e31" --checkpoint artifacts/classifier/classifier.pt
```

本仓库固定使用 Python 3.11。若 `uv` 首次运行时没有合适解释器，可以先执行：

```powershell
uv python install 3.11
```
