# AMD ROCm 服务器训练说明

本项目在 ROCm 服务器上仍使用 PyTorch 的 `cuda` 设备命名。原因是 PyTorch ROCm
构建会通过 `torch.cuda.is_available()` 暴露 AMD GPU，并通过 `torch.version.hip`
标识 ROCm/HIP 后端。

## 1. 准备环境

默认假设服务器是 Ubuntu Linux，已经安装官方 AMD ROCm 驱动栈，并且可以运行：

```bash
rocminfo
rocm-smi
```

安装项目依赖和 ROCm PyTorch：

```bash
export ROCM_VERSION=6.4
bash scripts/install_rocm_torch.sh
```

如果服务器 ROCm 版本不同，按 PyTorch 官方 selector 或 AMD ROCm 文档选择对应版本：

```bash
export ROCM_VERSION=6.3
bash scripts/install_rocm_torch.sh
```

也可以直接覆盖 wheel index：

```bash
export TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.4
bash scripts/install_rocm_torch.sh
```

## 2. 检查 ROCm 可用性

```bash
bash scripts/check_rocm_env.sh
```

成功条件：

- `torch.cuda.is_available()` 为 `true`
- `torch.version.hip` 非空
- 至少能看到 1 个 GPU 名称

如果 `torch.version.hip` 是 `null`，通常表示装到了 CPU/CUDA 版 PyTorch，而不是 ROCm 版。

## 3. 先跑小规模 smoke

建议先用小参数验证训练链路、step 日志、checkpoint 和 eval 输出：

```bash
PLAN=bert-supervised \
OUTPUT_ROOT=artifacts/rocm_smoke \
CLASSIFIER_EPOCHS=1 \
CLASSIFIER_BATCH_SIZE=2 \
MAX_LENGTH=128 \
MAX_WINDOWS=1 \
bash scripts/train_ustc_formal_rocm.sh
```

检查输出：

```bash
ls artifacts/rocm_smoke/classifier/history
cat artifacts/rocm_smoke/split_audit.json
```

至少应包含：

- `history/steps.csv`
- `history/train.csv`
- `checkpoints/epoch_001.pt`
- `eval/metrics.json`
- `test_eval/metrics.json`

## 4. 正式训练

第一轮建议先跑监督分类，不默认跑 MLM：

```bash
PLAN=bert-supervised \
OUTPUT_ROOT=artifacts/ustc_formal_rocm \
CLASSIFIER_EPOCHS=3 \
CLASSIFIER_BATCH_SIZE=4 \
MAX_LENGTH=512 \
MAX_WINDOWS=4 \
bash scripts/train_ustc_formal_rocm.sh
```

链路稳定后再跑 MLM + classifier：

```bash
PLAN=bert-mlm \
OUTPUT_ROOT=artifacts/ustc_formal_rocm_mlm \
MLM_EPOCHS=1 \
CLASSIFIER_EPOCHS=3 \
bash scripts/train_ustc_formal_rocm.sh
```

中断后继续：

```bash
RESUME=1 OUTPUT_ROOT=artifacts/ustc_formal_rocm bash scripts/train_ustc_formal_rocm.sh
```

## 5. 结果解释

服务器正式结果必须同时查看：

- `split_audit.json`：确认数据划分是否存在 source file 重叠。
- `history/steps.csv`：每 step loss/accuracy 曲线。
- `history/train.csv`：每 epoch 的 `train_loss`、`val_loss`、`val_accuracy`、`val_macro_f1`。
- `test_eval/metrics.json`：最终 `eval_loss`、major、detection、minor、minor_constrained 指标。

旧的 `split_label_stratified` 结果只适合作为 pipeline sanity check。论文或最终报告应优先使用
strict split 或跨数据集测试结果。

## 参考

- [PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/)
- [AMD ROCm PyTorch 文档](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.1/install/3rd-party/pytorch-install.html)
