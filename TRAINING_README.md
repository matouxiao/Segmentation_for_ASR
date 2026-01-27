# 训练脚本使用说明

## 修复的问题

1. **网络连接问题**：修复了模型路径处理，优先使用本地文件，避免从 HuggingFace 下载
2. **从 Checkpoint 恢复训练**：添加了完整的训练状态保存和恢复功能

## 使用方法

### 1. 从头开始训练

```bash
cd /workapp1219/models/Segment
python train.py
```

### 2. 从 Checkpoint 恢复训练

从指定的 checkpoint 继续训练（保留优化器状态、学习率调度器等）：

```bash
# 从 checkpoint_epoch_8 继续训练
python train.py --resume_from checkpoint_epoch_8

# 或使用完整路径
python train.py --resume_from ./output_model/checkpoint_epoch_8
```

## 训练状态保存

从 **下一个 epoch 开始**，每个 checkpoint 会保存：

1. **模型权重** (`model.safetensors`)
2. **Tokenizer** (相关配置文件)
3. **训练状态** (`training_state.pt`)，包括：
   - 优化器状态（Adam 的 momentum、variance 等）
   - 学习率调度器状态
   - 当前 epoch 编号
   - 全局步数
   - 最佳 F1-score
   - 早停计数器
   - GradScaler 状态（如果使用混合精度）

## 重要说明

### 关于数据 Shuffle

- **每个 epoch 都会重新 shuffle**（`shuffle=True`）
- 这是正常的训练实践，有助于模型更好地学习
- 即使从 checkpoint 恢复，数据也会重新 shuffle

### 关于优化器状态

- **如果从 checkpoint 恢复且存在 `training_state.pt`**：
  - ✅ 优化器的 momentum 等状态会保留
  - ✅ 学习率调度器状态会保留
  - ✅ 从正确的 epoch 继续训练

- **如果 checkpoint 中没有 `training_state.pt`**（旧版本保存的）：
  - ⚠️ 优化器状态会丢失（momentum 等会重置）
  - ⚠️ 学习率调度器会重新初始化
  - ✅ 但模型权重会保留，仍然可以从该 checkpoint 继续训练

### 关于现有的 Checkpoint（epoch 1-8）

现有的 checkpoint（`checkpoint_epoch_1` 到 `checkpoint_epoch_8`）**没有保存训练状态**，所以：

- 如果从这些 checkpoint 恢复：
  - ✅ 模型权重会加载
  - ⚠️ 优化器状态会丢失（momentum 等会重置）
  - ⚠️ 学习率调度器会重新初始化
  - ✅ 训练会从指定的 epoch 继续（但优化器是新的）

- 从 **epoch 9 开始**，新的 checkpoint 会包含完整的训练状态

## 配置说明

配置文件 `config.json` 中的路径可以是：

- **相对路径**：相对于 `train.py` 所在目录
- **绝对路径**：完整路径

例如：
```json
{
  "model": {
    "model_path": "sentence_bert_wwm"  // 相对路径，会自动转换为绝对路径
  }
}
```

## 示例

### 场景1：从头训练 10 个 epoch

```bash
python train.py
```

### 场景2：从 epoch 8 继续训练到 epoch 15

```bash
python train.py --resume_from checkpoint_epoch_8
```

注意：`config.json` 中的 `num_epochs` 应该设置为 15 或更大，否则训练会在配置的 epoch 数停止。

### 场景3：从 epoch 8 继续训练，但使用新的优化器状态

如果你想从 epoch 8 的模型权重开始，但使用全新的优化器状态：

1. 修改 `config.json` 中的 `model_path` 为 `"./output_model/checkpoint_epoch_8"`
2. 运行 `python train.py`（不使用 `--resume_from`）

这样会加载 epoch 8 的模型权重，但优化器和调度器会重新初始化。

## 故障排除

### 问题：找不到模型文件

确保 `sentence_bert_wwm` 目录存在于 `train.py` 同级目录下。

### 问题：网络连接错误

代码已修复，会优先使用本地文件。如果仍有问题，检查：
1. `sentence_bert_wwm` 目录是否存在
2. 路径是否正确

### 问题：从 checkpoint 恢复后优化器状态丢失

检查 checkpoint 目录中是否有 `training_state.pt` 文件。如果没有，说明该 checkpoint 是旧版本保存的，不包含训练状态。
