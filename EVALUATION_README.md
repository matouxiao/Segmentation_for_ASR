# Checkpoint评估说明

本目录包含用于创建评估集和批量评估checkpoint的脚本。

## 文件说明

- `create_eval_set.py`: 从训练数据中随机采样创建评估集
- `evaluate_all_checkpoints.py`: 批量评估所有checkpoint的性能
- `data/eval_set.jsonl`: 生成的评估集（1000条数据）
- `evaluation_results.json`: 所有checkpoint的评估结果汇总

## 使用方法

### 1. 创建评估集

从 `large_data_train.jsonl` 中随机采样数据创建评估集：

```bash
python create_eval_set.py --num_samples 1000 --output data/eval_set.jsonl
```

参数说明：
- `--input`: 输入JSONL文件路径（默认: `data/large_data_train.jsonl`）
- `--output`: 输出JSONL文件路径（默认: `data/eval_set.jsonl`）
- `--num_samples`: 采样数量（默认: 1000）
- `--seed`: 随机种子（默认: 42）

### 2. 批量评估所有Checkpoint

评估所有8个checkpoint的性能：

```bash
python evaluate_all_checkpoints.py --eval_file data/eval_set.jsonl --output evaluation_results.json
```

参数说明：
- `--checkpoint_dir`: Checkpoint目录路径（默认: `./output_model`）
- `--eval_file`: 评估数据文件路径（默认: `data/eval_set.jsonl`）
- `--output`: 输出结果文件路径（默认: `evaluation_results.json`）
- `--num_checkpoints`: Checkpoint数量（默认: 8）

### 3. 后台运行评估

由于评估8个checkpoint需要较长时间（约30-40分钟），可以在后台运行：

```bash
nohup python evaluate_all_checkpoints.py --eval_file data/eval_set.jsonl --output evaluation_results.json > eval_log.txt 2>&1 &
```

查看进度：
```bash
tail -f eval_log.txt
```

## 评估结果格式

`evaluation_results.json` 包含以下信息：

```json
{
  "eval_file": "data/eval_set.jsonl",
  "checkpoint_dir": "./output_model",
  "device": "cuda",
  "checkpoints": [
    {
      "epoch": 1,
      "checkpoint_path": "./output_model/checkpoint_epoch_1",
      "metrics": {
        "accuracy": 0.xxxx,
        "precision": 0.xxxx,
        "recall": 0.xxxx,
        "f1": 0.xxxx,
        "precision_per_class": [0.xxxx, 0.xxxx],
        "recall_per_class": [0.xxxx, 0.xxxx],
        "f1_per_class": [0.xxxx, 0.xxxx],
        "confusion_matrix": [[xx, xx], [xx, xx]]
      }
    },
    ...
  ],
  "summary": {
    "total_checkpoints": 8,
    "successful_evaluations": 8,
    "best_checkpoint": {
      "epoch": X,
      "f1_score": 0.xxxx,
      "accuracy": 0.xxxx,
      "precision": 0.xxxx,
      "recall": 0.xxxx
    },
    "f1_scores": {
      "1": 0.xxxx,
      "2": 0.xxxx,
      ...
    }
  }
}
```

## 注意事项

1. 评估过程需要GPU支持，确保CUDA可用
2. 评估8个checkpoint需要较长时间，建议在后台运行
3. 评估过程中会占用GPU内存，确保有足够的显存
4. 评估结果会自动保存到指定的输出文件
