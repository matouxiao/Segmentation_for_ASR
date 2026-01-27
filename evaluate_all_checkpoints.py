#!/usr/bin/env python
"""
批量评估所有checkpoint的性能
"""
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# 导入训练脚本中的配置和测试脚本中的评估函数
from train import load_config, CONFIG
from test import evaluate_model, load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_checkpoint_paths(base_dir, num_checkpoints=8):
    """
    获取所有checkpoint路径
    
    Args:
        base_dir: checkpoint基础目录
        num_checkpoints: checkpoint数量（默认8）
    
    Returns:
        checkpoint路径列表，按epoch顺序排序
    """
    checkpoint_paths = []
    for epoch in range(1, num_checkpoints + 1):
        checkpoint_path = os.path.join(base_dir, f"checkpoint_epoch_{epoch}")
        if os.path.exists(checkpoint_path):
            checkpoint_paths.append((epoch, checkpoint_path))
        else:
            print(f"警告: checkpoint路径不存在: {checkpoint_path}")
    
    return checkpoint_paths


def evaluate_all_checkpoints(checkpoint_dir, eval_file, config, output_file=None):
    """
    评估所有checkpoint
    
    Args:
        checkpoint_dir: checkpoint目录路径
        eval_file: 评估数据文件路径
        config: 配置字典
        output_file: 输出结果文件路径（可选）
    
    Returns:
        评估结果字典
    """
    print("=" * 80)
    print("批量评估所有Checkpoint")
    print("=" * 80)
    print(f"Checkpoint目录: {checkpoint_dir}")
    print(f"评估数据文件: {eval_file}")
    print(f"设备: {DEVICE}")
    print("=" * 80)
    print()
    
    # 获取所有checkpoint路径
    checkpoint_paths = get_checkpoint_paths(checkpoint_dir)
    
    if not checkpoint_paths:
        print("错误: 未找到任何checkpoint！")
        return None
    
    print(f"找到 {len(checkpoint_paths)} 个checkpoint")
    print()
    
    # 存储所有checkpoint的评估结果
    all_results = {
        "eval_file": eval_file,
        "checkpoint_dir": checkpoint_dir,
        "device": str(DEVICE),
        "checkpoints": []
    }
    
    # 逐个评估checkpoint
    for epoch, checkpoint_path in tqdm(checkpoint_paths, desc="评估进度"):
        print("\n" + "=" * 80)
        print(f"评估 Checkpoint Epoch {epoch}")
        print(f"路径: {checkpoint_path}")
        print("=" * 80)
        
        try:
            # 加载模型和tokenizer
            model = load_model(checkpoint_path, num_labels=config["model"]["num_labels"])
            tokenizer = BertTokenizer.from_pretrained(checkpoint_path)
            
            # 评估模型
            results = evaluate_model(model, tokenizer, eval_file, config)
            
            # 保存checkpoint结果
            checkpoint_result = {
                "epoch": epoch,
                "checkpoint_path": checkpoint_path,
                "metrics": {
                    "accuracy": results["accuracy"],
                    "precision": results["precision"],
                    "recall": results["recall"],
                    "f1": results["f1"],
                    "precision_per_class": results["precision_per_class"],
                    "recall_per_class": results["recall_per_class"],
                    "f1_per_class": results["f1_per_class"],
                    "confusion_matrix": results["confusion_matrix"]
                }
            }
            
            all_results["checkpoints"].append(checkpoint_result)
            
            # 打印当前checkpoint结果
            print(f"\nEpoch {epoch} 评估结果:")
            print(f"  Accuracy:  {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall:    {results['recall']:.4f}")
            print(f"  F1-score:  {results['f1']:.4f}")
            
            # 清理GPU内存
            del model
            del tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"错误: 评估 checkpoint_epoch_{epoch} 时出错: {e}")
            import traceback
            traceback.print_exc()
            all_results["checkpoints"].append({
                "epoch": epoch,
                "checkpoint_path": checkpoint_path,
                "error": str(e)
            })
    
    # 计算汇总统计
    valid_results = [r for r in all_results["checkpoints"] if "metrics" in r]
    if valid_results:
        # 找出最佳checkpoint（按F1-score）
        best_checkpoint = max(valid_results, key=lambda x: x["metrics"]["f1"])
        all_results["summary"] = {
            "total_checkpoints": len(checkpoint_paths),
            "successful_evaluations": len(valid_results),
            "best_checkpoint": {
                "epoch": best_checkpoint["epoch"],
                "f1_score": best_checkpoint["metrics"]["f1"],
                "accuracy": best_checkpoint["metrics"]["accuracy"],
                "precision": best_checkpoint["metrics"]["precision"],
                "recall": best_checkpoint["metrics"]["recall"]
            },
            "f1_scores": {r["epoch"]: r["metrics"]["f1"] for r in valid_results},
            "accuracy_scores": {r["epoch"]: r["metrics"]["accuracy"] for r in valid_results}
        }
        
        # 打印汇总结果
        print("\n" + "=" * 80)
        print("评估汇总")
        print("=" * 80)
        print(f"成功评估的checkpoint数量: {len(valid_results)}/{len(checkpoint_paths)}")
        print(f"\n最佳Checkpoint: Epoch {best_checkpoint['epoch']}")
        print(f"  F1-score:  {best_checkpoint['metrics']['f1']:.4f}")
        print(f"  Accuracy:  {best_checkpoint['metrics']['accuracy']:.4f}")
        print(f"  Precision: {best_checkpoint['metrics']['precision']:.4f}")
        print(f"  Recall:    {best_checkpoint['metrics']['recall']:.4f}")
        
        print(f"\n所有Checkpoint的F1-score:")
        for epoch, f1 in sorted(all_results["summary"]["f1_scores"].items()):
            marker = " <-- 最佳" if epoch == best_checkpoint["epoch"] else ""
            print(f"  Epoch {epoch}: {f1:.4f}{marker}")
    
    # 保存结果
    if output_file:
        print(f"\n正在保存评估结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print("结果已保存！")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='批量评估所有checkpoint')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='./output_model',
                       help='Checkpoint目录路径（默认: ./output_model）')
    parser.add_argument('--eval_file', type=str,
                       default='data/eval_set.jsonl',
                       help='评估数据文件路径（默认: data/eval_set.jsonl）')
    parser.add_argument('--output', type=str,
                       default='evaluation_results.json',
                       help='输出结果文件路径（默认: evaluation_results.json）')
    parser.add_argument('--num_checkpoints', type=int, default=8,
                       help='Checkpoint数量（默认: 8）')
    
    args = parser.parse_args()
    
    # 检查评估文件是否存在
    if not os.path.exists(args.eval_file):
        print(f"错误: 评估文件不存在: {args.eval_file}")
        print("请先运行 create_eval_set.py 创建评估集")
        return
    
    # 加载配置
    config = CONFIG
    
    # 评估所有checkpoint
    results = evaluate_all_checkpoints(
        args.checkpoint_dir,
        args.eval_file,
        config,
        args.output
    )
    
    if results:
        print("\n评估完成！")


if __name__ == '__main__':
    main()
