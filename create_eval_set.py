#!/usr/bin/env python
"""
从训练数据中随机采样创建评估集
"""
import json
import random
import argparse
from pathlib import Path


def create_eval_set(input_file, output_file, num_samples=1000, seed=42):
    """
    从输入文件中随机采样指定数量的数据创建评估集
    
    Args:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
        num_samples: 采样数量（默认1000）
        seed: 随机种子（默认42）
    """
    # 设置随机种子
    random.seed(seed)
    
    print(f"正在读取数据文件: {input_file}")
    
    # 读取所有数据
    all_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                all_data.append(item)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行解析失败: {e}")
                continue
    
    total_count = len(all_data)
    print(f"总共读取 {total_count} 条数据")
    
    # 确定采样数量
    if num_samples > total_count:
        print(f"警告: 请求采样 {num_samples} 条，但只有 {total_count} 条数据，将采样全部数据")
        num_samples = total_count
    
    # 随机采样
    print(f"正在随机采样 {num_samples} 条数据...")
    sampled_data = random.sample(all_data, num_samples)
    
    # 保存评估集
    print(f"正在保存评估集到: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"评估集创建完成！")
    print(f"  - 总数据量: {total_count}")
    print(f"  - 采样数量: {num_samples}")
    print(f"  - 输出文件: {output_file}")
    
    # 统计信息
    if sampled_data:
        segment_counts = [len(item.get('segment_positions', [])) for item in sampled_data]
        avg_segments = sum(segment_counts) / len(segment_counts)
        print(f"  - 平均分段数: {avg_segments:.2f}")
        print(f"  - 最小分段数: {min(segment_counts)}")
        print(f"  - 最大分段数: {max(segment_counts)}")


def main():
    parser = argparse.ArgumentParser(description='从训练数据中创建评估集')
    parser.add_argument('--input', type=str, 
                       default='data/large_data_train.jsonl',
                       help='输入JSONL文件路径（默认: data/large_data_train.jsonl）')
    parser.add_argument('--output', type=str,
                       default='data/eval_set.jsonl',
                       help='输出JSONL文件路径（默认: data/eval_set.jsonl）')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='采样数量（默认: 1000）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认: 42）')
    
    args = parser.parse_args()
    
    create_eval_set(args.input, args.output, args.num_samples, args.seed)


if __name__ == '__main__':
    main()
