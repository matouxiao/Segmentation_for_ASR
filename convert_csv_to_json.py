#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 CSV 文件转换为 JSON 文件
提取 id 和 text（no_point_text）字段
"""

import csv
import json
import argparse
from pathlib import Path


def convert_csv_to_json(csv_file: str, output_file: str):
    """
    将 CSV 文件转换为 JSON 文件
    
    Args:
        csv_file: 输入 CSV 文件路径
        output_file: 输出 JSON 文件路径
    """
    print(f"开始转换: {csv_file}")
    
    # 读取 CSV 文件
    data = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, start=2):  # 从第2行开始（跳过表头）
                # 提取 id 和 text（no_point_text）
                record = {
                    "id": row.get('id', '').strip(),
                    "text": row.get('no_point_text', '').strip()
                }
                
                # 跳过空记录
                if record['id'] and record['text']:
                    data.append(record)
                else:
                    print(f"警告: 第 {row_num} 行数据不完整，跳过")
                
                # 每处理 1000 条输出一次进度
                if len(data) % 1000 == 0:
                    print(f"已处理: {len(data)} 条")
    
    except Exception as e:
        print(f"错误: 读取 CSV 文件失败: {e}")
        return
    
    print(f"✓ 共读取 {len(data)} 条有效数据")
    
    # 保存为 JSONL 格式（每行一个 JSON 对象）
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"✓ 转换完成！")
        print(f"  输出文件: {output_file}")
        print(f"  总记录数: {len(data)}")
        
        # 显示前3条示例
        print(f"\n示例数据（前3条）:")
        for i, record in enumerate(data[:3], 1):
            print(f"  {i}. ID: {record['id']}")
            print(f"     Text: {record['text'][:80]}...")
    
    except Exception as e:
        print(f"错误: 写入 JSON 文件失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='将 CSV 文件转换为 JSON 文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python convert_csv_to_json.py \\
      --input data/split_lingyin_audio_info_con_f60.csv \\
      --output data/audio_text_data.jsonl

  # 自定义输出文件名
  python convert_csv_to_json.py \\
      --input data/split_lingyin_audio_info_con_f60.csv \\
      --output data/output.jsonl
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/split_lingyin_audio_info_con_f60.csv',
        help='输入 CSV 文件路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/audio_text_data.jsonl',
        help='输出 JSON 文件路径（JSONL 格式）'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.input).exists():
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 执行转换
    convert_csv_to_json(args.input, args.output)


if __name__ == "__main__":
    main()