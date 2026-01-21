#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建段落级训练数据集
从现有的短文本数据合并为长文本，使用 LLM 自动标注段落分割点
"""

import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from openai import OpenAI
from data_processor import ContinuousTextProcessor
from dotenv import load_dotenv
load_dotenv()

class ParagraphDatasetCreator:
    """段落级数据集创建器"""
    
    def __init__(self, api_key: str = None):
        self.processor = ContinuousTextProcessor()
        
        # 初始化 LLM 客户端
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
        
        self.api_key = api_key
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # 创建主客户端（用于单线程场景）
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )
        
        # 用于线程安全的锁和计数器
        self.lock = Lock()
        self.processed_count = 0
        
        print("✓ 初始化完成")
    
    def _create_client(self):
        """创建新的客户端实例（用于多线程）"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
    def segment_text_with_llm(self, text: str, client: OpenAI = None) -> str:
        """
        使用 LLM 进行自然段落划分
        
        Args:
            text: 输入文本（连续文本，不带空格）
            client: OpenAI 客户端实例（可选，用于多线程场景）
            
        Returns:
            分割后的文本（用 <PARAGRAPH_BREAK/> 分隔）
        """
        if client is None:
            client = self.client
            
        prompt = f"""## 角色：
你是一个文本段落划分专家，能够根据语义将长文本划分为自然段落。

## 目标：
请将以下长文本划分为若干个自然段落。每个自然段应该是一个完整的语义单元，讨论一个主题或观点。

## 要求：
1. 保持原文内容不变，不遗漏、不添加任何内容
2. 每个自然段应该包含多个句子，形成完整的语义单元
3. 段落之间用 <PARAGRAPH_BREAK/> 标记分隔
4. 每个段落长度建议在 200-500 个字符之间
5. 段落划分应该符合自然的阅读习惯和语义逻辑

## 示例：

输入文本：
这种智能体之间的协作包括协作过程中对智能体权限数据安全等的控制。这些是我们的设计理念我们认为也是在引领中国整个AI发展的趋势。我们不是在做单独的一个点而是在做一个超级智能体结合超级群使得智能体的未来和人工同时在一个项目里面完成协作共同推进。在一些场景上面完成人工的替代这是我们的理念和发展趋势。

输出文本：
这种智能体之间的协作包括协作过程中对智能体权限数据安全等的控制。这些是我们的设计理念我们认为也是在引领中国整个AI发展的趋势。<PARAGRAPH_BREAK/>我们不是在做单独的一个点而是在做一个超级智能体结合超级群使得智能体的未来和人工同时在一个项目里面完成协作共同推进。<PARAGRAPH_BREAK/>在一些场景上面完成人工的替代这是我们的理念和发展趋势。

## 正式任务

输入文本：{text}

输出文本："""
        
        try:
            completion = client.chat.completions.create(
                model="qwen-flash",
                messages=[
                    {"role": "system", "content": "你是段落划分专家。"},
                    {"role": "user", "content": prompt},
                ],
            )
            result = completion.choices[0].message.content.strip()
            return result
        except Exception as e:
            print(f"  LLM API调用错误: {e}")
            return ""
    
    def parse_paragraphs(self, segmented_text: str) -> List[str]:
        """
        解析 LLM 返回的段落
        
        Args:
            segmented_text: 包含 <PARAGRAPH_BREAK/> 的文本
            
        Returns:
            段落列表
        """
        if not segmented_text:
            return []
        
        paragraphs = segmented_text.split('<PARAGRAPH_BREAK/>')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def create_training_sample(self, text: str, sample_id: str, client: OpenAI = None) -> Optional[Dict]:
        """
        从长文本创建训练样本
        
        Args:
            text: 连续文本（不带空格）
            sample_id: 样本ID
            client: OpenAI 客户端实例（可选，用于多线程场景）
            
        Returns:
            训练样本字典或 None
        """
        # 检查长度
        if len(text) < 500:
            print(f"  [{sample_id}] 文本太短 ({len(text)} 字符)，跳过")
            return None
        
        if len(text) > 5000:
            print(f"  [{sample_id}] 文本较长 ({len(text)} 字符)，截断到 5000")
            text = text[:5000]
        
        print(f"  [{sample_id}] 文本长度: {len(text)} 字符")
        
        # LLM 标注段落
        print(f"  [{sample_id}] 调用 LLM 标注段落...")
        segmented_text = self.segment_text_with_llm(text, client)
        
        if not segmented_text:
            print(f"  [{sample_id}] LLM 标注失败")
            return None
        
        # 解析段落
        paragraphs = self.parse_paragraphs(segmented_text)
        
        if len(paragraphs) < 2:
            print(f"  [{sample_id}] 段落太少 ({len(paragraphs)})，跳过")
            return None
        
        print(f"  [{sample_id}] 划分为 {len(paragraphs)} 个段落")
        
        # 合并所有段落为完整文本
        full_text = "".join(paragraphs)
        
        # 转换为句子级空格分隔格式（一句话一个空格）
        spaced_text = self.processor.preprocess_sentence_level_text(full_text)
        sentences = spaced_text.split()
        
        # 计算段落分割点（在句子级格式中，基于句子索引）
        segment_positions = []
        current_sentence_idx = 0
        
        for i, paragraph in enumerate(paragraphs[:-1]):  # 最后一个段落不需要标记分割点
            # 将段落转换为句子级格式
            para_sentence_text = self.processor.preprocess_sentence_level_text(paragraph)
            para_sentences = para_sentence_text.split()
            
            # 累加句子位置
            current_sentence_idx += len(para_sentences)
            segment_positions.append(current_sentence_idx - 1)  # 标记句子索引（最后一个句子后分段）
        
        return {
            "id": sample_id,
            "text": spaced_text,
            "segment_positions": segment_positions,
            "original_text": full_text,
            "num_paragraphs": len(paragraphs)
        }
    
    def create_from_existing_data(
        self,
        input_file: str,
        output_file: str,
        num_samples: int = 500,
        merge_count: int = 7,
        min_text_length: int = 288,
        max_text_length: int = 3888,
        max_workers: int = 10
    ):
        """
        从现有的短文本数据合并创建长文本数据集
        
        Args:
            input_file: 输入文件路径（现有的 train_data.jsonl）
            output_file: 输出文件路径
            num_samples: 要生成的样本数量
            merge_count: 每个样本合并的短文本数量
            min_text_length: 最小文本长度
            max_text_length: 最大文本长度
            max_workers: 并发线程数（默认: 5）
        """
        print(f"\n开始处理: {input_file}")
        print(f"目标: 生成 {num_samples} 条数据，每个样本合并 {merge_count} 条短文本")
        print(f"并发线程数: {max_workers}")
        
        # 读取现有数据
        print(f"读取现有数据...")
        existing_data = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing_data.append(json.loads(line))
        except Exception as e:
            print(f"错误：无法读取文件 {input_file}: {e}")
            return
        
        print(f"✓ 共有 {len(existing_data)} 条现有数据")
        
        if len(existing_data) < merge_count:
            print(f"错误：现有数据太少 ({len(existing_data)})，无法合并 {merge_count} 条")
            return
        
        # 准备所有任务（顺序拼接）
        tasks = []
        data_index = 0  # 当前数据索引
        
        for i in range(num_samples):
            # 顺序选择若干短文本合并
            selected = []
            for _ in range(merge_count):
                if data_index >= len(existing_data):
                    # 如果数据不够，循环使用（从头开始）
                    data_index = 0
                selected.append(existing_data[data_index])
                data_index += 1
            
            # 合并文本（去掉空格，还原为连续文本）
            merged_text = ""
            for item in selected:
                text = item['text'].replace(" ", "")
                merged_text += text
            
            # 检查合并后的长度
            if len(merged_text) < min_text_length:
                continue
            
            if len(merged_text) > max_text_length:
                merged_text = merged_text[:max_text_length]
            
            tasks.append((merged_text, f"paragraph_{i}"))
        
        print(f"✓ 准备了 {len(tasks)} 个有效任务（顺序拼接，每 {merge_count} 条合并）")
        
        # 使用线程池并发处理
        dataset = []
        failed_count = 0
        start_time = time.time()
        # 重置计数器
        self.processed_count = 0

        # 用于定期保存的临时文件
        temp_output_file = output_file.replace('.jsonl', '_temp.jsonl')
        saved_count = 0  # 已保存的数量
        
        def process_task(task_data):
            """处理单个任务的函数"""
            text, sample_id = task_data
            # 为每个线程创建独立的客户端
            client = self._create_client()
            sample = self.create_training_sample(text, sample_id, client)
            return sample, sample_id
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(process_task, task): task for task in tasks}
            
            # 处理完成的任务
            for future in as_completed(future_to_task):
                try:
                    sample, sample_id = future.result()
                    with self.lock:
                        self.processed_count += 1
                        current_count = self.processed_count
                        
                        if sample:
                            dataset.append(sample)
                            print(f"  [{sample_id}] ✓ 成功创建样本 ({current_count}/{len(tasks)})")
                        else:
                            failed_count += 1
                            print(f"  [{sample_id}] ✗ 创建失败 ({current_count}/{len(tasks)})")
                        
                        # 每100个样本输出一次进度
                        if current_count % 100 == 0:
                            elapsed = time.time() - start_time
                            avg_time = elapsed / current_count
                            remaining = (len(tasks) - current_count) * avg_time
                            print(f"\n进度: {current_count}/{len(tasks)}, 成功: {len(dataset)}, 失败: {failed_count}")
                            print(f"已用时间: {elapsed:.1f}秒, 预计剩余: {remaining:.1f}秒")

                            # 保存新增的数据到临时文件（追加模式）
                            if len(dataset) > saved_count:
                                new_samples = dataset[saved_count:]
                                with open(temp_output_file, 'a', encoding='utf-8') as f:
                                    for item in new_samples:
                                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                                saved_count = len(dataset)
                                print(f"  已保存 {saved_count} 条数据到临时文件")
                            
                except Exception as e:
                    with self.lock:
                        failed_count += 1
                        self.processed_count += 1
                        print(f"  处理任务时出错: {e}")
        
        total_time = time.time() - start_time
        print(f"\n✓ 所有任务完成！总耗时: {total_time:.1f}秒")

        # 保存剩余的数据（如果有）
        if len(dataset) > saved_count:
            new_samples = dataset[saved_count:]
            with open(temp_output_file, 'a', encoding='utf-8') as f:
                for item in new_samples:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            saved_count = len(dataset)
        
        # 从临时文件读取所有数据，进行最终处理和保存
        if os.path.exists(temp_output_file):
            with open(temp_output_file, 'r', encoding='utf-8') as f:
                dataset = [json.loads(line) for line in f if line.strip()]
            os.remove(temp_output_file)  # 删除临时文件
        
        
        # 保存数据集
        self._save_dataset(dataset, output_file)
    
    def _save_dataset(self, dataset: List[Dict], output_file: str):
        """保存数据集"""
        if len(dataset) == 0:
            print("\n错误：没有成功处理的数据")
            return
        
        print(f"\n数据处理完成！共生成 {len(dataset)} 条数据")
        
        # 随机打乱
        random.shuffle(dataset)
        
        # 划分训练集和测试集 (99:1)
        train_size = int(len(dataset) * 0.99)
        train_data = dataset[:train_size]
        test_data = dataset[train_size:]
        
        print(f"数据集划分 - 训练集: {len(train_data)}, 测试集: {len(test_data)}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存训练集
        train_path = output_file.replace('.jsonl', '_train.jsonl')
        with open(train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 保存测试集
        test_path = output_file.replace('.jsonl', '_test.jsonl')
        with open(test_path, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n数据集已保存:")
        print(f"  训练集: {train_path}")
        print(f"  测试集: {test_path}")
        
        # 统计信息
        avg_length = sum(len(s['text'].split()) for s in dataset) / len(dataset)
        avg_paragraphs = sum(s['num_paragraphs'] for s in dataset) / len(dataset)
        avg_segments = sum(len(s['segment_positions']) for s in dataset) / len(dataset)
        
        print(f"\n数据集统计:")
        print(f"  平均文本长度: {avg_length:.0f} 个单元")
        print(f"  平均段落数: {avg_paragraphs:.1f} 个")
        print(f"  平均分割点数: {avg_segments:.1f} 个")
        
        # 显示几个示例
        print(f"\n示例数据（前3条）:")
        for i, sample in enumerate(dataset[:3]):
            print(f"\n  示例 {i+1}:")
            print(f"    ID: {sample['id']}")
            print(f"    段落数: {sample['num_paragraphs']}")
            print(f"    分割点: {sample['segment_positions']}")
            print(f"    原文前100字符: {sample['original_text'][:100]}...")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='从现有数据创建段落级训练数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python create_paragraph_dataset.py --input data/audio_text_data2_merged.jsonl --output data/paragraph_data1.jsonl --num_samples 10

  # 自定义参数
  python create_paragraph_dataset.py \\
      --input data/train_data.jsonl \\
      --output data/paragraph_data.jsonl \\
      --num_samples 1000 \\
      --merge_count 15 \\
      --min_length 800 \\
      --max_length 4000
        """
    )
    
    parser.add_argument('--input', type=str,
                       default='data/train_data.jsonl',
                       help='输入文件路径（现有的 train_data.jsonl）')
    parser.add_argument('--output', type=str,
                       default='data/paragraph_data.jsonl',
                       help='输出文件路径（不含扩展名，会自动添加 _train.jsonl 和 _test.jsonl）')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='生成样本数量（默认: 500）')
    parser.add_argument('--merge_count', type=int, default=7,
                       help='每个样本合并的短文本数量（默认: 10）')
    parser.add_argument('--min_length', type=int, default=500,
                       help='最小文本长度（字符数，默认: 500）')
    parser.add_argument('--max_length', type=int, default=5000,
                       help='最大文本长度（字符数，默认: 5000）')
    parser.add_argument('--api_key', type=str, default=None,
                       help='LLM API key（可选，默认从环境变量读取）')
    parser.add_argument('--max_workers', type=int, default=10,
                       help='并发线程数（默认: 5，建议根据API限制调整）')
    
    args = parser.parse_args()
    
    # 设置 API key
    if args.api_key:
        os.environ["DASHSCOPE_API_KEY"] = args.api_key
    
    # 检查 API key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("错误: 请设置 DASHSCOPE_API_KEY 环境变量或使用 --api_key 参数")
        return
    
    # 创建数据集
    creator = ParagraphDatasetCreator()
    
    creator.create_from_existing_data(
        input_file=args.input,
        output_file=args.output,
        num_samples=args.num_samples,
        merge_count=args.merge_count,
        min_text_length=args.min_length,
        max_text_length=args.max_length,
        max_workers=args.max_workers
    )
    
    print("\n✓ 完成！")


if __name__ == "__main__":
    main()
