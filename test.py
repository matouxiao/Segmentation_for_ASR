import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import sys

# 导入训练脚本中的数据集类
from train import ParagraphSegmentationDataset, load_config, CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_labels=2):
    """加载训练好的模型"""
    print(f"Loading model from {model_path}...")
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded. Device: {DEVICE}")
    return model

def predict_sentence_pairs(model, tokenizer, sentences, window_size=20, window_overlap=10, 
                          context_sentences=2, max_length=512, max_tokens_per_window=500):
    """对句子列表进行预测，返回每个句子对的分段预测"""
    if len(sentences) < 2:
        return []
    
    predictions = []
    step_size = window_size - window_overlap
    total_sentences = len(sentences)
    
    # 使用滑动窗口处理
    start_idx = 0
    while start_idx < total_sentences - 1:
        end_idx = min(start_idx + window_size, total_sentences)
        window_sentences = sentences[start_idx:end_idx]
        
        # 动态调整窗口大小
        total_chars = sum(len(s) for s in window_sentences)
        estimated_tokens = int(total_chars * 1.5)
        if estimated_tokens > max_tokens_per_window:
            while len(window_sentences) > 2 and estimated_tokens > max_tokens_per_window:
                window_sentences = window_sentences[:-1]
                total_chars = sum(len(s) for s in window_sentences)
                estimated_tokens = int(total_chars * 1.5)
        
        # 对窗口内的每个句子对进行预测
        for i in range(len(window_sentences) - 1):
            original_idx = start_idx + i
            if original_idx >= total_sentences - 1:
                break
            
            sent1 = window_sentences[i]
            sent2 = window_sentences[i + 1]
            
            if not sent1 or not sent2:
                continue
            
            # 获取上下文
            context_before = window_sentences[max(0, i - context_sentences):i]
            context_after = window_sentences[i + 2:min(len(window_sentences), i + 2 + context_sentences)]
            
            # 构建输入
            context_before_text = " ".join(context_before) if context_before else ""
            context_after_text = " ".join(context_after) if context_after else ""
            
            seq1_parts = []
            if context_before_text:
                seq1_parts.append(context_before_text)
            seq1_parts.append(sent1)
            seq1 = " ".join(seq1_parts)
            
            seq2_parts = [sent2]
            if context_after_text:
                seq2_parts.append(context_after_text)
            seq2 = " ".join(seq2_parts)
            
            # 编码和预测
            encoded = tokenizer(
                seq1,
                seq2,
                max_length=max_length,
                padding='max_length',
                truncation='only_second',
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(DEVICE)
            attention_mask = encoded['attention_mask'].to(DEVICE)
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(logits, dim=-1).cpu().item()
                confidence = probs[0][pred].item()
            
            predictions.append({
                'index': original_idx,
                'prediction': pred,
                'confidence': confidence,
                'prob_class_0': probs[0][0].item(),
                'prob_class_1': probs[0][1].item()
            })
        
        start_idx += step_size
    
    return predictions

def evaluate_model(model, tokenizer, test_file, config):
    """评估模型性能"""
    print("=" * 60)
    print("模型评估")
    print("=" * 60)
    
    # 读取测试数据
    all_true_labels = []
    all_pred_labels = []
    all_predictions = []
    
    window_size = config["data"].get("window_size", 20)
    window_overlap = config["data"].get("window_overlap", 10)
    context_sentences = config["data"].get("context_sentences", 2)
    max_length = config["data"]["max_length"]
    max_tokens_per_window = config["data"].get("max_tokens_per_window", 500)
    
    print(f"Loading test data from {test_file}...")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_items = [json.loads(line.strip()) for line in f]
    
    print(f"Total test samples: {len(test_items)}")
    print()
    
    # 对每个测试样本进行预测
    for item_idx, item in enumerate(tqdm(test_items, desc="Evaluating")):
        text = item['text']
        segment_positions = set(item.get('segment_positions', []))
        sentences = [s.strip() for s in text.split(" ") if s.strip()]
        
        if len(sentences) < 2:
            continue
        
        # 获取预测结果
        predictions = predict_sentence_pairs(
            model, tokenizer, sentences,
            window_size=window_size,
            window_overlap=window_overlap,
            context_sentences=context_sentences,
            max_length=max_length,
            max_tokens_per_window=max_tokens_per_window
        )
        
        # 构建真实标签和预测标签
        for pred in predictions:
            idx = pred['index']
            true_label = 1 if idx in segment_positions else 0
            pred_label = pred['prediction']
            
            all_true_labels.append(true_label)
            all_pred_labels.append(pred_label)
            all_predictions.append({
                'item_id': item.get('sentenceId', f'item_{item_idx}'),
                'index': idx,
                'true_label': true_label,
                'pred_label': pred_label,
                'confidence': pred['confidence']
            })
    
    # 计算评估指标
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels, all_pred_labels, average='binary', zero_division=0
    )
    
    # 计算每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_true_labels, all_pred_labels, average=None, zero_division=0
    )
    
    # 混淆矩阵
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"总样本数: {len(all_true_labels)}")
    print(f"\n整体指标:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    
    print(f"\n各类别指标:")
    print(f"  类别 0 (不分段):")
    print(f"    Precision: {precision_per_class[0]:.4f}")
    print(f"    Recall:    {recall_per_class[0]:.4f}")
    print(f"    F1-score:  {f1_per_class[0]:.4f}")
    print(f"  类别 1 (分段):")
    print(f"    Precision: {precision_per_class[1]:.4f}")
    print(f"    Recall:    {recall_per_class[1]:.4f}")
    print(f"    F1-score:  {f1_per_class[1]:.4f}")
    
    print(f"\n混淆矩阵:")
    print(f"              预测")
    print(f"           0      1")
    print(f"  实际 0  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       1  {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    # 统计标签分布
    true_label_0 = sum(1 for l in all_true_labels if l == 0)
    true_label_1 = sum(1 for l in all_true_labels if l == 1)
    print(f"\n标签分布:")
    print(f"  真实标签 0 (不分段): {true_label_0} ({true_label_0/len(all_true_labels)*100:.2f}%)")
    print(f"  真实标签 1 (分段):   {true_label_1} ({true_label_1/len(all_true_labels)*100:.2f}%)")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'predictions': all_predictions
    }

def segment_text(model, tokenizer, text, config, threshold=0.9):
    """对输入文本进行自然段划分"""
    sentences = [s.strip() for s in text.split(" ") if s.strip()]
    
    if len(sentences) < 2:
        return [text]
    
    window_size = config["data"].get("window_size", 20)
    window_overlap = config["data"].get("window_overlap", 10)
    context_sentences = config["data"].get("context_sentences", 2)
    max_length = config["data"]["max_length"]
    max_tokens_per_window = config["data"].get("max_tokens_per_window", 500)
    
    predictions = predict_sentence_pairs(
        model, tokenizer, sentences,
        window_size=window_size,
        window_overlap=window_overlap,
        context_sentences=context_sentences,
        max_length=max_length,
        max_tokens_per_window=max_tokens_per_window
    )
    
    # 确定分段位置
    segment_positions = []
    for pred in predictions:
        if pred['prediction'] == 1 and pred['confidence'] >= threshold:
            segment_positions.append(pred['index'])
    
    # 根据分段位置划分段落
    paragraphs = []
    start_idx = 0
    
    for seg_idx in sorted(segment_positions):
        if seg_idx >= start_idx:
            paragraph_sentences = sentences[start_idx:seg_idx + 1]
            paragraphs.append(" ".join(paragraph_sentences))
            start_idx = seg_idx + 1
    
    # 添加最后一段
    if start_idx < len(sentences):
        paragraph_sentences = sentences[start_idx:]
        paragraphs.append(" ".join(paragraph_sentences))
    
    return paragraphs

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='测试自然段划分模型')
    parser.add_argument('--model_path', type=str, default='./output_model/checkpoint_epoch_3',
                       help='模型路径')
    parser.add_argument('--test_file', type=str, default=None,
                       help='测试文件路径（默认使用config.json中的test_file）')
    parser.add_argument('--mode', type=str, default='evaluate', choices=['evaluate', 'segment'],
                       help='模式：evaluate（评估）或segment（分段）')
    parser.add_argument('--text', type=str, default=None,
                       help='要分段的文本（segment模式）')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='分段阈值（segment模式）')
    
    args = parser.parse_args()
    
    # 加载配置
    config = CONFIG
    
    # 加载模型和tokenizer
    model = load_model(args.model_path, num_labels=config["model"]["num_labels"])
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    
    if args.mode == 'evaluate':
        # 评估模式
        test_file = args.test_file or config["data"]["test_file"]
        if not test_file:
            print("错误：未指定测试文件，请在config.json中设置test_file或使用--test_file参数")
            return
        
        results = evaluate_model(model, tokenizer, test_file, config)
        
        # 保存结果
        output_file = 'test_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': {
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1': results['f1'],
                    'precision_per_class': results['precision_per_class'],
                    'recall_per_class': results['recall_per_class'],
                    'f1_per_class': results['f1_per_class'],
                    'confusion_matrix': results['confusion_matrix']
                }
            }, f, ensure_ascii=False, indent=2)
        print(f"\n评估结果已保存到: {output_file}")
        
    elif args.mode == 'segment':
        # 分段模式
        if args.text:
            text = args.text
        else:
            print("错误：segment模式需要提供--text参数")
            return
        
        paragraphs = segment_text(model, tokenizer, text, config, threshold=args.threshold)
        
        print("\n" + "=" * 60)
        print("分段结果")
        print("=" * 60)
        print(f"输入文本长度: {len(text)} 字符")
        print(f"分段数量: {len(paragraphs)}")
        print("\n分段内容:")
        for i, para in enumerate(paragraphs, 1):
            print(f"\n段落 {i}:")
            print(para)

if __name__ == "__main__":
    main()