import os
import json
import sys
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from train import load_config, CONFIG
from test import predict_sentence_pairs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_labels=2):
    """加载训练好的模型"""
    print(f"Loading model from {model_path}...", file=sys.stderr)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded. Device: {DEVICE}", file=sys.stderr)
    return model

def analyze_segmentation(model, tokenizer, test_file, config, threshold=0.7):
    """分析分段效果，返回结构化数据"""
    window_size = config["data"].get("window_size", 20)
    window_overlap = config["data"].get("window_overlap", 10)
    context_sentences = config["data"].get("context_sentences", 2)
    max_length = config["data"]["max_length"]
    max_tokens_per_window = config["data"].get("max_tokens_per_window", 500)
    
    # 读取测试数据 - 支持 JSONL 和标准 JSON 格式
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # 尝试作为标准 JSON 解析
    try:
        data = json.loads(content)
        # 如果是字典且包含 result.utterances，则转换格式（output_processed_merged.json 格式）
        if isinstance(data, dict) and 'result' in data and 'utterances' in data.get('result', {}):
            # 转换 output_processed_merged.json 格式
            utterances = data['result']['utterances']
            test_items = []
            for utt in utterances:
                if 'text' in utt:
                    test_items.append({
                        'id': f"utterance_{utt.get('sentenceId', len(test_items))}",
                        'text': utt['text'],
                        'segment_positions': []  # 如果没有提供，使用空列表
                    })
        # 如果是数组，直接使用
        elif isinstance(data, list):
            test_items = data
        # 如果是单个对象，包装成数组
        elif isinstance(data, dict):
            test_items = [data]
        else:
            test_items = []
    except json.JSONDecodeError:
        # 如果标准 JSON 解析失败，尝试作为 JSONL 解析（每行一个 JSON 对象）
        test_items = [json.loads(line.strip()) for line in content.split('\n') if line.strip()]
    
    print(f"Processing {len(test_items)} samples...", file=sys.stderr)
    
    results = {
        "total_samples": len(test_items),
        "samples": []
    }
    
    for item_idx, item in enumerate(test_items, 1):
        text = item['text']
        true_segment_positions = set(item.get('segment_positions', []))
        item_id = item.get('id', f'item_{item_idx}')
        
        # 分割句子
        sentences = [s.strip() for s in text.split(" ") if s.strip()]
        
        if len(sentences) < 2:
            results["samples"].append({
                "id": item_id,
                "index": item_idx,
                "skipped": True,
                "reason": "句子数不足"
            })
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
        
        # 确定预测的分段位置
        pred_segment_positions = set()
        pred_details = {}
        
        for pred in predictions:
            idx = pred['index']
            if pred['prediction'] == 1 and pred['confidence'] >= threshold:
                pred_segment_positions.add(idx)
                pred_details[idx] = {
                    'confidence': pred['confidence'],
                    'prob_class_0': pred['prob_class_0'],
                    'prob_class_1': pred['prob_class_1']
                }
        
        # 计算准确率
        correct = len(true_segment_positions & pred_segment_positions)
        precision = correct / len(pred_segment_positions) if pred_segment_positions else 0
        recall = correct / len(true_segment_positions) if true_segment_positions else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 构建真实分段段落
        true_paragraphs = []
        start_idx = 0
        for seg_idx in sorted(true_segment_positions):
            if seg_idx >= start_idx:
                para_sentences = sentences[start_idx:seg_idx + 1]
                true_paragraphs.append({
                    "start_sentence": start_idx,
                    "end_sentence": seg_idx,
                    "sentence_count": len(para_sentences),
                    "text": " ".join(para_sentences)
                })
                start_idx = seg_idx + 1
        if start_idx < len(sentences):
            para_sentences = sentences[start_idx:]
            true_paragraphs.append({
                "start_sentence": start_idx,
                "end_sentence": len(sentences) - 1,
                "sentence_count": len(para_sentences),
                "text": " ".join(para_sentences)
            })
        
        # 构建预测分段段落（包含置信度信息）
        pred_paragraphs = []
        start_idx = 0
        for seg_idx in sorted(pred_segment_positions):
            if seg_idx >= start_idx:
                para_sentences = sentences[start_idx:seg_idx + 1]
                para_info = {
                    "start_sentence": start_idx,
                    "end_sentence": seg_idx,
                    "sentence_count": len(para_sentences),
                    "text": " ".join(para_sentences)
                }
                # 添加分段位置的置信度信息（seg_idx是这个段落的结束位置，也是分段位置）
                para_info["position"] = seg_idx
                para_info["is_true_segment"] = seg_idx in true_segment_positions
                para_info["is_predicted_segment"] = True
                if seg_idx in true_segment_positions:
                    para_info["status"] = "correct"
                else:
                    para_info["status"] = "false_positive"
                para_info["prediction_details"] = {
                    "confidence": pred_details[seg_idx]['confidence'],
                    "prob_class_0": pred_details[seg_idx]['prob_class_0'],
                    "prob_class_1": pred_details[seg_idx]['prob_class_1']
                }
                pred_paragraphs.append(para_info)
                start_idx = seg_idx + 1
        # 最后一段（没有分段位置，不需要添加置信度信息）
        if start_idx < len(sentences):
            para_sentences = sentences[start_idx:]
            pred_paragraphs.append({
                "start_sentence": start_idx,
                "end_sentence": len(sentences) - 1,
                "sentence_count": len(para_sentences),
                "text": " ".join(para_sentences)
            })
        
        # 构建样本结果
        sample_result = {
            "id": item_id,
            "index": item_idx,
            "total_sentences": len(sentences),
            "true_segment_positions": sorted(list(true_segment_positions)),
            "predicted_segment_positions": sorted(list(pred_segment_positions)),
            "true_paragraph_count": len(true_segment_positions) + 1,
            "predicted_paragraph_count": len(pred_segment_positions) + 1,
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "correct_segments": correct,
                "total_true_segments": len(true_segment_positions),
                "total_predicted_segments": len(pred_segment_positions)
            },
            "true_paragraphs": true_paragraphs,
            "predicted_paragraphs": pred_paragraphs
        }
        
        results["samples"].append(sample_result)
    
    # 计算总体指标
    if results["samples"]:
        valid_samples = [s for s in results["samples"] if not s.get("skipped", False)]
        if valid_samples:
            total_precision = sum(s["metrics"]["precision"] for s in valid_samples) / len(valid_samples)
            total_recall = sum(s["metrics"]["recall"] for s in valid_samples) / len(valid_samples)
            total_f1 = sum(s["metrics"]["f1_score"] for s in valid_samples) / len(valid_samples)
            
            results["overall_metrics"] = {
                "average_precision": total_precision,
                "average_recall": total_recall,
                "average_f1_score": total_f1,
                "valid_samples": len(valid_samples)
            }
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='分析模型分段效果并输出JSON')
    parser.add_argument('--model_path', type=str, default='./output_model/checkpoint_epoch_3',
                       help='模型路径')
    parser.add_argument('--test_file', type=str, default='data/10piece_test.jsonl',
                       help='测试文件路径')
    parser.add_argument('--threshold', type=float, default=0.9,
                       help='分段阈值')
    parser.add_argument('--output', type=str, default='results.json',
                       help='输出JSON文件路径（默认：results.json）')

    
    args = parser.parse_args()
    
    # 加载配置
    config = CONFIG
    
    # 加载模型和tokenizer
    model = load_model(args.model_path, num_labels=config["model"]["num_labels"])
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    
    # 分析分段效果
    results = analyze_segmentation(model, tokenizer, args.test_file, config, args.threshold)
    
    # 输出JSON
    output_json = json.dumps(results, ensure_ascii=False, indent=2)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_json)
        print(f"结果已保存到: {args.output}", file=sys.stderr)
    else:
        print(output_json)

if __name__ == "__main__":
    main()