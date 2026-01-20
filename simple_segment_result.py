import json
import sys
import argparse

def extract_results(results_file, test_file, output_file=None):
    """从results.json中提取原始文本和预测分段结果"""
    
    # 读取results.json
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 读取测试数据以获取原始文本
    test_data = {}
    if test_file:
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    test_data[item['id']] = item
    
    # 构建输出数据
    output_data = {
        "total_samples": results.get("total_samples", 0),
        "samples": []
    }
    
    # 处理每个样本
    for sample in results.get("samples", []):
        sample_id = sample.get("id", "")
        
        # 获取原始文本（保持连续字符串，由查看器自动换行）
        original_text = ""
        if test_file and sample_id in test_data:
            original_text = test_data[sample_id].get("text", "")
        else:
            # 如果没有测试文件，尝试从predicted_paragraphs重建
            paragraphs = sample.get("predicted_paragraphs", [])
            if paragraphs:
                sentences = []
                for para in paragraphs:
                    para_text = para.get("text", "")
                    if para_text:
                        sentences.append(para_text)
                original_text = " ".join(sentences)
        
        # 提取预测分段结果（保持连续字符串，由查看器自动换行）
        predicted_paragraphs = []
        for para in sample.get("predicted_paragraphs", []):
            para_text = para.get("text", "")
            if para_text:
                predicted_paragraphs.append({
                    "text": para_text
                })
        
        output_data["samples"].append({
            "id": sample_id,
            "original_text": original_text,
            "predicted_paragraphs": predicted_paragraphs
        })
    
    # 输出JSON
    output_json = json.dumps(output_data, ensure_ascii=False, indent=2)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_json)
        print(f"结果已保存到: {output_file}", file=sys.stderr)
    else:
        print(output_json)

def main():
    parser = argparse.ArgumentParser(description='从results.json中提取原始文本和预测分段结果')
    parser.add_argument('--results_file', type=str, default='results.json',
                       help='输入的results.json文件路径')
    parser.add_argument('--test_file', type=str, default='data/10piece_test.jsonl',
                       help='测试数据文件路径（用于获取原始文本）')
    parser.add_argument('--output', type=str, default='demo_results.json',
                       help='输出JSON文件路径（默认：demo_results.json）')
    
    args = parser.parse_args()
    
    extract_results(args.results_file, args.test_file, args.output)

if __name__ == "__main__":
    main()
