import json
import re

def is_sentence_end(text):
    """
    判断文本是否以句子结尾标点结束
    
    Args:
        text: 文本字符串
    
    Returns:
        bool: 如果以句子结尾标点结束返回True，否则返回False
    """
    # 中文句子结尾标点：。！？；等
    sentence_end_punctuation = ['。', '！', '？', '；', '.', '!', '?', ';']
    text = text.strip()
    if not text:
        return False
    return text[-1] in sentence_end_punctuation

def merge_consecutive_speakers(input_file, output_file):
    """
    将相同说话人且连续的文本合并到一个text中
    在两句话之间添加空格
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    utterances = data['result']['utterances']
    merged_utterances = []
    
    if not utterances:
        # 如果没有utterances，直接保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return
    
    # 初始化第一个utterance
    current_utterance = {
        'sentenceId': utterances[0]['sentenceId'],
        'start_time': utterances[0]['start_time'],
        'end_time': utterances[0]['end_time'],
        'text': utterances[0]['text'],
        'additions': {
            'speaker': utterances[0]['additions']['speaker']
        }
    }
    
    # 从第二个utterance开始遍历
    for i in range(1, len(utterances)):
        current_speaker = utterances[i]['additions']['speaker']
        prev_speaker = current_utterance['additions']['speaker']
        
        # 如果说话人相同，合并文本
        if current_speaker == prev_speaker:
            # 更新结束时间
            current_utterance['end_time'] = utterances[i]['end_time']
            # 判断前一个文本是否以句子结尾标点结束
            if is_sentence_end(current_utterance['text']):
                # 如果前一个文本以句子结尾，则在两句话之间加空格
                current_utterance['text'] += ' ' + utterances[i]['text']
            else:
                # 如果前一个文本不以句子结尾，直接拼接
                current_utterance['text'] += utterances[i]['text']
        else:
            # 说话人不同，保存当前的utterance，开始新的
            merged_utterances.append(current_utterance)
            current_utterance = {
                'sentenceId': utterances[i]['sentenceId'],
                'start_time': utterances[i]['start_time'],
                'end_time': utterances[i]['end_time'],
                'text': utterances[i]['text'],
                'additions': {
                    'speaker': utterances[i]['additions']['speaker']
                }
            }
    
    # 添加最后一个utterance
    merged_utterances.append(current_utterance)
    
    # 更新sentenceId为连续编号
    for idx, utterance in enumerate(merged_utterances):
        utterance['sentenceId'] = idx
    
    # 更新数据
    data['result']['utterances'] = merged_utterances
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"合并完成！")
    print(f"原始utterances数量: {len(utterances)}")
    print(f"合并后utterances数量: {len(merged_utterances)}")


if __name__ == '__main__':
    input_file = 'data/asr_data\output2_新版_processed.json'
    output_file = 'data/asr_data/output2_processed_merged.json'
    
    merge_consecutive_speakers(input_file, output_file)