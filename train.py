import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import swanlab

# 1. 加载配置文件
def load_config(config_path="config.json"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

# 加载配置
CONFIG = load_config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 数据集类
class ParagraphSegmentationDataset(Dataset):
    """自然段划分数据集类（滑动窗口 + 上下文句子对）
    
    针对长文本的处理策略：
    1. 使用滑动窗口将长段落分成多个窗口
    2. 对每个句子对，包含前后N个句子作为上下文
    3. 动态调整窗口大小，确保不超过512 tokens
    """
    def __init__(self, jsonl_path, tokenizer, max_length=512, 
                 window_size=20, window_overlap=10, 
                 context_sentences=2, max_tokens_per_window=500):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.context_sentences = context_sentences
        self.max_tokens_per_window = max_tokens_per_window
        self.data = []
        
        # 读取JSONL文件并处理
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                text = item['text']
                segment_positions = set(item.get('segment_positions', []))
                
                # 按单个空格分割句子
                sentences = [s.strip() for s in text.split(" ") if s.strip()]
                
                if len(sentences) < 2:
                    continue
                
                # 使用滑动窗口处理长段落
                self._process_with_sliding_window(sentences, segment_positions)
    
    def _process_with_sliding_window(self, sentences, segment_positions):
        """使用滑动窗口处理句子列表"""
        total_sentences = len(sentences)
        step_size = self.window_size - self.window_overlap
        
        # 生成窗口
        start_idx = 0
        while start_idx < total_sentences - 1:
            # 确定窗口结束位置
            end_idx = min(start_idx + self.window_size, total_sentences)
            
            # 获取窗口内的句子
            window_sentences = sentences[start_idx:end_idx]
            
            # 动态调整窗口大小（如果超过token限制）
            window_sentences = self._adjust_window_size(window_sentences)
            
            # 对窗口内的每个句子对生成样本
            for i in range(len(window_sentences) - 1):
                # 计算原始句子索引
                original_idx = start_idx + i
                
                # 确保不超出原始句子列表范围
                if original_idx >= total_sentences - 1:
                    break
                
                # 获取当前句子对
                sent1 = window_sentences[i]
                sent2 = window_sentences[i + 1]
                
                # 跳过空句子
                if not sent1 or not sent2:
                    continue
                
                # 获取上下文句子
                context_before = window_sentences[max(0, i - self.context_sentences):i]
                context_after = window_sentences[i + 2:min(len(window_sentences), i + 2 + self.context_sentences)]
                
                # 标签：如果original_idx在segment_positions中，则为1（需要分段），否则为0
                label = 1 if original_idx in segment_positions else 0
                
                self.data.append({
                    'sentence1': sent1,
                    'sentence2': sent2,
                    'context_before': context_before,
                    'context_after': context_after,
                    'label': label,
                    'original_idx': original_idx
                })
            
            # 移动到下一个窗口
            start_idx += step_size
    
    def _adjust_window_size(self, window_sentences):
        """动态调整窗口大小，确保不超过token限制"""
        if not window_sentences:
            return window_sentences
        
        # 估算tokens数量（简单估算：中文字符数 * 1.5）
        total_chars = sum(len(s) for s in window_sentences)
        estimated_tokens = int(total_chars * 1.5)
        
        # 如果超过限制，缩小窗口
        if estimated_tokens > self.max_tokens_per_window:
            # 从后往前移除句子，直到满足限制
            while len(window_sentences) > 2 and estimated_tokens > self.max_tokens_per_window:
                window_sentences = window_sentences[:-1]
                total_chars = sum(len(s) for s in window_sentences)
                estimated_tokens = int(total_chars * 1.5)
        
        return window_sentences
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建包含上下文的输入
        # 格式：[前K句] [当前句1] [SEP] [当前句2] [后K句]
        context_before_text = " ".join(item['context_before']) if item['context_before'] else ""
        context_after_text = " ".join(item['context_after']) if item['context_after'] else ""
        
        # 构建第一个序列：上下文 + 句子1
        seq1_parts = []
        if context_before_text:
            seq1_parts.append(context_before_text)
        seq1_parts.append(item['sentence1'])
        seq1 = " ".join(seq1_parts)
        
        # 构建第二个序列：句子2 + 上下文
        seq2_parts = [item['sentence2']]
        if context_after_text:
            seq2_parts.append(context_after_text)
        seq2 = " ".join(seq2_parts)
        
        # 使用BERT tokenizer编码句子对（带上下文）
        # 修改：使用 truncation=True 确保总长度不超过 max_length
        encoded = self.tokenizer(
            seq1,
            seq2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,  # 改为 True，确保总长度不超过 max_length
            return_tensors='pt'
        )
        
        # 确保返回的 tensor 形状正确
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # 安全检查：如果长度不对，强制截断/填充
        if input_ids.size(0) != self.max_length:
            # 如果长度超过 max_length，截断
            if input_ids.size(0) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            # 如果长度小于 max_length，填充
            elif input_ids.size(0) < self.max_length:
                pad_length = self.max_length - input_ids.size(0)
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                input_ids = torch.cat([input_ids, torch.full((pad_length,), pad_token_id, dtype=input_ids.dtype)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=attention_mask.dtype)])
        
        return {
            'input_ids': input_ids.detach(), # 确保不带梯度轨迹
            'attention_mask': attention_mask.detach(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }
    
    def get_statistics(self):
        """返回数据集统计信息"""
        total_count = len(self.data)
        
        # 统计窗口信息
        window_info = {
            'total_samples': total_count,
            'window_size': self.window_size,
            'window_overlap': self.window_overlap,
            'context_sentences': self.context_sentences
        }
        
        # 采样检查截断情况
        sample_size = min(500, total_count)
        if total_count > sample_size:
            indices = random.sample(range(total_count), sample_size)
        else:
            indices = list(range(total_count))
        
        truncated_count = 0
        for idx in indices:
            item = self.data[idx]
            # 构建输入序列
            context_before_text = " ".join(item['context_before']) if item['context_before'] else ""
            context_after_text = " ".join(item['context_after']) if item['context_after'] else ""
            
            seq1_parts = []
            if context_before_text:
                seq1_parts.append(context_before_text)
            seq1_parts.append(item['sentence1'])
            seq1 = " ".join(seq1_parts)
            
            seq2_parts = [item['sentence2']]
            if context_after_text:
                seq2_parts.append(context_after_text)
            seq2 = " ".join(seq2_parts)
            
            # 检查长度
            encoded = self.tokenizer(
                seq1,
                seq2,
                add_special_tokens=True,
                return_length=True
            )
            # 直接计算input_ids的长度
            actual_length = len(encoded['input_ids'])
            if actual_length > self.max_length:
                truncated_count += 1
        
        truncation_rate = truncated_count / sample_size if sample_size > 0 else 0
        
        window_info.update({
            'sampled_samples': sample_size,
            'truncated_in_sample': truncated_count,
            'estimated_truncation_rate': truncation_rate
        })
        
        return window_info

def train():
    # 从配置文件读取参数
    MODEL_PATH = CONFIG["model"]["model_path"]
    DATA_PATH = CONFIG["data"]["train_file"]
    SAVE_PATH = CONFIG["training"]["save_dir"]
    BATCH_SIZE = CONFIG["training"]["batch_size"]
    EPOCHS = CONFIG["training"]["num_epochs"]
    LR = CONFIG["training"]["learning_rate"]
    MAX_LENGTH = CONFIG["data"]["max_length"]
    NUM_WORKERS = CONFIG["data"]["num_workers"]
    WEIGHT_DECAY = CONFIG["training"]["weight_decay"]
    WARMUP_RATIO = CONFIG["training"]["warmup_ratio"]
    CLASS_WEIGHTS = CONFIG["training"]["class_weights"]
    
    # 打印配置信息
    print("=" * 60)
    print("训练配置信息")
    print("=" * 60)
    print(f"模型路径: {MODEL_PATH}")
    print(f"训练数据: {DATA_PATH}")
    print(f"保存目录: {SAVE_PATH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Num workers: {NUM_WORKERS}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Warmup ratio: {WARMUP_RATIO}")
    print(f"Class weights: {CLASS_WEIGHTS}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    print()
    
    # 初始化swanlab
    swanlab.init(
        project="paragraph-segmentation",
        experiment_name=f"segment_model_{EPOCHS}epochs",
        config={
            "model_path": MODEL_PATH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "max_length": MAX_LENGTH,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "class_weights": CLASS_WEIGHTS,
            "num_workers": NUM_WORKERS,
        }
    )
    print("SwanLab initialized for training monitoring")
    print()
    
    # 创建保存目录
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    
    # 从配置读取窗口参数
    WINDOW_SIZE = CONFIG["data"].get("window_size", 20)
    WINDOW_OVERLAP = CONFIG["data"].get("window_overlap", 10)
    CONTEXT_SENTENCES = CONFIG["data"].get("context_sentences", 2)
    MAX_TOKENS_PER_WINDOW = CONFIG["data"].get("max_tokens_per_window", 500)
    
    # 更新swanlab配置（添加窗口参数）
    swanlab.config.update({
        "window_size": WINDOW_SIZE,
        "window_overlap": WINDOW_OVERLAP,
        "context_sentences": CONTEXT_SENTENCES,
        "max_tokens_per_window": MAX_TOKENS_PER_WINDOW,
    })
    
    print(f"Loading dataset from {DATA_PATH}...")
    print(f"Window parameters: size={WINDOW_SIZE}, overlap={WINDOW_OVERLAP}, context={CONTEXT_SENTENCES}")
    dataset = ParagraphSegmentationDataset(
        DATA_PATH, 
        tokenizer, 
        max_length=MAX_LENGTH,
        window_size=WINDOW_SIZE,
        window_overlap=WINDOW_OVERLAP,
        context_sentences=CONTEXT_SENTENCES,
        max_tokens_per_window=MAX_TOKENS_PER_WINDOW
    )
    print(f"Dataset size: {len(dataset)}")
    
    # 统计信息
    print("Analyzing data statistics...")
    stats = dataset.get_statistics()
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Sampled {stats['sampled_samples']} samples for truncation analysis")
    print(f"  Estimated truncation rate: {stats['estimated_truncation_rate']*100:.2f}%")
    print(f"\nWindow configuration:")
    print(f"  - Window size: {stats['window_size']} sentences")
    print(f"  - Window overlap: {stats['window_overlap']} sentences")
    print(f"  - Context sentences: {stats['context_sentences']} (before/after)")
    print(f"\nNote: BERT max_length={MAX_LENGTH} includes special tokens:")
    print(f"      - [CLS] token (1)")
    print(f"      - [SEP] tokens (2)")
    print(f"      - Actual available tokens: {MAX_LENGTH - 3}")
    print(f"      - Truncation strategy: 'only_second' (preserve first sequence)")
    print()
    
    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 3. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=CONFIG["model"]["num_labels"])
    model.to(DEVICE)
    print(f"Model loaded. Device: {DEVICE}")

    # 4. 优化器与损失函数（解决样本不均衡）
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # 计算总步数用于学习率调度器
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    # 学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 重点：给 Label 1 增加权重，因为分段点在自然段中较少
    # 实际数据分布：不分段:分段 比例约为 3.87:1
    # 使用权重 [1.0, 3.9] 来平衡类别
    weights = torch.tensor(CLASS_WEIGHTS).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    # 5. 训练循环
    model.train()
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()

            # 混合精度训练
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
            
            scheduler.step()

            total_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            global_step = epoch * len(train_loader) + step
            
            # 记录到swanlab（每步都记录）
            swanlab.log({
                "train/loss": loss.item(),
                "train/avg_loss": total_loss / (step + 1),
                "train/learning_rate": current_lr,
                "train/epoch": epoch + 1,
                "train/step": global_step
            })
            
            # 每 100 步记录一次 Loss（控制台输出）
            if step % 100 == 0:
                loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(step+1):.4f}',
                    'lr': f'{current_lr:.2e}'
                })

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # 记录epoch级别的指标到swanlab
        swanlab.log({
            "train/epoch_loss": avg_loss,
            "train/epoch": epoch + 1
        })

        # 每个 Epoch 保存一次
        epoch_save_path = os.path.join(SAVE_PATH, f"checkpoint_epoch_{epoch+1}")
        model.save_pretrained(epoch_save_path)
        tokenizer.save_pretrained(epoch_save_path)
        print(f"Epoch {epoch+1} saved to {epoch_save_path}")
    
    print("Training completed!")
    swanlab.finish()

if __name__ == "__main__":
    train()