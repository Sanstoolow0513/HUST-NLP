import os
import sys
import torch
import torch.nn as nn
# import torch.optim as optim # optim 已被下面的导入覆盖
# from torch.utils.data import Dataset, DataLoader # 已被下面的导入覆盖
import pickle
import numpy as np
from tqdm import tqdm
import time
# 修改 AdamW 的导入来源
# from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import AdamW # 使用 PyTorch 的 AdamW
from transformers import get_linear_schedule_with_warmup # 保持 scheduler 的导入
from torch.utils.data import Dataset, DataLoader # 重新整理导入顺序
from torch.nn.utils.rnn import pad_sequence # 导入 pad_sequence

# 添加父目录到路径，以便导入model.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 不再直接从data_u.py导入，而是从pickle文件加载
# from data.data_u import id2tag, tag2id, word2id, id2word
from model import BertSegmenter, BertTokenizerForSegmentation

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 定义数据集
class BertSegmentationDataset(Dataset):
    # 使用 tokenizer 的特殊 token ID
    def __init__(self, x_data, y_data, tokenizer, id2word, max_len=512):
        self.x_data = x_data
        self.y_data = y_data
        self.tokenizer = tokenizer # BertTokenizerForSegmentation 实例
        self.max_len = max_len
        self.id2word = id2word
        self.cls_token_id = tokenizer.tokenizer.cls_token_id
        self.sep_token_id = tokenizer.tokenizer.sep_token_id
        self.pad_token_id = tokenizer.tokenizer.pad_token_id

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx] # 原始标签序列

        # 将ID序列转换为文本
        try:
            text = ''.join([self.id2word[id] for id in x])
        except IndexError as e:
            print(f"错误：在处理索引 {idx} 的样本时出现索引错误: {e}")
            print(f"x: {x}")
            # 返回一个空/无效样本，由collate_fn处理或过滤
            return None # 让 collate_fn 忽略

        # 使用BERT tokenizer处理文本，获取 input_ids, attention_mask, 和 offset_mapping
        # 注意：这里不再进行 padding，由 collate_fn 处理
        encoding = self.tokenizer.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            return_offsets_mapping=True,
            # padding=False # 不在这里 padding
        )

        input_ids = encoding['input_ids']
        offset_mapping = encoding['offset_mapping']

        # 创建标签序列，与 input_ids 对齐
        labels = [-100] * len(input_ids) # 初始化为忽略标签

        label_idx = 0
        for i, offset in enumerate(offset_mapping):
            # 跳过特殊标记 [CLS], [SEP], [PAD] 对应的 offset (0, 0)
            if offset == (0, 0):
                continue

            # 如果是第一个 subword token (offset[0] == 0 for the first char of the word)
            # 并且原始标签序列还有未分配的标签
            # 检查 offset[0] 是否是当前 label_idx 指向的字符的开始
            # （更简单的方法：只要不是 (0,0) 且 label_idx 没越界，就认为是有效 token 的开始）
            if label_idx < len(y):
                 # 将原始标签 y[label_idx] 赋给当前 token
                 labels[i] = y[label_idx]
                 # 移动到下一个原始标签，只有当 token 覆盖了新的字符时才移动
                 # offset[1] > offset_mapping[i-1][1] if i > 0 else True
                 # 简化：我们假设每个非特殊token至少对应一个字符的开始，
                 # 并且原始y的长度与字符数一致
                 # 当一个token的offset[1]（结束位置）大于上一个token的offset[1]时，
                 # 意味着它覆盖了新的字符范围。
                 # 或者更简单：只要offset不是(0,0)，就认为它对应一个字符（或字符的开始部分）
                 # 并且移动label_idx。这对于BMES标签是合理的。
                 # 检查当前 token 是否是新词的开始（或者单字词）
                 # offset[0] == 0 or (i > 0 and offset[0] > offset_mapping[i-1][1])
                 # 修正：对于序列标注，通常将标签赋给每个词的第一个token
                 # 我们需要知道哪个token对应哪个原始字符索引
                 # offset_mapping 提供了字符级别的映射
                 # 如果当前 token 的起始字符索引 `offset[0]` 是新的（与上一个 token 不同），
                 # 并且我们还没有用完标签，则分配标签并增加 label_idx
                 is_new_char_token = (i > 0 and offset[0] > offset_mapping[i-1][1]) or (i == 1 and offset != (0,0)) # i==1 是第一个真实 token
                 if is_new_char_token and label_idx < len(y):
                     labels[i] = y[label_idx]
                     label_idx += 1
                 elif offset != (0,0) and label_idx < len(y): # 处理subword，继承前一个token的标签？通常设为-100
                     # 对于非首个subword，也设置为-100，让模型只预测首个subword
                     labels[i] = -100
                 else:
                     # 其他情况（如超出原始标签长度的token）保持-100
                     pass


        # 返回 Tensor，collate_fn 会处理 padding
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            # attention_mask 将在 collate_fn 中创建
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# 定义数据整理函数 (collate_fn)
def create_bert_collate_fn(pad_id, label_pad_id=-100):
    def collate_fn(batch):
        # 过滤掉 None 的样本 (加载数据或 __getitem__ 出错时可能返回 None)
        batch = [item for item in batch if item is not None]
        if not batch:
            # 如果整个批次都是无效的，返回一个空字典或引发错误
            return {'input_ids': torch.empty(0, 0, dtype=torch.long),
                    'attention_mask': torch.empty(0, 0, dtype=torch.long),
                    'labels': torch.empty(0, 0, dtype=torch.long)}

        # 从 batch 中分离 input_ids 和 labels
        batch_input_ids = [item['input_ids'] for item in batch]
        batch_labels = [item['labels'] for item in batch]

        # 对 input_ids 和 labels 进行 padding
        padded_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_id)
        padded_labels = pad_sequence(batch_labels, batch_first=True, padding_value=label_pad_id)

        # 创建 attention mask (1 表示有效 token, 0 表示 padding)
        attention_mask = (padded_input_ids != pad_id).long()

        # BERT 不需要 token_type_ids 对于单句任务，但如果模型需要，可以全零传入
        token_type_ids = torch.zeros_like(padded_input_ids)

        return {
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids, # 添加 token_type_ids
            'labels': padded_labels
        }
    return collate_fn


# 定义训练函数
def train(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        # 跳过空的 batch
        if batch['input_ids'].numel() == 0:
            continue
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device) # 获取 token_type_ids
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids # 传递 token_type_ids
        )
        # 计算损失
        # outputs: [batch, seq_len, num_classes]
        # labels: [batch, seq_len]
        loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        # 可以添加梯度裁剪
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader) if len(train_loader) > 0 else 0.0


# 定义验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
             # 跳过空的 batch
            if batch['input_ids'].numel() == 0:
                continue
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) # 获取 token_type_ids
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids # 传递 token_type_ids
            )
            # 计算损失
            outputs_flat = outputs.view(-1, outputs.shape[-1])
            labels_flat = labels.view(-1)

            loss = criterion(outputs_flat, labels_flat)
            total_loss += loss.item()

            # 获取预测结果
            preds = torch.argmax(outputs, dim=2)

            # 收集预测和标签（忽略填充部分 -100）
            for i in range(labels.size(0)):
                # 使用 attention_mask 来确定有效长度可能更准确
                # 或者直接比较 labels != -100
                valid_indices = labels[i] != -100
                valid_preds = preds[i][valid_indices].cpu().tolist()
                valid_labels = labels[i][valid_indices].cpu().tolist()
                all_preds.extend(valid_preds)
                all_labels.extend(valid_labels)

    # 计算准确率
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) if all_labels else 0.0

    return total_loss / len(val_loader) if len(val_loader) > 0 else 0.0, accuracy


def main():
    set_seed(42)

    # --- 参数设置 ---
    batch_size = 16
    epochs = 5
    learning_rate = 2e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = 128 # BERT 最大长度，与 Dataset 保持一致或更大
    data_path = "../data/datasave.pkl" # 确认路径正确
    model_save_path = "bert_segmenter_best.pth" # 模型保存路径

    # --- 加载数据 ---
    print(f"加载数据: {data_path}")
    try:
        with open(data_path, 'rb') as f:
            # 根据 data_u.py 的保存结构加载
            processed_data = pickle.load(f)
        word2id = processed_data['word2id']
        id2word = processed_data['id2word']
        tag2id = processed_data['tag2id']
        id2tag = processed_data['id2tag']
        x_train = processed_data['x_train']
        y_train = processed_data['y_train']
        x_test = processed_data['x_test']
        y_test = processed_data['y_test']
        # pad_id = processed_data.get('pad_id', 0) # 从数据获取 pad_id
        # label_pad_id = -100
    except FileNotFoundError:
        print(f"错误：无法找到数据文件 {data_path}。请确保路径正确并已运行数据处理脚本。")
        sys.exit(1)
    except Exception as e:
        print(f"加载数据文件 {data_path} 时出错: {e}")
        sys.exit(1)

    print(f"数据加载成功。训练集: {len(x_train)}, 测试集: {len(x_test)}")
    print(f"词表大小: {len(word2id)}, 标签数: {len(tag2id)}")

    # --- 初始化 tokenizer ---
    tokenizer = BertTokenizerForSegmentation()
    pad_id = tokenizer.tokenizer.pad_token_id # 使用 BERT 的 pad token id
    label_pad_id = -100 # 标签的 padding value

    # --- 创建 Dataset 和 DataLoader ---
    # 注意传递 max_len
    train_dataset = BertSegmentationDataset(x_train, y_train, tokenizer, id2word, max_len=max_len)
    val_dataset = BertSegmentationDataset(x_test, y_test, tokenizer, id2word, max_len=max_len)

    # 使用新的 collate_fn
    bert_collate_fn = create_bert_collate_fn(pad_id=pad_id, label_pad_id=label_pad_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=bert_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=bert_collate_fn)

    # --- 初始化模型 ---
    num_classes = len(id2tag)
    model = BertSegmenter(num_classes=num_classes).to(device)
    print("模型结构:")
    print(model)

    # --- 定义损失函数和优化器 ---
    criterion = nn.CrossEntropyLoss(ignore_index=label_pad_id) # 使用 label_pad_id
    # 设置不同的学习率
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    # 这里的 AdamW 现在是 torch.optim.AdamW
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # 学习率调度器 (保持不变)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, # 可以设置 warmup步数，例如 total_steps * 0.1
        num_training_steps=total_steps
    )


    # --- 训练循环 ---
    best_val_loss = float('inf')
    print("\n开始训练...")
    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        elapsed_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed_time:.2f}s | LR: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  模型已保存到: {model_save_path} (Best Loss)")

        print("-" * 60)

    print("训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()