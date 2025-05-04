import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.utils.rnn import pad_sequence # 使用 pad_sequence 简化 padding

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TransformerSegmenter # 假设 model.py 在同一目录下或父目录

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 定义数据集 (简化，直接使用加载的数据)
class SegmentationDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
        if len(self.sentences) != len(self.labels):
             raise ValueError("Sentences and labels must have the same length!")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # 返回原始列表，collate_fn 会处理成 Tensor
        return {
            'x': self.sentences[idx],
            'y': self.labels[idx]
        }

# 定义数据整理函数 (使用 pad_sequence)
def create_collate_fn(pad_id, label_pad_id=-100):
    def collate_fn(batch):
        # 从 batch 中分离 x 和 y
        batch_x = [torch.tensor(item['x'], dtype=torch.long) for item in batch]
        batch_y = [torch.tensor(item['y'], dtype=torch.long) for item in batch]

        # 对 x 和 y 进行 padding
        # batch_first=True 让输出形状为 (batch_size, max_len)
        padded_x = pad_sequence(batch_x, batch_first=True, padding_value=pad_id)
        padded_y = pad_sequence(batch_y, batch_first=True, padding_value=label_pad_id)

        # 创建 attention mask (True 表示是 padding)
        mask = (padded_x == pad_id)

        return {
            'x': padded_x,
            'y': padded_y,
            'mask': mask
        }
    return collate_fn

# 定义训练函数
def train(model, train_loader, optimizer, criterion, device, use_crf=False, clip_grad_norm=1.0):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        mask = batch['mask'].to(device) # mask 现在是 True 表示 padding

        optimizer.zero_grad()

        # 前向传播
        if use_crf:
            # CRF 模型直接返回负对数似然损失
            loss = model(x, mask=mask, labels=y) # 确保模型 forward 接受 mask
        else:
            # 非 CRF 模型
            emissions = model(x, mask=mask) # 确保模型 forward 接受 mask
            # 计算损失 (需要忽略 padding 的标签)
            # emissions: [batch, seq_len, num_classes]
            # y: [batch, seq_len]
            loss = criterion(emissions.view(-1, emissions.shape[-1]), y.view(-1))

        # 处理 NaN loss
        if torch.isnan(loss):
            print("警告：检测到 NaN 损失，跳过此批次。")
            continue

        # 反向传播和优化
        loss.backward()

        # 梯度裁剪
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader) if len(train_loader) > 0 else 0.0


# 定义验证函数 (需要更新以处理 mask 和 CRF 输出)
def validate(model, val_loader, criterion, device, id2tag, use_crf=False, label_pad_id=-100):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_classes = len(id2tag)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            mask = batch['mask'].to(device) # True for padding

            # 前向传播
            if use_crf:
                # 推理模式，CRF 返回最佳路径 (List[List[int]])
                preds_batch = model(x, mask=mask) # 确保模型 forward 在 eval 模式下返回路径
                # 计算损失（仅用于记录, 需要 labels）
                try:
                    # 某些CRF实现可能在eval时不计算loss，或者需要特定调用
                    loss = model(x, mask=mask, labels=y)
                    total_loss += loss.item()
                except Exception:
                     # 如果模型在eval时不能直接计算loss，则跳过loss计算
                     pass # 或者设置为0

                # 收集有效的预测和标签
                for i, pred_seq in enumerate(preds_batch):
                    true_labels = y[i][~mask[i]].cpu().tolist() # 获取非padding的真实标签
                    # CRF 输出的 pred_seq 应该已经是对应非 padding 部分的长度
                    if len(pred_seq) != len(true_labels):
                         print(f"警告: CRF 输出长度 {len(pred_seq)} 与非填充标签长度 {len(true_labels)} 不匹配。")
                         # 尝试截断或填充，或者跳过此样本
                         min_len = min(len(pred_seq), len(true_labels))
                         all_preds.extend(pred_seq[:min_len])
                         all_labels.extend(true_labels[:min_len])
                    else:
                        all_preds.extend(pred_seq)
                        all_labels.extend(true_labels)

            else:
                # 非 CRF 模型
                emissions = model(x, mask=mask)
                # 计算损失
                loss = criterion(emissions.view(-1, emissions.shape[-1]), y.view(-1))
                total_loss += loss.item()

                # 获取预测结果
                preds_batch = torch.argmax(emissions, dim=2) # [batch, seq_len]

                # 收集有效的预测和标签 (忽略 padding)
                for i in range(x.size(0)):
                    valid_indices = ~mask[i] # 非 padding 的位置为 True
                    valid_preds = preds_batch[i][valid_indices].cpu().tolist()
                    valid_labels = y[i][valid_indices].cpu().tolist()
                    all_preds.extend(valid_preds)
                    all_labels.extend(valid_labels)

    # 计算指标
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    if not all_labels: # 如果没有收集到任何标签（例如验证集为空或全是错误）
        print("警告：验证过程中未收集到有效标签，无法计算指标。")
        return avg_loss, 0.0, 0.0

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    # 计算 F1 (macro average, 忽略 label_pad_id)
    # labels 参数指定了要包含在计算中的标签类别 ID
    valid_label_ids = list(range(num_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', labels=valid_label_ids, zero_division=0
    )

    # 打印每个标签的指标 (可选)
    # ...

    return avg_loss, accuracy, f1 # 返回 macro F1


# --- 主函数 ---
def main():
    set_seed(42)

    # --- 参数设置 ---
    data_path = "../data/datasave.pkl"
    model_save_dir = "./saved_models_transformer" # 模型保存目录
    batch_size = 256
    epochs = 20
    learning_rate = 0.0005 # 调整学习率
    embedding_dim = 256
    # hidden_dim = 512 # 重命名为 transformer_ff_dim
    transformer_ff_dim = 512 # Transformer 内部前馈网络维度
    num_layers = 4
    num_heads = 8
    dropout = 0.2
    use_crf = True # 保持使用 CRF
    clip_grad_norm = 1.0 # 梯度裁剪阈值
    patience = 3 # Early stopping patience
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用设备: {device}")
    os.makedirs(model_save_dir, exist_ok=True) # 创建模型保存目录

    # --- 加载数据 ---
    print(f"加载数据: {data_path}")
    try:
        with open(data_path, 'rb') as f:
            processed_data = pickle.load(f)
        word2id = processed_data['word2id']
        id2word = processed_data['id2word']
        tag2id = processed_data['tag2id']
        id2tag = processed_data['id2tag']
        x_train = processed_data['x_train']
        y_train = processed_data['y_train']
        x_test = processed_data['x_test']
        y_test = processed_data['y_test']
        pad_id = processed_data['pad_id']
        label_pad_id = -100 # PyTorch CrossEntropyLoss 默认忽略 -100
    except FileNotFoundError:
        print(f"错误：数据文件 {data_path} 未找到。请先运行 data_u.py。")
        sys.exit(1)
    except Exception as e:
        print(f"加载数据文件 {data_path} 时出错: {e}")
        sys.exit(1)

    print(f"数据加载成功。训练集: {len(x_train)}, 测试集: {len(x_test)}")
    print(f"词表大小: {len(word2id)}, 标签数: {len(tag2id)}")

    # --- 创建 Dataset 和 DataLoader ---
    train_dataset = SegmentationDataset(x_train, y_train)
    val_dataset = SegmentationDataset(x_test, y_test)

    # 使用新的 collate_fn
    collate_fn_with_padding = create_collate_fn(pad_id=pad_id, label_pad_id=label_pad_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_with_padding)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn_with_padding)

    # --- 初始化模型 ---
    vocab_size = len(word2id)
    num_classes = len(tag2id)
    model = TransformerSegmenter(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        transformer_ff_dim=transformer_ff_dim, # 使用新名称
        num_classes=num_classes,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        use_crf=use_crf,
        padding_idx=pad_id # !!! 传递 pad_id !!!
    ).to(device)
    print("模型结构:")
    print(model)

    # --- 定义损失函数和优化器 ---
    # CRF 模型内部处理损失，非 CRF 使用 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(ignore_index=label_pad_id) if not use_crf else None
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience // 2) # 基于 F1 调整

    # --- 训练循环 ---
    best_val_f1 = 0.0
    epochs_no_improve = 0
    train_losses, val_losses, val_accuracies, val_f1_scores = [], [], [], []

    print("\n开始训练...")
    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, device, use_crf, clip_grad_norm)
        val_loss, val_accuracy, val_f1 = validate(model, val_loader, criterion, device, id2tag, use_crf, label_pad_id)

        elapsed_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)

        print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed_time:.2f}s | LR: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f}") # 打印 F1

        # 更新学习率 (基于验证 F1)
        scheduler.step(val_f1)

        # 保存最佳模型 (基于验证 F1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # model_save_path = os.path.join(model_save_dir, "transformer_segmenter_best.pth") # 可以改名区分
            model_save_path = os.path.join(model_save_dir, "transformer_segmenter_best_f1.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"  模型已保存到: {model_save_path} (Best F1)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n验证集 F1 分数连续 {patience} 个 epoch 没有提升，触发 Early Stopping。")
            break

        print("-" * 60)

    # --- 绘制训练过程图表 ---
    print("训练完成！绘制指标图表...")
    plt.figure(figsize=(12, 8))
    epochs_range = range(1, len(train_losses) + 1)

    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.plot(epochs_range, val_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Metrics')

    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, 'training_metrics_transformer.png'))
    plt.close()
    print(f"训练指标图表已保存到: {os.path.join(model_save_dir, 'training_metrics_transformer.png')}")
    print(f"最佳验证 F1 分数: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()