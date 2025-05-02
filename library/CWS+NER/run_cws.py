import pickle
import logging
import argparse
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import CWS
from dataloader import Sentence, DataAugmentation

def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=128)  # 增加嵌入维度
    parser.add_argument('--lr', type=float, default=0.001)  # 调整学习率
    parser.add_argument('--max_epoch', type=int, default=20)  # 增加训练轮数
    parser.add_argument('--batch_size', type=int, default=256)  # 调整批次大小
    parser.add_argument('--hidden_dim', type=int, default=256)  # 增加隐藏层维度
    parser.add_argument('--dropout', type=float, default=0.3)  # 添加dropout参数
    parser.add_argument('--num_layers', type=int, default=2)  # 添加LSTM层数参数
    parser.add_argument('--patience', type=int, default=3)  # 早停耐心值
    parser.add_argument('--weight_decay', type=float, default=1e-5)  # 权重衰减
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--save_dir', type=str, default='save/CWS')
    parser.add_argument('--data_augmentation', action='store_true', default=False)  # 是否使用数据增强
    return parser.parse_args()


def set_logger(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = os.path.join(save_dir, 'log.txt')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m%d %H:%M:%S',
        filename=log_file,
        filemode='w',
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def entity_split(x, y, id2tag, entities, cur):
    start, end = -1, -1
    for j in range(len(x)):
        if id2tag[y[j]] == 'B':
            start = cur + j
        elif id2tag[y[j]] == 'M' and start != -1:
            continue
        elif id2tag[y[j]] == 'E' and start != -1:
            end = cur + j
            entities.add((start, end))
            start, end = -1, -1
        elif id2tag[y[j]] == 'S':
            entities.add((cur + j, cur + j))
            start, end = -1, -1
        else:
            start, end = -1, -1


def evaluate(model, data_loader, id2tag, use_cuda):
    """评估模型性能"""
    entity_predict = set()
    entity_label = set()
    
    with torch.no_grad():
        model.eval()
        cur = 0
        for sentence, label, mask, length in data_loader:
            if use_cuda:
                sentence = sentence.cuda()
                label = label.cuda()
                mask = mask.cuda()
            predict = model.infer(sentence, mask, length)

            for i in range(len(length)):
                entity_split(sentence[i, :length[i]], predict[i], id2tag, entity_predict, cur)
                entity_split(sentence[i, :length[i]], label[i, :length[i]], id2tag, entity_label, cur)
                cur += length[i]

    right_predict = [i for i in entity_predict if i in entity_label]
    
    if len(entity_predict) == 0:
        precision = 0
    else:
        precision = float(len(right_predict)) / len(entity_predict)
        
    if len(entity_label) == 0:
        recall = 0
    else:
        recall = float(len(right_predict)) / len(entity_label)
        
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
        
    return precision, recall, f1


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    if use_cuda:
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    with open('data/CWS/data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    # 创建模型
    model = CWS(len(word2id), tag2id, args.embedding_dim, args.hidden_dim, 
                dropout=args.dropout, num_layers=args.num_layers)
    if use_cuda:
        model = model.to(device)
        
    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'模型总参数量: {total_params}, 可训练参数量: {trainable_params}')
    
    for name, param in model.named_parameters():
        logging.debug('%s: %s, require_grad=%s' % (name, str(param.shape), str(param.requires_grad)))

    # 优化器
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    # 数据加载
    train_data = DataLoader(
        dataset=Sentence(x_train, y_train),
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=4
    )

    test_data = DataLoader(
        dataset=Sentence(x_test, y_test),
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=4
    )

    # 训练
    best_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(args.max_epoch):
        start_time = time.time()
        model.train()
        step = 0
        total_loss = 0
        
        for sentence, label, mask, length in train_data:
            if use_cuda:
                sentence = sentence.to(device)
                label = label.to(device)
                mask = mask.to(device)

            # 前向传播
            loss = model(sentence, label, mask, length)
            total_loss += loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()

            step += 1
            if step % 50 == 0:
                logging.debug('epoch %d-step %d loss: %.4f' % (epoch, step, loss.item()))

        # 计算平均损失
        avg_loss = total_loss / step
        
        # 评估
        precision, recall, f1 = evaluate(model, test_data, id2tag, use_cuda)
        
        # 更新学习率
        scheduler.step(f1)
        
        # 记录训练信息
        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch}: loss={avg_loss:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, time={epoch_time:.2f}s")
        
        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            patience_counter = 0
            
            # 保存模型
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            best_model_path = os.path.join(args.save_dir, "best_model.pkl")
            torch.save(model, best_model_path)
            logging.info(f"最佳模型已保存至 {best_model_path}, F1={best_f1:.4f}")
        else:
            patience_counter += 1
            
        # 保存当前轮次模型
        path_name = os.path.join(args.save_dir, f"model_epoch{epoch}.pkl")
        torch.save(model, path_name)
        
        # 早停
        if patience_counter >= args.patience:
            logging.info(f"早停触发! {args.patience} 轮未提升")
            break
    
    logging.info(f"训练完成! 最佳F1: {best_f1:.4f}, 最佳轮次: {best_epoch}")


if __name__ == '__main__':
    args = get_param()
    set_logger(args.save_dir)
    main(args)
