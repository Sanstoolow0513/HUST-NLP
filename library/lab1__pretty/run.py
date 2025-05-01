import pickle
import logging
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import CWS # 确保 CWS 类已正确导入
from dataloader import Sentence # 确保 Sentence 类已正确导入
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=200)  # 增加embedding维度
    parser.add_argument('--lr', type=float, default=0.001)  # 调整学习率
    parser.add_argument('--max_epoch', type=int, default=20)  # 增加训练轮数
    parser.add_argument('--batch_size', type=int, default=256)  # 调整batch大小
    parser.add_argument('--hidden_dim', type=int, default=300)  # 增加隐藏层维度
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=1e-5)  # 添加权重衰减
    return parser.parse_args()


def set_logger():
    log_file = os.path.join('save', 'log.txt')
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
    # 注意：这里的 len(x) 应该是实际序列长度，而不是填充后的长度
    # 在调用 entity_split 时需要确保传入的是未填充的部分
    # 或者在 entity_split 内部根据 mask 或 length 处理
    # 但当前代码似乎依赖于外部传入正确的长度相关的 y
    # 为了保持与原逻辑一致，暂时不修改这里，但需注意潜在问题
    for j in range(len(y)): # 使用标签 y 的长度可能更安全，因为它对应有效部分
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


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") # 定义 device

    # 加载数据，包含字符类型相关数据
    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        char_type_vocab_size = pickle.load(inp) # <--- 加载 char_type_vocab_size
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_char_types_train = pickle.load(inp) # <--- 加载 x_char_types_train
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
        x_char_types_test = pickle.load(inp) # <--- 加载 x_char_types_test

    # 初始化模型，传入 char_type_vocab_size
    model = CWS(len(word2id), tag2id, args.embedding_dim, args.hidden_dim, char_type_vocab_size) # <--- 添加 char_type_vocab_size
    model.to(device) # 将模型移动到指定设备

    for name, param in model.named_parameters():
        logging.debug('%s: %s, require_grad=%s' % (name, str(param.shape), str(param.requires_grad)))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5, verbose=True) # 使用 'max' 因为我们希望 F1 score 越大越好, 添加 verbose

    # 创建 DataLoader，传入字符类型数据
    # 注意：如果数据集很大，num_workers > 0 在 Windows 上有时会出问题，可以先设为 0 调试
    train_data = DataLoader(
        dataset=Sentence(x_train, y_train, x_char_types_train), # <--- 传入 x_char_types_train
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=0 # 建议先用 0 调试
    )

    # 限制测试集大小以加快评估速度
    test_limit = 1000
    test_data = DataLoader(
        dataset=Sentence(x_test[:test_limit], y_test[:test_limit], x_char_types_test[:test_limit]), # <--- 传入 x_char_types_test
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=0 # 建议先用 0 调试
    )

    best_f1 = 0.0 # 用于保存最佳 F1 分数
    save_dir = 'save' # 定义模型保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(args.max_epoch):
        model.train() # 确保模型处于训练模式
        step = 0
        epoch_loss = 0.0
        log = []
        # 训练循环，解包并传递 char_types
        for sentence, label, char_types, mask, length in train_data: # <--- 解包 char_types
            sentence, label, char_types, mask = sentence.to(device), label.to(device), char_types.to(device), mask.to(device) # 移动数据到设备

            # forward
            loss = model(sentence, label, char_types, mask, length) # <--- 传递 char_types
            log.append(loss.item())
            epoch_loss += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            # 可以添加梯度裁剪防止梯度爆炸
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            step += 1
            if step % 100 == 0:
                avg_loss = sum(log)/len(log)
                logging.debug(f'Epoch {epoch}-Step {step} Avg Loss: {avg_loss:.4f}')
                log = []

        avg_epoch_loss = epoch_loss / len(train_data)
        logging.info(f'Epoch {epoch} Average Training Loss: {avg_epoch_loss:.4f}')

        # 测试
        entity_predict = set()
        entity_label = set()
        model.eval() # 设置为评估模式
        cur = 0
        with torch.no_grad(): # 评估时不需要计算梯度
            # 测试循环，解包并传递 char_types
            for sentence, label, char_types, mask, length in test_data: # <--- 解包 char_types
                sentence, char_types, mask = sentence.to(device), char_types.to(device), mask.to(device) # 移动数据到设备
                # label 也移到 device，虽然 infer 不用，但 entity_split 会用
                label = label.to(device)

                predict = model.infer(sentence, char_types, mask, length) # <--- 传递 char_types

                # 注意：predict 是一个列表，列表中的每个元素是对应句子的预测标签序列 (tensor)
                # label 是一个 batch 的标签 tensor
                # 需要逐个处理 batch 中的样本
                for i in range(len(length)):
                    sent_len = length[i]
                    # 确保只使用有效长度的数据进行评估
                    pred_tags = predict[i] # predict[i] 已经是解码后的标签列表/tensor
                    true_tags = label[i, :sent_len].cpu().numpy() # 获取真实标签并转到 cpu
                    # sentence 也需要对应处理，但 entity_split 似乎没用到 sentence 内容，只用了长度信息
                    # 如果 entity_split 需要原始字符，需要传递 sentence[i, :sent_len]
                    entity_split(sentence[i, :sent_len], pred_tags, id2tag, entity_predict, cur)
                    entity_split(sentence[i, :sent_len], true_tags, id2tag, entity_label, cur)
                    cur += sent_len # cur 应该基于句子实际长度累加

            # 计算 P, R, F1
            right_predict = entity_predict.intersection(entity_label)
            precision = len(right_predict) / len(entity_predict) if len(entity_predict) > 0 else 0.0
            recall = len(right_predict) / len(entity_label) if len(entity_label) > 0 else 0.0
            f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            logging.info(f"Epoch {epoch} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f_score:.4f}")

            # 根据 F1 分数调整学习率
            scheduler.step(f_score)

            # 保存 F1 分数最高的模型
            if f_score > best_f1:
                best_f1 = f_score
                path_name = os.path.join(save_dir, "best_model.pkl")
                torch.save(model.state_dict(), path_name) # 保存模型状态字典通常更好
                logging.info(f"Best model saved with F1: {best_f1:.4f} in {path_name}")

        # 每个 epoch 后保存一次模型（可选）
        # path_name = os.path.join(save_dir, f"model_epoch{epoch}.pkl")
        # torch.save(model.state_dict(), path_name)
        # logging.info(f"Model saved for epoch {epoch} in {path_name}")


if __name__ == '__main__':
    set_logger()
    args = get_param()
    logging.info(f"Arguments: {args}") # 记录参数
    main(args)
