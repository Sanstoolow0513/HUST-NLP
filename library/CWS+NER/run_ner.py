import pickle
import logging
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import CWS
from dataloader import Sentence

def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--save_dir', type=str, default='save/NER')
    # 在训练参数中添加
    # parser.add_argument('--cws_weight', type=float, default=0.5, help='CWS loss权重') # 这个参数似乎与NER无关，可以移除或注释掉
    parser.add_argument('--use_features', action='store_true', default=False, help='是否使用字符特征') # 新增：控制是否使用特征
    parser.add_argument('--feature_dim', type=int, default=4, help='字符特征的维度') # 新增：特征维度，根据dataloader.py中的实现设置
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
    entity_type = None
    
    # 确保 x 和 y 是列表或numpy数组
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
        
    for j in range(len(x)):
        tag = id2tag[y[j]]
        if tag.startswith('B-'):
            start = cur + j
            entity_type = tag[2:]  # 提取实体类型
        elif tag.startswith('I-') and start != -1 and entity_type == tag[2:]:
            continue
        elif tag.startswith('E-') and start != -1 and entity_type == tag[2:]:
            end = cur + j
            entities.add((start, end, entity_type))
            start, end = -1, -1
            entity_type = None
        elif tag.startswith('S-'):
            entities.add((cur + j, cur + j, tag[2:]))
            start, end = -1, -1
            entity_type = None
        elif tag == 'O':
            start, end = -1, -1
            entity_type = None


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") # 定义 device

    set_logger(args.save_dir) # 设置日志记录器

    # 修改数据加载，尝试加载特征
    try:
        with open('data/NER/data/ner_datasave.pkl', 'rb') as inp:
            word2id = pickle.load(inp)
            id2word = pickle.load(inp)
            tag2id = pickle.load(inp)
            id2tag = pickle.load(inp)
            x_train = pickle.load(inp)
            y_train = pickle.load(inp)
            x_test = pickle.load(inp) # 注意：原始脚本加载的是 x_valid, y_valid
            y_test = pickle.load(inp) # 注意：原始脚本加载的是 x_valid, y_valid
            # 尝试加载特征数据 (如果存在)
            try:
                features_train = pickle.load(inp)
                features_test = pickle.load(inp)
                logging.info("特征数据已加载。")
            except EOFError:
                features_train = None
                features_test = None
                logging.info("未找到特征数据，将不使用特征。")
                args.use_features = False # 如果文件末尾没有特征，则强制不使用

    except FileNotFoundError:
        logging.error("错误：找不到数据文件 data/NER/data/ner_datasave.pkl")
        return
    except Exception as e:
        logging.error(f"加载数据时出错: {e}")
        return

    # 根据是否使用特征来确定 feature_dim
    actual_feature_dim = args.feature_dim if args.use_features else 0

    # 初始化模型时传入 feature_dim
    model = CWS(len(word2id), tag2id, args.embedding_dim, args.hidden_dim, feature_dim=actual_feature_dim) # 传入 feature_dim
    model.to(device) # 将模型移动到指定设备

    for name, param in model.named_parameters():
        logging.debug('%s: %s, require_grad=%s' % (name, str(param.shape), str(param.requires_grad)))

    optimizer = Adam(model.parameters(), lr=args.lr)

    # 创建 Dataset 实例时传入 id2word 和 use_features
    train_dataset = Sentence(x_train, y_train, id2word, use_features=args.use_features)
    train_data = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=0 # 建议在Windows下设为0，避免多进程问题
    )

    # 注意：原始脚本使用的是 x_valid[:1000], y_valid[:1000]
    # 如果 pkl 文件中存的是 x_test, y_test，则使用它们
    test_dataset = Sentence(x_test[:1000], y_test[:1000], id2word, use_features=args.use_features)
    test_data = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=0 # 建议在Windows下设为0
    )

    logging.info(f"开始训练，共 {args.max_epoch} 个 epochs...")
    logging.info(f"使用特征: {args.use_features}, 特征维度: {actual_feature_dim}")

    for epoch in range(args.max_epoch):
        model.train() # 确保模型处于训练模式
        step = 0
        log = []
        # 修改训练循环以处理特征
        for batch in train_data:
            # 根据是否有特征解包 batch
            if args.use_features:
                sentence, label, mask, length, features = batch
                features = features.to(device) # 移动特征到设备
            else:
                sentence, label, mask, length = batch
                features = None # 没有特征时设为 None

            sentence = sentence.to(device)
            label = label.to(device)
            mask = mask.to(device)
            # length 不需要移动到 device，pack_padded_sequence 会处理

            # forward
            loss = model(sentence, label, mask, length, features=features) # 传递 features
            log.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            # 可以添加梯度裁剪防止梯度爆炸
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            step += 1
            if step % 100 == 0:
                avg_loss = sum(log)/len(log) if log else 0
                logging.debug('epoch %d-step %d loss: %f' % (epoch, step, avg_loss))
                log = []

        # 测试部分
        entity_predict = set()
        entity_label = set()
        model.eval() # 设置为评估模式
        cur = 0
        with torch.no_grad():
            for batch in test_data:
                # 根据是否有特征解包 batch
                if args.use_features:
                    sentence, label, mask, length, features = batch
                    features = features.to(device) # 移动特征到设备
                else:
                    sentence, label, mask, length = batch
                    features = None # 没有特征时设为 None

                sentence = sentence.to(device)
                label = label.to(device)
                mask = mask.to(device)
                # length 不需要移动到 device

                predict = model.infer(sentence, mask, length, features=features) # 传递 features

                for i in range(len(length)):
                    # 确保 predict[i] 是列表或一维张量
                    pred_tags = predict[i]
                    # entity_split 需要的是 ID 列表，不是 Tensor
                    entity_split(sentence[i, :length[i]].cpu().numpy(), # 转到 CPU 并转为 numpy
                                 pred_tags, # 直接使用解码后的 tag ID 列表
                                 id2tag, entity_predict, cur)
                    entity_split(sentence[i, :length[i]].cpu().numpy(),
                                 label[i, :length[i]].cpu().numpy(), # 转到 CPU 并转为 numpy
                                 id2tag, entity_label, cur)
                    cur += length[i]


        right_predict = [i for i in entity_predict if i in entity_label]
        precision = float(len(right_predict)) / len(entity_predict) if len(entity_predict) > 0 else 0
        recall = float(len(right_predict)) / len(entity_label) if len(entity_label) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        logging.info(f"Epoch {epoch} - Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}")


        # 保存模型
        save_path = os.path.join(args.save_dir, f"model_epoch{epoch}.pkl")
        try:
            torch.save(model, save_path)
            logging.info(f"模型已保存到 {save_path}")
        except Exception as e:
            logging.error(f"保存模型时出错: {e}")

if __name__ == '__main__':
    args = get_param()
    main(args)
