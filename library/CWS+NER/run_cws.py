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
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--save_dir', type=str, default='save/CWS')
    parser.add_argument('--use_features', action='store_true', default=False, help='是否使用字符特征')
    parser.add_argument('--feature_dim', type=int, default=4, help='字符特征的维度')
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
    
    # 确保 x 和 y 是列表或numpy数组
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
        
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


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    try:
        with open('data/CWS/data/datasave.pkl', 'rb') as inp:
            word2id = pickle.load(inp)
            id2word = pickle.load(inp)
            tag2id = pickle.load(inp)
            id2tag = pickle.load(inp)
            x_train = pickle.load(inp)
            y_train = pickle.load(inp)
            x_test = pickle.load(inp)
            y_test = pickle.load(inp)
            # 尝试加载特征数据 (如果存在)
            try:
                features_train = pickle.load(inp)
                features_test = pickle.load(inp)
                logging.info("特征数据已加载。")
            except EOFError:
                features_train = None
                features_test = None
                logging.info("未找到特征数据，将不使用特征。")
                args.use_features = False
    except Exception as e:
        logging.error(f"加载数据时出错: {e}")
        return

    # 根据是否使用特征来确定 feature_dim
    actual_feature_dim = args.feature_dim if args.use_features else 0

    # 初始化模型时传入 feature_dim
    model = CWS(len(word2id), tag2id, args.embedding_dim, args.hidden_dim, feature_dim=actual_feature_dim)
    model.to(device)

    if use_cuda:
        model = model.cuda()
    for name, param in model.named_parameters():
        logging.debug('%s: %s, require_grad=%s' % (name, str(param.shape), str(param.requires_grad)))

    optimizer = Adam(model.parameters(), lr=args.lr)

    train_data = DataLoader(
        dataset=Sentence(x_train, y_train, id2word, use_features=args.use_features),
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=0  # 建议在Windows下设为0
    )

    test_dataset = Sentence(x_test[:1000], y_test[:1000], id2word, use_features=args.use_features)
    test_data = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=0  # 建议在Windows下设为0
    )

    for epoch in range(args.max_epoch):
        step = 0
        log = []
        for batch in train_data:
            # 根据是否有特征解包 batch
            if args.use_features:
                sentence, label, mask, length, features = batch
                features = features.to(device)
            else:
                sentence, label, mask, length = batch
                features = None

            sentence = sentence.to(device)
            label = label.to(device)
            mask = mask.to(device)

            # forward
            loss = model(sentence, label, mask, length, features=features)
            log.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 100 == 0:
                logging.debug('epoch %d-step %d loss: %f' % (epoch, step, sum(log)/len(log)))
                log = []

        # test
        entity_predict = set()
        entity_label = set()
        with torch.no_grad():
            model.eval()
            cur = 0
            for sentence, label, mask, length in test_data:
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
            if len(right_predict) != 0:
                precision = float(len(right_predict)) / len(entity_predict)
                recall = float(len(right_predict)) / len(entity_label)
                logging.info("precision: %f" % precision)
                logging.info("recall: %f" % recall)
                logging.info("fscore: %f" % ((2 * precision * recall) / (precision + recall)))
            else:
                logging.info("precision: 0")
                logging.info("recall: 0")
                logging.info("fscore: 0")
            model.train()

        # 保存模型
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        path_name = os.path.join(args.save_dir, "model_epoch" + str(epoch) + ".pkl")
        torch.save(model, path_name)
        logging.info("model has been saved in %s" % path_name)


if __name__ == '__main__':
    args = get_param()
    set_logger(args.save_dir)
    main(args)
