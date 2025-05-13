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
    parser.add_argument('--embedding_dim', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--max_epoch', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--early_stopping', type=int, default=5, help='提前停止训练的耐心值')
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
        if len(right_predict) != 0 and len(entity_predict) != 0:
            precision = float(len(right_predict)) / len(entity_predict)
            recall = float(len(right_predict)) / len(entity_label)
            fscore = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            return precision, recall, fscore
        else:
            return 0, 0, 0


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()

    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_val = pickle.load(inp)  # 加载验证集
        y_val = pickle.load(inp)  # 加载验证集
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    model = CWS(len(word2id), tag2id, args.embedding_dim, args.hidden_dim)
    if use_cuda:
        model = model.cuda()
    for name, param in model.named_parameters():
        logging.debug('%s: %s, require_grad=%s' % (name, str(param.shape), str(param.requires_grad)))

    optimizer = Adam(model.parameters(), lr=args.lr)

    train_data = DataLoader(
        dataset=Sentence(x_train, y_train),
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=6
    )

    val_data = DataLoader(  # 创建验证集数据加载器
        dataset=Sentence(x_val, y_val),
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=6
    )

    test_data = DataLoader(
        dataset=Sentence(x_test[:1000], y_test[:1000]),
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=6
    )

    best_fscore = 0
    patience = args.early_stopping
    patience_counter = 0

    for epoch in range(args.max_epoch):
        step = 0
        log = []
        for sentence, label, mask, length in train_data:
            if use_cuda:
                sentence = sentence.cuda()
                label = label.cuda()
                mask = mask.cuda()

            # forward
            loss = model(sentence, label, mask, length)
            log.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 100 == 0:
                logging.debug('epoch %d-step %d loss: %f' % (epoch, step, sum(log)/len(log)))
                log = []

        # 在验证集上评估
        logging.info("在验证集上评估...")
        val_precision, val_recall, val_fscore = evaluate(model, val_data, id2tag, use_cuda)
        logging.info("验证集 - precision: %f, recall: %f, fscore: %f" % (val_precision, val_recall, val_fscore))
        
        # 在测试集上评估
        logging.info("在测试集上评估...")
        test_precision, test_recall, test_fscore = evaluate(model, test_data, id2tag, use_cuda)
        logging.info("测试集 - precision: %f, recall: %f, fscore: %f" % (test_precision, test_recall, test_fscore))
        
        # 保存最佳模型
        if val_fscore > best_fscore:
            best_fscore = val_fscore
            patience_counter = 0
            best_model_path = "./save/best_model.pkl"
            torch.save(model, best_model_path)
            logging.info("发现更好的模型，已保存至 %s" % best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("早停：验证集性能 %d 个epoch未提升，停止训练" % patience)
                break
        
        # 每个epoch都保存一个模型
        path_name = "./save/model_epoch" + str(epoch) + ".pkl"
        torch.save(model, path_name)
        logging.info("模型已保存至 %s" % path_name)
        
        # 重置模型为训练模式
        model.train()


if __name__ == '__main__':
    set_logger()
    main(get_param())
