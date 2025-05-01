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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--save_dir', type=str, default='save/NER')
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

    with open('data/NER/data/ner_datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
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

    test_data = DataLoader(
        dataset=Sentence(x_test[:1000], y_test[:1000]),
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=6
    )

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

        # 测试部分
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
                precision = float(len(right_predict)) / len(entity_predict) if len(entity_predict) > 0 else 0
                recall = float(len(right_predict)) / len(entity_label) if len(entity_label) > 0 else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                logging.info("precision: %f" % precision)
                logging.info("recall: %f" % recall)
                logging.info("f1 score: %f" % f1)
            else:
                logging.info("precision: 0")
                logging.info("recall: 0")
                logging.info("f1 score: 0")
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
