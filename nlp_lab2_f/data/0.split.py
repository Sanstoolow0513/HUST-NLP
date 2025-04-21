corpus_file = 'RMRB_NER_CORPUS.txt'
corpus = []
with open(corpus_file, 'r', encoding='utf-8') as f:  # 添加编码参数
    record = []
    for line in f:
        if line != '\n':
            record.append(line.strip('\n').split(' '))
        else:
            corpus.append(record)
            record = []

import random
random.seed(43)  # 固定随机种子确保可复现
random.shuffle(corpus)

total_len = len(corpus)
train_len = int(total_len * 0.8)  # 80%训练集
valid_len = int(total_len * 0.1)  # 10%验证集

train = corpus[:train_len]
valid = corpus[train_len:train_len+valid_len]
test = corpus[train_len+valid_len:]  # 剩余10%测试集

# 添加数据统计信息
print(f"总样本数: {total_len}")
print(f"训练集: {len(train)} ({len(train)/total_len:.1%})")
print(f"验证集: {len(valid)} ({len(valid)/total_len:.1%})")
print(f"测试集: {len(test)} ({len(test)/total_len:.1%})")

train_file = 'ner_train.txt'
valid_file = 'ner_valid.txt'
test_file = 'ner_test.txt'

for split_file, split_corpus in zip([train_file, valid_file, test_file],
                                   [train, valid, test]):
    with open(split_file, 'w', encoding='utf-8') as f:  # 添加编码参数
        for sentence in split_corpus:
            for word, label in sentence:
                f.write(f"{word} {label}\n")  # 优化写入格式
            f.write('\n')

