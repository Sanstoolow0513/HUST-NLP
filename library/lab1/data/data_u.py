import codecs
from sklearn.model_selection import train_test_split
import pickle
from collections import Counter
import matplotlib.pyplot as plt  # 用于可视化

INPUT_DATA = "train.txt"
SAVE_PATH = "./datasave.pkl"
id2tag = ['B', 'M', 'E', 'S']  # B：分词头部 M：分词词中 E：分词词尾 S：独立成词
tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
word2id = {}
id2word = []


def getList(input_str):
    '''
    单个分词转换为tag序列
    :param input_str: 单个分词
    :return: tag序列
    '''
    outpout_str = []
    if len(input_str) == 1:
        outpout_str.append(tag2id['S'])
    elif len(input_str) == 2:
        outpout_str = [tag2id['B'], tag2id['E']]
    else:
        M_num = len(input_str) - 2
        M_list = [tag2id['M']] * M_num
        outpout_str.append(tag2id['B'])
        outpout_str.extend(M_list)
        outpout_str.append(tag2id['E'])
    return outpout_str


def length_group(length):
    '''
    根据句子长度分组
    :param length: 句子长度
    :return: 分组类别
    '''
    if length <= 5:
        return 0
    elif length <= 10:
        return 1
    elif length <= 20:
        return 2
    else:
        return 3


def visualize_length_distribution(lengths):
    '''
    可视化句子长度分布
    :param lengths: 所有句子的长度列表
    '''
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title("Sentence Length Distribution", fontsize=16)
    plt.xlabel("Sentence Length", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def handle_data():
    '''
    处理数据，并保存至savepath
    :return:
    '''
    x_data = []
    y_data = []
    wordnum = 0
    line_num = 0
    with open(INPUT_DATA, 'r', encoding="utf-8") as ifp:
        for line in ifp:
            line_num = line_num + 1
            line = line.strip()
            if not line:
                continue
            line_x = []
            for i in range(len(line)):
                if line[i] == " ":
                    continue
                if (line[i] in id2word):
                    line_x.append(word2id[line[i]])
                else:
                    id2word.append(line[i])
                    word2id[line[i]] = wordnum
                    line_x.append(wordnum)
                    wordnum = wordnum + 1
            x_data.append(line_x)

            lineArr = line.split()
            line_y = []
            for item in lineArr:
                line_y.extend(getList(item))
            y_data.append(line_y)

    print(x_data[0])
    print([id2word[i] for i in x_data[0]])
    print(y_data[0])
    print([id2tag[i] for i in y_data[0]])
    
    # 统计句子长度分布
    sentence_lengths = [len(s) for s in x_data]
    length_counts = Counter(sentence_lengths)
    print(length_counts)  # 打印每种长度的句子数量

    # 可视化句子长度分布
    visualize_length_distribution(sentence_lengths)

    # 根据句子长度分组
    length_groups = [length_group(len(s)) for s in x_data]

    # 在handle_data函数中改进数据划分
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, 
        test_size=0.2,  # 增加测试集比例
        random_state=42,
        stratify=length_groups  # 使用分组后的类别进行分层抽样
    )
    with open(SAVE_PATH, 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)


if __name__ == "__main__":
    handle_data()
