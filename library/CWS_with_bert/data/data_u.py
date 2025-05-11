import codecs
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import pickle
import numpy as np
import collections

INPUT_DATA = "train.txt"
# INPUT_DATA = "debug_train.txt"
SAVE_PATH = "./datasave.pkl"
# B：分词头部 M：分词词中 E：分词词尾 S：独立成词 O: 其他
id2tag = ['B', 'M', 'E', 'S', 'O']  
tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3, 'O': 4}
word2id = {}
id2word = []
UNK_TOKEN = '<UNK>'  # 未登录词标记


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


def get_stratification_labels(y_data):
    '''
    为分层采样创建标签
    基于每个样本中的标签分布创建分层标签
    :param y_data: 标签数据
    :return: 分层标签列表
    '''
    strat_labels = []
    for tags in y_data:
        # 计算每个样本中BMES标签的分布
        tag_counts = collections.Counter([id2tag[t] for t in tags])
        # 创建一个简化的分层标签，基于样本长度和S标签比例
        sample_len = len(tags)
        s_ratio = tag_counts.get('S', 0) / sample_len if sample_len > 0 else 0
        
        # 将样本分为几个桶，减少桶的数量以确保每个桶有足够的样本
        if sample_len < 20:
            len_bucket = 0
        elif sample_len < 50:
            len_bucket = 1
        else:
            len_bucket = 2
            
        if s_ratio < 0.3:
            ratio_bucket = 0
        elif s_ratio < 0.6:
            ratio_bucket = 1
        else:
            ratio_bucket = 2
            
        strat_labels.append(f"{len_bucket}_{ratio_bucket}")
    
    return strat_labels


def handle_data():
    '''
    处理数据，并保存至savepath
    :return:
    '''
    x_data = []
    y_data = []
    wordnum = 0
    line_num = 0
    
    # 添加未登录词标记
    id2word.append(UNK_TOKEN)
    word2id[UNK_TOKEN] = wordnum
    wordnum += 1
    
    # 统计词频
    word_counts = collections.Counter()
    
    # 第一遍扫描，统计词频
    with open(INPUT_DATA, 'r', encoding="utf-8") as ifp:
        for line in ifp:
            line = line.strip()
            if not line:
                continue
            for char in line:
                if char != " ":
                    word_counts[char] += 1
    
    # 设置低频词阈值，低于此阈值的词被视为未登录词
    min_freq = 2
    
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
                
                # 处理低频词为未登录词
                if word_counts[line[i]] < min_freq:
                    line_x.append(word2id[UNK_TOKEN])
                elif line[i] in word2id:
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

    print("数据集总样本数:", len(x_data))
    print("词表大小:", len(id2word))
    print("未登录词处理阈值:", min_freq)
    
    # 示例数据
    print("示例数据:")
    print(x_data[0])
    print([id2word[i] for i in x_data[0]])
    print(y_data[0])
    print([id2tag[i] for i in y_data[0]])
    
    # 获取分层采样的标签
    strat_labels = get_stratification_labels(y_data)
    
    # 统计每个分层标签的样本数量
    label_counts = collections.Counter(strat_labels)
    print("分层标签统计:")
    for label, count in label_counts.items():
        print(f"{label}: {count}样本")
    
    # 恢复原始的分割逻辑
    x_train, x_temp, y_train, y_temp, strat_train, strat_temp = train_test_split(
        x_data, y_data, strat_labels, test_size=0.2, random_state=42, stratify=strat_labels
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"训练集大小: {len(x_train)}, 验证集大小: {len(x_val)}, 测试集大小: {len(x_test)}")
    print(f"划分比例: 训练集 {len(x_train)/len(x_data):.2f}, 验证集 {len(x_val)/len(x_data):.2f}, 测试集 {len(x_test)/len(x_data):.2f}")
    
    print("\nDebug prints for first sample are now commented out in data_u.py.")
    
    with open(SAVE_PATH, 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_val, outp)
        pickle.dump(y_val, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)
    
    print(f"数据已保存至 {SAVE_PATH}")


if __name__ == "__main__":
    handle_data()
