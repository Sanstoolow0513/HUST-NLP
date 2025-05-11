import codecs
from sklearn.model_selection import train_test_split
import pickle
from collections import Counter
import matplotlib.pyplot as plt  # 用于可视化

INPUT_DATA = "train.txt"
SAVE_PATH = "./datasave.pkl"
id2tag = ['B', 'M', 'E', 'S']
tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
word2id = {}
id2word = []
# 字符类型映射
char_type_map = {'CN': 0, 'EN': 1, 'NUM': 2, 'PUNC': 3, 'OTHER': 4}
char_type_vocab_size = len(char_type_map) # 添加字符类型词汇大小

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
    x_char_types_data = [] # 修改变量名以包含所有样本
    wordnum = 0
    line_num = 0

    with open(INPUT_DATA, 'r', encoding="utf-8") as ifp:
        for line in ifp:
            line_num = line_num + 1
            line = line.strip()
            if not line:
                continue
            
            line_x = []
            line_char_types = [] # 当前行的字符类型
            processed_chars = [] # 处理后的字符（去空格）

            for i in range(len(line)):
                char = line[i]
                if char == " ":
                    continue
                processed_chars.append(char) # 添加有效字符
                
                # 处理 word2id
                if char in word2id: # Use word2id directly
                    line_x.append(word2id[char])
                else:
                    id2word.append(char)
                    word2id[char] = wordnum
                    line_x.append(wordnum)
                    wordnum = wordnum + 1
                    
                # 处理字符类型
                if '\u4e00' <= char <= '\u9fff':
                    char_type = char_type_map['CN']
                elif 'a' <= char <= 'z' or 'A' <= char <= 'Z':
                    char_type = char_type_map['EN']
                elif '0' <= char <= '9':
                    char_type = char_type_map['NUM']
                # 更全面的标点符号判断 (示例)
                elif char in ',.!?;:，。！？；：\'"()[]{}<>《》“”‘’': 
                    char_type = char_type_map['PUNC']
                else:
                    char_type = char_type_map['OTHER']
                line_char_types.append(char_type)

            # 确保 line_x 和 line_char_types 长度一致
            assert len(line_x) == len(line_char_types), f"Length mismatch in line {line_num}"

            x_data.append(line_x)
            x_char_types_data.append(line_char_types) # 添加当前行的类型列表
        
            # 处理 y_data (标签)
            lineArr = line.split()
            line_y = []
            current_len = 0
            for item in lineArr:
                tags = getList(item)
                line_y.extend(tags)
                current_len += len(item)
            
            # 验证处理后的字符数是否与标签数匹配
            assert len(processed_chars) == len(line_y), \
                f"Mismatch between processed chars ({len(processed_chars)}) and tags ({len(line_y)}) in line {line_num}: '{''.join(processed_chars)}' vs tags"

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

    # 改进数据划分，同时划分 x_char_types_data
    x_train, x_test, y_train, y_test, x_char_types_train, x_char_types_test = train_test_split(
        x_data, y_data, x_char_types_data, # 添加 x_char_types_data
        test_size=0.2,
        random_state=42,
        stratify=length_groups
    )
    
    with open(SAVE_PATH, 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(char_type_vocab_size, outp) # 保存字符类型词汇大小
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_char_types_train, outp) # 保存训练集字符类型
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)
        pickle.dump(x_char_types_test, outp) # 保存测试集字符类型


if __name__ == "__main__":
    handle_data()
