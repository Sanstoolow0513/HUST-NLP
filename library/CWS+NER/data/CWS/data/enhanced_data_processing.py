import codecs
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split

# 配置
INPUT_DATA = "train_clean.txt"  # 使用预处理后的数据
SAVE_PATH = "./enhanced_datasave.pkl"
id2tag = ['B', 'M', 'E', 'S']
tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
word2id = {}
id2word = []

# 字符特征提取
# 字符特征提取
def extract_char_features(char):
    """提取单个字符的特征"""
    features = []
    
    # 是否为数字
    features.append(1 if char.isdigit() else 0)
    
    # 是否为英文字母
    features.append(1 if char.isalpha() and ord(char) < 128 else 0)
    
    # 是否为标点符号
    punctuations = set(""",.!?;:()[]<>'\"，。！？；：（）【】《》——……、　""")
    features.append(1 if char in punctuations else 0)
    
    # 是否为中文字符
    features.append(1 if '\u4e00' <= char <= '\u9fff' else 0)
    
    return features

def getList(input_str):
    """单个分词转换为tag序列"""
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

def data_augmentation(x_data, y_data, char_features, id2word, augment_ratio=0.3):
    """增强数据集"""
    augmented_x = []
    augmented_y = []
    augmented_features = []
    
    # 复制原始数据
    augmented_x.extend(x_data)
    augmented_y.extend(y_data)
    augmented_features.extend(char_features)
    
    num_to_augment = int(len(x_data) * augment_ratio)
    indices = np.random.choice(len(x_data), num_to_augment, replace=False)
    
    for idx in indices:
        # 随机删除
        if len(x_data[idx]) > 5:
            new_x = x_data[idx].copy()
            new_y = y_data[idx].copy()
            new_features = char_features[idx].copy()
            
            del_pos = random.randint(1, len(new_x) - 2)
            del new_x[del_pos]
            del new_y[del_pos]
            del new_features[del_pos]
            
            augmented_x.append(new_x)
            augmented_y.append(new_y)
            augmented_features.append(new_features)
        
        # 随机替换
        if len(x_data[idx]) > 3:
            new_x = x_data[idx].copy()
            new_features = char_features[idx].copy()
            
            replace_pos = random.randint(0, len(new_x) - 1)
            replace_id = random.choice(list(range(len(id2word))))
            
            new_x[replace_pos] = replace_id
            # 更新特征
            if replace_id < len(id2word):
                char = id2word[replace_id]
                new_features[replace_pos] = extract_char_features(char)
            
            augmented_x.append(new_x)
            augmented_y.append(y_data[idx].copy())
            augmented_features.append(new_features)
    
    return augmented_x, augmented_y, augmented_features

def handle_data():
    """处理数据并保存"""
    x_data = []
    y_data = []
    char_features = []  # 存储字符特征
    wordnum = 0
    
    with open(INPUT_DATA, 'r', encoding="utf-8") as ifp:
        for line in ifp:
            line = line.strip()
            if not line:
                continue
            
            line_x = []
            line_features = []
            
            for i in range(len(line)):
                if line[i] == " ":
                    continue
                
                # 处理字符ID
                if line[i] in id2word:
                    line_x.append(word2id[line[i]])
                else:
                    id2word.append(line[i])
                    word2id[line[i]] = wordnum
                    line_x.append(wordnum)
                    wordnum += 1
                
                # 提取字符特征
                line_features.append(extract_char_features(line[i]))
            
            x_data.append(line_x)
            char_features.append(line_features)
            
            # 处理标签
            lineArr = line.split()
            line_y = []
            for item in lineArr:
                line_y.extend(getList(item))
            y_data.append(line_y)
    
    print(f"原始数据样本数: {len(x_data)}")
    
    # 数据增强
    x_data, y_data, char_features = data_augmentation(x_data, y_data, char_features, id2word)
    print(f"数据增强后样本数: {len(x_data)}")
    
    # 划分训练集和测试集
    indices = list(range(len(x_data)))
    train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=43)
    
    x_train = [x_data[i] for i in train_indices]
    y_train = [y_data[i] for i in train_indices]
    features_train = [char_features[i] for i in train_indices]
    
    x_test = [x_data[i] for i in test_indices]
    y_test = [y_data[i] for i in test_indices]
    features_test = [char_features[i] for i in test_indices]
    
    # 保存处理后的数据
    with open(SAVE_PATH, 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)
        pickle.dump(features_train, outp)
        pickle.dump(features_test, outp)
    
    print(f"数据处理完成，已保存到 {SAVE_PATH}")

if __name__ == "__main__":
    handle_data()