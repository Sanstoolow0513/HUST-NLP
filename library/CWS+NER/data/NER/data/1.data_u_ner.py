import codecs
from sklearn.model_selection import train_test_split
import pickle

INPUT_DATA = "train.txt"
TRAIN_DATA = "ner_train.txt"
VALID_DATA = "ner_valid.txt"
SAVE_PATH = "./ner_datasave.pkl"

# 添加BIO到BMES的转换函数
def convert_bio_to_bmes(bio_tags):
    """将BIO标注转换为BMES标注"""
    bmes_tags = []
    i = 0
    while i < len(bio_tags):
        tag = bio_tags[i]
        
        # 处理O标签
        if tag == 'O':
            bmes_tags.append('O')
            i += 1
            continue
            
        # 处理B-开头的标签
        if tag.startswith('B-'):
            entity_type = tag[2:]  # 提取实体类型
            # 查找该实体的结束位置
            j = i + 1
            while j < len(bio_tags) and bio_tags[j].startswith('I-') and bio_tags[j][2:] == entity_type:
                j += 1
                
            entity_length = j - i
            
            if entity_length == 1:  # 单字实体
                bmes_tags.append('S-' + entity_type)
            else:  # 多字实体
                bmes_tags.append('B-' + entity_type)
                for k in range(i+1, j-1):
                    bmes_tags.append('M-' + entity_type)
                bmes_tags.append('E-' + entity_type)
            
            i = j
        else:
            # 处理异常情况（如I-开头但前面没有B-）
            if tag.startswith('I-'):
                entity_type = tag[2:]
                bmes_tags.append('S-' + entity_type)  # 将孤立的I-标签视为S-
            else:
                bmes_tags.append(tag)  # 保留其他标签
            i += 1
            
    return bmes_tags

# 首先收集所有BIO标签并转换为BMES标签
all_bio_tags = set()
all_bmes_tags = set()

with open('ner_train.txt', 'r', encoding="utf-8") as f:
    current_sentence_tags = []
    for line in f:
        line = line.strip()
        if not line:
            # 处理一个完整句子的标签
            if current_sentence_tags:
                bmes_tags = convert_bio_to_bmes(current_sentence_tags)
                all_bmes_tags.update(bmes_tags)
                current_sentence_tags = []
            continue
        
        try:
            tag = line.split(' ')[1]
            all_bio_tags.add(tag)
            current_sentence_tags.append(tag)
        except:
            pass
    
    # 处理最后一个句子
    if current_sentence_tags:
        bmes_tags = convert_bio_to_bmes(current_sentence_tags)
        all_bmes_tags.update(bmes_tags)

# 创建id2tag和tag2id
id2tag = list(all_bmes_tags)
print("转换后的BMES标签:", id2tag)
tag2id = {}
for i, label in enumerate(id2tag):
    tag2id[label] = i

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

def handle_data():
    '''
    处理数据，并保存至savepath
    :return:
    '''
    outp = open(SAVE_PATH, 'wb')

    x_train = []
    y_train = []
    x_valid = []
    y_valid = []

    wordnum = 0
    with open(TRAIN_DATA, 'r', encoding="utf-8") as ifp:
        line_x = []
        line_y_bio = []  # 临时存储BIO标签
        for line in ifp:
            line = line.strip()
            if not line:
                # 转换整个句子的标签
                if line_x and line_y_bio:
                    line_y_bmes = convert_bio_to_bmes(line_y_bio)
                    x_train.append(line_x)
                    y_train.append([tag2id[tag] for tag in line_y_bmes])
                line_x = []
                line_y_bio = []
                continue
            line = line.split(' ')
            if line[0] in id2word:
                line_x.append(word2id[line[0]])
            else:
                id2word.append(line[0])
                word2id[line[0]] = wordnum
                line_x.append(wordnum)
                wordnum = wordnum + 1
            line_y_bio.append(line[1])  # 存储原始BIO标签
            
    # 对验证集也进行BIO到BMES的转换
    with open(VALID_DATA, 'r', encoding="utf-8") as ifp:
        line_x = []
        line_y_bio = []  # 临时存储BIO标签
        for line in ifp:
            line = line.strip()
            if not line:
                # 转换整个句子的标签
                if line_x and line_y_bio:
                    line_y_bmes = convert_bio_to_bmes(line_y_bio)
                    x_valid.append(line_x)
                    y_valid.append([tag2id[tag] for tag in line_y_bmes])
                line_x = []
                line_y_bio = []
                continue
            line = line.split(' ')
            if line[0] in id2word:
                line_x.append(word2id[line[0]])
            else:
                id2word.append(line[0])
                word2id[line[0]] = wordnum
                line_x.append(wordnum)
                wordnum = wordnum + 1
            line_y_bio.append(line[1])  # 存储原始BIO标签

    print(x_train[0])
    print([id2word[i] for i in x_train[0]])
    print(y_train[0])
    print([id2tag[i] for i in y_train[0]])
    
    pickle.dump(word2id, outp)
    pickle.dump(id2word, outp)
    pickle.dump(tag2id, outp)
    pickle.dump(id2tag, outp)
    pickle.dump(x_train, outp)
    pickle.dump(y_train, outp)
    pickle.dump(x_valid, outp)
    pickle.dump(y_valid, outp)

    outp.close()

if __name__ == "__main__":
    handle_data()
