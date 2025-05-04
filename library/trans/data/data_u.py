import codecs
from sklearn.model_selection import train_test_split
import pickle
import random
import os

# --- 常量定义 ---
INPUT_DATA = "train.txt"
SAVE_PATH = "./datasave.pkl"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
TAGS = ['B', 'M', 'E', 'S']  # B: Begin, M: Middle, E: End, S: Single

# --- 辅助函数 ---
def get_tags_for_word(word, tag2id):
    """为单个词生成 BMES 标签序列"""
    if not word:
        return []
    if len(word) == 1:
        return [tag2id['S']]
    elif len(word) == 2:
        return [tag2id['B'], tag2id['E']]
    else:
        tags = [tag2id['B']] + [tag2id['M']] * (len(word) - 2) + [tag2id['E']]
        return tags

# --- 数据增强 ---
def augment_data(sentences, labels, word2id, augmentation_factor=1):
    """
    对数据进行增强。
    - 随机删除字符
    - 随机替换字符 (使用有效字符)
    - 随机交换相邻字符
    """
    augmented_sentences = []
    augmented_labels = []

    # 复制原始数据
    augmented_sentences.extend(sentences)
    augmented_labels.extend(labels)

    valid_char_ids = [id for char, id in word2id.items() if char not in [PAD_TOKEN, UNK_TOKEN]]
    if not valid_char_ids:
        print("警告：词表中没有有效的非特殊字符，无法进行替换增强。")
        return augmented_sentences, augmented_labels # 只返回原始数据

    for _ in range(augmentation_factor): # 控制增强倍数
        for i in range(len(sentences)):
            sentence = sentences[i]
            label = labels[i]
            n = len(sentence)

            # 随机选择一种增强方式
            augmentation_type = random.choice(['delete', 'replace', 'swap'])

            # 1. 随机删除 (句子长度 > 3)
            if augmentation_type == 'delete' and n > 3:
                del_pos = random.randint(1, n - 2) # 不删除首尾
                new_sentence = sentence[:del_pos] + sentence[del_pos+1:]
                new_label = label[:del_pos] + label[del_pos+1:]
                # 简单修复删除后可能产生的非法标签序列 (例如 B E -> B)
                # 更复杂的修复可能需要重新生成标签，这里简化处理
                augmented_sentences.append(new_sentence)
                augmented_labels.append(new_label)

            # 2. 随机替换 (句子长度 > 1)
            elif augmentation_type == 'replace' and n > 1:
                replace_pos = random.randint(0, n - 1)
                original_char_id = sentence[replace_pos]
                
                # 确保有其他字符可选
                possible_replacements = [cid for cid in valid_char_ids if cid != original_char_id]
                if not possible_replacements:
                    continue # 没有可替换的字符

                replace_char_id = random.choice(possible_replacements)
                new_sentence = sentence[:replace_pos] + [replace_char_id] + sentence[replace_pos+1:]
                # 替换不改变标签序列
                augmented_sentences.append(new_sentence)
                augmented_labels.append(label[:]) # 复制标签

            # 3. 随机交换相邻字符 (句子长度 > 1)
            elif augmentation_type == 'swap' and n > 1:
                swap_pos = random.randint(0, n - 2)
                new_sentence = sentence[:]
                new_label = label[:]
                # 交换字符
                new_sentence[swap_pos], new_sentence[swap_pos+1] = new_sentence[swap_pos+1], new_sentence[swap_pos]
                # 交换标签
                new_label[swap_pos], new_label[swap_pos+1] = new_label[swap_pos+1], new_label[swap_pos]
                augmented_sentences.append(new_sentence)
                augmented_labels.append(new_label)

    return augmented_sentences, augmented_labels


# --- 主处理函数 ---
def process_and_save_data(input_file=INPUT_DATA, save_path=SAVE_PATH, test_size=0.1, random_state=42, augment=True, augmentation_factor=1):
    """
    处理原始数据，构建词表，生成标签，进行数据增强（可选），划分数据集，并保存。
    """
    print(f"开始处理数据: {input_file}")

    # 1. 初始化词表和标签映射
    word2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    id2word = [PAD_TOKEN, UNK_TOKEN]
    tag2id = {tag: i for i, tag in enumerate(TAGS)}
    id2tag = {i: tag for i, tag in enumerate(TAGS)}
    next_word_id = 2 # 从2开始分配新词ID

    sentences = [] # 存储句子（字符ID列表）
    labels = []    # 存储标签（标签ID列表）

    # 2. 读取和处理原始文件
    line_count = 0
    processed_count = 0
    try:
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                line = line.strip()
                if not line:
                    continue

                words = line.split() # 按空格分割成词
                sentence_ids = []
                label_ids = []

                for word in words:
                    if not word: continue # 跳过空词（可能由多个空格产生）
                    word_label_ids = get_tags_for_word(word, tag2id)
                    label_ids.extend(word_label_ids)

                    for char in word:
                        if char not in word2id:
                            word2id[char] = next_word_id
                            id2word.append(char)
                            sentence_ids.append(next_word_id)
                            next_word_id += 1
                        else:
                            sentence_ids.append(word2id[char])

                if len(sentence_ids) != len(label_ids):
                     print(f"警告：第 {line_count} 行句子和标签长度不匹配！句子长度: {len(sentence_ids)}, 标签长度: {len(label_ids)}. 跳过此行。")
                     print(f"原文: {line}")
                     continue # 跳过不匹配的行

                sentences.append(sentence_ids)
                labels.append(label_ids)
                processed_count += 1

    except FileNotFoundError:
        print(f"错误：输入文件 {input_file} 未找到！")
        return
    except Exception as e:
        print(f"处理文件 {input_file} 时发生错误: {e}")
        return

    print(f"原始数据处理完成。总行数: {line_count}, 有效处理行数: {processed_count}")
    if not sentences:
        print("错误：未能从文件中加载任何有效数据。")
        return

    print(f"构建词表大小: {len(word2id)}")
    print(f"标签类别: {tag2id}")

    # 打印一个样本检查
    print("\n示例数据:")
    print(f"句子 (IDs): {sentences[0]}")
    print(f"句子 (Chars): {''.join([id2word[id] for id in sentences[0]])}")
    print(f"标签 (IDs): {labels[0]}")
    print(f"标签 (Tags): {' '.join([id2tag[id] for id in labels[0]])}")

    # 3. 数据增强 (可选)
    if augment:
        print(f"\n开始数据增强 (因子: {augmentation_factor})...")
        sentences, labels = augment_data(sentences, labels, word2id, augmentation_factor)
        print(f"数据增强后总样本数: {len(sentences)}")

    # 4. 划分训练集和测试集
    print("\n划分训练集和测试集...")
    x_train, x_test, y_train, y_test = train_test_split(
        sentences, labels, test_size=test_size, random_state=random_state
    )
    print(f"训练集大小: {len(x_train)}, 测试集大小: {len(x_test)}")

    # 5. 保存处理好的数据
    data_to_save = {
        'word2id': word2id,
        'id2word': id2word,
        'tag2id': tag2id,
        'id2tag': id2tag,
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'pad_token': PAD_TOKEN,
        'unk_token': UNK_TOKEN,
        'pad_id': word2id[PAD_TOKEN],
        'unk_id': word2id[UNK_TOKEN]
    }

    try:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"\n处理完成的数据已保存到: {save_path}")
    except Exception as e:
        print(f"保存数据到 {save_path} 时发生错误: {e}")


if __name__ == "__main__":
    # 运行数据处理，启用增强，增强因子为1（即增加一倍数据）
    process_and_save_data(augment=True, augmentation_factor=1)
