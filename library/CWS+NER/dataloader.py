import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import re # 新增导入

class Sentence(Dataset):
    # 修改 __init__ 以接收 id2word
    def __init__(self, x, y, id2word, batch_size=10, use_features=False):
        self.x = x
        self.y = y
        self.id2word = id2word # 新增
        self.batch_size = batch_size
        self.use_features = use_features
        # 定义标点符号集合，可以根据需要扩展
        self.punctuations = set(""",.!?;:()[]<>'\"，。！？；：（）【】《》——……、　""") # 新增

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        assert len(self.x[idx]) == len(self.y[idx])

        if self.use_features:
            # 使用真实的id2word映射来提取特征
            features = self.extract_features(self.x[idx])
            return self.x[idx], self.y[idx], features

        return self.x[idx], self.y[idx]

    def extract_features(self, char_ids):
        """提取字符级特征
        1. 是否为数字
        2. 是否为英文字母 (ASCII)
        3. 是否为标点符号
        4. 是否为中文字符
        """
        features = []
        for char_id in char_ids:
            # 处理 padding 或 UNK 等特殊情况
            if char_id >= len(self.id2word) or char_id < 0: # 检查ID是否有效
                 # 对于无效ID（例如padding），特征全为0
                char_feature = [0, 0, 0, 0]
            else:
                char = self.id2word[char_id]
                is_digit = 1 if char.isdigit() else 0
                is_alpha = 1 if char.isalpha() and ord(char) < 128 else 0
                is_punct = 1 if char in self.punctuations else 0
                is_chinese = 1 if '\u4e00' <= char <= '\u9fff' else 0
                char_feature = [is_digit, is_alpha, is_punct, is_chinese]

            features.append(char_feature)
        # 注意：这里返回的是 List[List[int]]，collate_fn会处理成Tensor
        return features # 直接返回列表，collate_fn中转为Tensor

    @staticmethod
    def collate_fn(train_data):
        train_data.sort(key=lambda data: len(data[0]), reverse=True)
        data_length = [len(data[0]) for data in train_data]
        data_x = [torch.LongTensor(data[0]) for data in train_data]
        data_y = [torch.LongTensor(data[1]) for data in train_data]
        # 修改为bool类型
        mask = [torch.ones(l, dtype=torch.bool) for l in data_length]
        data_x = pad_sequence(data_x, batch_first=True, padding_value=0)
        data_y = pad_sequence(data_y, batch_first=True, padding_value=0)
        mask = pad_sequence(mask, batch_first=True, padding_value=0)

        # 检查是否有特征数据
        if len(train_data[0]) > 2:
            # 从原始数据中提取特征列表
            features_list = [torch.FloatTensor(data[2]) for data in train_data]
            # 对特征序列进行填充
            features = pad_sequence(features_list, batch_first=True, padding_value=0.0) # 使用 0.0 填充浮点数
            return data_x, data_y, mask, data_length, features

        return data_x, data_y, mask, data_length


if __name__ == '__main__':
    # test - 需要修改以适应新的 __init__ 和特征提取
    # 假设 datasave.pkl 包含 id2word
    data_path = '../data/datasave.pkl' # 示例路径，请替换为实际NER数据路径
    try:
        with open(data_path, 'rb') as inp:
            word2id = pickle.load(inp)
            id2word = pickle.load(inp)
            tag2id = pickle.load(inp)
            id2tag = pickle.load(inp)
            x_train = pickle.load(inp)
            y_train = pickle.load(inp)
            x_test = pickle.load(inp)
            y_test = pickle.load(inp)
            # 尝试加载特征（如果存在）
            try:
                features_train = pickle.load(inp)
                features_test = pickle.load(inp)
                print("特征数据已加载。")
            except EOFError:
                features_train = None
                features_test = None
                print("未找到特征数据。")


        # 创建包含特征的数据集实例
        # 注意：这里需要传递 id2word
        train_dataset = Sentence(x_train, y_train, id2word, use_features=True)
        train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=Sentence.collate_fn)

        # 迭代数据加载器
        # for batch in train_dataloader:
        #     if len(batch) == 5: # 包含特征
        #         input_ids, labels, masks, lengths, features = batch
        #         print("Input IDs:", input_ids.shape)
        #         print("Labels:", labels.shape)
        #         print("Masks:", masks.shape)
        #         print("Lengths:", lengths)
        #         print("Features:", features.shape)
        #         print("Sample Feature Vector:", features[0, 0, :]) # 打印第一个样本第一个时间步的特征
        #     else: # 不包含特征
        #         input_ids, labels, masks, lengths = batch
        #         print("Input IDs:", input_ids.shape)
        #         print("Labels:", labels.shape)
        #         print("Masks:", masks.shape)
        #         print("Lengths:", lengths)
        #     break # 只处理第一个batch用于测试

    except FileNotFoundError:
        print(f"错误：找不到数据文件 {data_path}")
    except Exception as e:
        print(f"加载或处理数据时出错: {e}")