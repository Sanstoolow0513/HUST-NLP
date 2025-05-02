import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class Sentence(Dataset):
    def __init__(self, x, y, batch_size=10, use_char_bigram=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.use_char_bigram = use_char_bigram
        
        # 预计算数据长度，避免重复计算
        self.lengths = [len(item) for item in self.x]
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        assert len(self.x[idx]) == len(self.y[idx])
        return self.x[idx], self.y[idx]
    
    @staticmethod
    def collate_fn(train_data):
        # 按序列长度排序，提高计算效率
        train_data.sort(key=lambda data: len(data[0]), reverse=True)
        data_length = [len(data[0]) for data in train_data]
        data_x = [torch.LongTensor(data[0]) for data in train_data]
        data_y = [torch.LongTensor(data[1]) for data in train_data]
        
        # 使用布尔类型的掩码，更节省内存
        mask = [torch.ones(l, dtype=torch.bool) for l in data_length]
        
        # 使用0作为填充值
        data_x = pad_sequence(data_x, batch_first=True, padding_value=0)
        data_y = pad_sequence(data_y, batch_first=True, padding_value=0)
        mask = pad_sequence(mask, batch_first=True, padding_value=False)
        
        return data_x, data_y, mask, data_length


# 添加数据增强类
class DataAugmentation:
    @staticmethod
    def random_mask(sentence, mask_prob=0.1):
        """随机掩码，模拟BERT的MLM策略"""
        result = sentence.copy()
        for i in range(len(result)):
            if np.random.random() < mask_prob:
                result[i] = 0  # 使用0作为掩码
        return result
    
    @staticmethod
    def random_swap(sentence, swap_prob=0.1):
        """随机交换相邻字符"""
        result = sentence.copy()
        for i in range(len(result) - 1):
            if np.random.random() < swap_prob:
                result[i], result[i+1] = result[i+1], result[i]
        return result


if __name__ == '__main__':
    # test
    with open('data/CWS/data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    train_dataloader = DataLoader(Sentence(x_train, y_train), batch_size=10, shuffle=True, collate_fn=Sentence.collate_fn)

    for input, label, mask, length in train_dataloader:
        print(input, label)
        break