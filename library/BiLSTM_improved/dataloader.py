import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class Sentence(Dataset):
    # 修改 __init__ 接受 x_char_types
    def __init__(self, x, y, x_char_types, batch_size=10):
        self.x = x
        self.y = y
        self.x_char_types = x_char_types # 添加字符类型数据
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # 确保所有数据长度一致
        assert len(self.x[idx]) == len(self.y[idx]) == len(self.x_char_types[idx])
        # 返回字符类型数据
        return self.x[idx], self.y[idx], self.x_char_types[idx]

    @staticmethod
    def collate_fn(train_data):
        # train_data 现在是 (x, y, char_types) 元组的列表
        train_data.sort(key=lambda data: len(data[0]), reverse=True)
        data_length = [len(data[0]) for data in train_data]
        data_x = [torch.LongTensor(data[0]) for data in train_data]
        data_y = [torch.LongTensor(data[1]) for data in train_data]
        # 处理字符类型数据
        data_char_types = [torch.LongTensor(data[2]) for data in train_data]
        
        mask = [torch.ones(l, dtype=torch.bool) for l in data_length] # 使用 torch.bool
        
        data_x = pad_sequence(data_x, batch_first=True, padding_value=0)
        data_y = pad_sequence(data_y, batch_first=True, padding_value=0) # 标签通常用特定值填充，这里用0可能冲突，但CRF层会处理mask
        # 填充字符类型数据
        data_char_types = pad_sequence(data_char_types, batch_first=True, padding_value=0) # 假设类型0是某种默认或填充类型
        mask = pad_sequence(mask, batch_first=True, padding_value=0) # mask用0填充
        
        # 返回填充后的字符类型数据
        return data_x, data_y, data_char_types, mask, data_length


if __name__ == '__main__':
    # test - 需要修改以加载和传递 char_types 数据
    with open('data/datasave.pkl', 'rb') as inp: # 调整路径
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        char_type_vocab_size = pickle.load(inp) # 加载
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_char_types_train = pickle.load(inp) # 加载
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
        x_char_types_test = pickle.load(inp) # 加载

    # 传递 char_types 给 Sentence Dataset
    train_dataloader = DataLoader(Sentence(x_train, y_train, x_char_types_train), batch_size=10, shuffle=True, collate_fn=Sentence.collate_fn)

    # 解包时包含 char_types
    for input_ids, label, char_types, mask, length in train_dataloader:
        print("Input IDs:", input_ids.shape)
        print("Labels:", label.shape)
        print("Char Types:", char_types.shape) # 打印字符类型
        print("Mask:", mask.shape)
        print("Lengths:", length)
        break