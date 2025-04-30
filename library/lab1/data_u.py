import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class DataProcessor:
    """数据处理类，支持CWS和NER任务"""
    
    def __init__(self, data_dir, task='cws', batch_size=64):
        self.data_dir = data_dir
        self.task = task
        self.batch_size = batch_size
        
        # 根据任务类型设置文件名
        if task == 'cws':
            self.train_file = os.path.join(data_dir, 'train.txt')
            self.dev_file = os.path.join(data_dir, 'dev.txt')
            self.test_file = os.path.join(data_dir, 'test.txt')
            # CWS标签: B(开始), M(中间), E(结束), S(单字成词)
            self.tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
        else:  # ner
            self.train_file = os.path.join(data_dir, 'train_data.txt')
            self.dev_file = os.path.join(data_dir, 'dev_data.txt')
            self.test_file = os.path.join(data_dir, 'test_data.txt')
            # NER标签: 根据实际NER任务的标签集合
            self.tag2id = self._load_ner_tags()
        
        # 构建词汇表
        self.word2id = self._build_vocab()
        
    def _load_ner_tags(self):
        """加载NER任务的标签集合"""
        # 这里应该根据实际NER任务的标签集合进行设置
        # 简化起见，返回一个模拟的标签集合
        tags = {f'TAG_{i}': i for i in range(21)}  # 假设有21个标签
        return tags
    
    def _build_vocab(self):
        """构建词汇表"""
        word2id = {'<PAD>': 0, '<UNK>': 1}
        
        with open(self.train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    if self.task == 'cws':
                        # CWS数据格式处理
                        parts = line.strip().split()
                        for word in parts:
                            for char in word:
                                if char not in word2id:
                                    word2id[char] = len(word2id)
                    else:  # ner
                        # NER数据格式处理
                        parts = line.strip().split()
                        if parts:
                            char = parts[0]
                            if char not in word2id:
                                word2id[char] = len(word2id)
        
        return word2id
    
    def _load_data(self, file_path):
        """加载数据"""
        sentences, tags = [], []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if self.task == 'cws':
                # CWS数据加载
                for line in f:
                    if line.strip():
                        sentence = []
                        tag = []
                        words = line.strip().split()
                        for word in words:
                            if len(word) == 1:
                                sentence.append(word)
                                tag.append('S')
                            else:
                                for i, char in enumerate(word):
                                    sentence.append(char)
                                    if i == 0:
                                        tag.append('B')
                                    elif i == len(word) - 1:
                                        tag.append('E')
                                    else:
                                        tag.append('M')
                        sentences.append(sentence)
                        tags.append(tag)
            else:  # ner
                # NER数据加载
                sentence, tag = [], []
                for line in f:
                    line = line.strip()
                    if not line:
                        if sentence:
                            sentences.append(sentence)
                            tags.append(tag)
                            sentence, tag = [], []
                    else:
                        parts = line.split()
                        if len(parts) >= 2:
                            char, label = parts[0], parts[-1]
                            sentence.append(char)
                            tag.append(label)
                if sentence:  # 处理最后一个句子
                    sentences.append(sentence)
                    tags.append(tag)
        
        return sentences, tags
    
    def _convert_to_ids(self, sentences, tags):
        """将文本转换为ID"""
        sentence_ids, tag_ids = [], []
        
        for sentence, tag in zip(sentences, tags):
            s_ids = [self.word2id.get(char, self.word2id['<UNK>']) for char in sentence]
            t_ids = [self.tag2id.get(t, 0) for t in tag]  # 对于未知标签，使用0
            
            sentence_ids.append(s_ids)
            tag_ids.append(t_ids)
        
        return sentence_ids, tag_ids
    
    def get_data_loaders(self):
        """获取数据加载器"""
        # 加载训练集
        train_sentences, train_tags = self._load_data(self.train_file)
        train_sentence_ids, train_tag_ids = self._convert_to_ids(train_sentences, train_tags)
        
        # 加载验证集
        dev_sentences, dev_tags = self._load_data(self.dev_file)
        dev_sentence_ids, dev_tag_ids = self._convert_to_ids(dev_sentences, dev_tags)
        
        # 创建数据集
        train_dataset = TextDataset(train_sentence_ids, train_tag_ids)
        dev_dataset = TextDataset(dev_sentence_ids, dev_tag_ids)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=self.collate_fn
        )
        
        dev_loader = DataLoader(
            dev_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn
        )
        
        return train_loader, dev_loader, len(self.word2id), self.tag2id
    
    def collate_fn(self, batch):
        """数据批处理函数"""
        # 排序句子（按长度降序）
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        sentences, tags = zip(*batch)
        
        # 获取长度
        lengths = [len(s) for s in sentences]
        max_len = max(lengths)
        
        # 填充
        padded_sentences = torch.zeros((len(sentences), max_len), dtype=torch.long)
        padded_tags = torch.zeros((len(tags), max_len), dtype=torch.long)
        mask = torch.zeros((len(sentences), max_len), dtype=torch.bool)
        
        for i, (sentence, tag) in enumerate(zip(sentences, tags)):
            padded_sentences[i, :lengths[i]] = torch.tensor(sentence)
            padded_tags[i, :lengths[i]] = torch.tensor(tag)
            mask[i, :lengths[i]] = 1
        
        return padded_sentences, padded_tags, mask, lengths

class TextDataset(Dataset):
    """文本数据集类"""
    
    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.tags = tags
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx]

def load_data(data_dir, task='cws', batch_size=64):
    """加载数据的便捷函数"""
    processor = DataProcessor(data_dir, task, batch_size)
    return processor.get_data_loaders()