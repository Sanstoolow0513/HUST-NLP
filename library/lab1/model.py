import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CWS(nn.Module):
    """Chinese Word Segmentation Model using BiLSTM-CRF"""
    
    def __init__(self, vocab_size, tag2id, embedding_dim=100, hidden_dim=400):
        """
        Args:
            vocab_size: 词汇表大小
            tag2id: 标签到ID的映射字典
            embedding_dim: 词向量维度
            hidden_dim: LSTM隐藏层维度
        """
        super(CWS, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)

        # 词嵌入层
        self.word_embeds = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        
        # 双向LSTM层
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # 因为是双向LSTM，所以隐藏单元数除以2
            num_layers=2,     # 2层LSTM
            bidirectional=True,
            batch_first=True,
            dropout=0.5       # 层间dropout
        )

        # 全连接层
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)  # 输出dropout
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # CRF层
        self.crf = CRF(self.tagset_size, batch_first=True)

    def init_hidden(self, batch_size, device):
        """初始化LSTM的隐藏状态
        
        Args:
            batch_size: 当前batch的大小
            device: 设备类型 (cpu/cuda)
            
        Returns:
            (h_0, c_0): 初始化后的隐藏状态和细胞状态
        """
        # 形状: (num_layers * num_directions, batch_size, hidden_dim // 2)
        return (
            torch.zeros(4, batch_size, self.hidden_dim // 2, device=device),
            torch.zeros(4, batch_size, self.hidden_dim // 2, device=device)
        )

    def _get_lstm_features(self, sentence, length):
        """获取LSTM的特征输出
        
        Args:
            sentence: 输入句子张量 (batch_size, seq_len)
            length: 每个句子的实际长度列表
            
        Returns:
            lstm_feats: LSTM输出的特征 (batch_size, seq_len, tagset_size)
        """
        batch_size, seq_len = sentence.size()
        
        # 1. 获取词嵌入 (batch_size, seq_len, embedding_dim)
        embeds = self.word_embeds(sentence)
        
        # 2. 打包变长序列
        packed_embeds = pack_padded_sequence(
            embeds, 
            length, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 3. LSTM前向传播
        hidden = self.init_hidden(batch_size, sentence.device)
        lstm_out, _ = self.lstm(packed_embeds, hidden)
        
        # 4. 解包序列 (batch_size, seq_len, hidden_dim)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # 5. 通过全连接层
        lstm_out = self.dropout(lstm_out)
        lstm_out = torch.relu(self.linear(lstm_out))
        lstm_feats = self.hidden2tag(lstm_out)
        
        return lstm_feats

    def forward(self, sentence, tags, mask, length):
        """模型前向传播
        
        Args:
            sentence: 输入句子 (batch_size, seq_len)
            tags: 真实标签 (batch_size, seq_len)
            mask: 有效token的掩码 (batch_size, seq_len)
            length: 每个句子的实际长度列表
            
        Returns:
            loss: CRF的负对数似然损失
        """
        emissions = self._get_lstm_features(sentence, length)
        loss = -self.crf(emissions, tags, mask, reduction='mean')
        return loss

    def infer(self, sentence, mask, length):
        """推理预测
        
        Args:
            sentence: 输入句子 (batch_size, seq_len)
            mask: 有效token的掩码 (batch_size, seq_len)
            length: 每个句子的实际长度列表
            
        Returns:
            best_tag_list: 预测的标签序列列表
        """
        emissions = self._get_lstm_features(sentence, length)
        return self.crf.decode(emissions, mask)


class Attention(nn.Module):
    """注意力机制层 (可选)"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        """前向传播
        
        Args:
            lstm_out: LSTM输出 (batch_size, seq_len, hidden_dim)
            
        Returns:
            context: 注意力加权后的上下文向量 (batch_size, hidden_dim)
        """
        weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return context
