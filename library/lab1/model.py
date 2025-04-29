import torch
import torch.nn as nn
import torch.nn.functional as F  # 确保导入了F模块
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CWS(nn.Module):
    """Chinese Word Segmentation Model using BiLSTM-CRF"""
    
    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim):
        super(CWS, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)

        # 使用预训练词向量或增加embedding维度
        self.word_embeds = nn.Embedding(vocab_size + 1, embedding_dim)
        
        # 增加LSTM层数和添加dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.2)
        
        # 添加额外的线性层和dropout
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.crf = CRF(4, batch_first=True)

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
        batch_size, seq_len = sentence.size(0), sentence.size(1)

        # idx->embedding
        embeds = self.word_embeds(sentence.view(-1)).reshape(batch_size, seq_len, -1)
        embeds = pack_padded_sequence(embeds, length, batch_first=True)

        # LSTM forward
        self.hidden = self.init_hidden(batch_size, sentence.device)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # 使用注意力机制
        attention = Attention(self.hidden_dim)
        # 将attention模块移到与输入相同的设备上
        attention = attention.to(sentence.device)
        # 确保hidden张量在正确的设备上
        hidden_for_attn = self.hidden[0].transpose(0, 1).contiguous().view(batch_size, -1)
        attention_weights = attention(hidden_for_attn, lstm_out)
        
        # 添加额外的处理层
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.linear(lstm_out)
        lstm_out = self.relu(lstm_out)
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
    """注意力机制层"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_dim]
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # 计算注意力权重 - 确保在正确的设备上创建张量
        attn_energies = torch.zeros(batch_size, seq_len, device=hidden.device)
        
        for i in range(seq_len):
            attn_energies[:, i] = self._score(hidden, encoder_outputs[:, i])
        
        # 归一化注意力权重
        return F.softmax(attn_energies, dim=1)
    
    def _score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        energy = torch.tanh(energy)
        energy = torch.sum(self.v * energy, dim=1)
        return energy
