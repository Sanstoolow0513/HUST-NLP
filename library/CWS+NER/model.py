import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CWS(nn.Module):
    """
    BiLSTM-CRF模型，可用于中文分词(CWS)和命名实体识别(NER)
    """
    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim, dropout=0.5, num_layers=2):
        super(CWS, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)
        self.num_layers = num_layers
        
        # 增加dropout防止过拟合
        self.dropout = nn.Dropout(dropout)
        
        # 词嵌入层
        self.word_embeds = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        
        # 增加层数和dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 全连接层
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # CRF层
        self.crf = CRF(self.tagset_size, batch_first=True)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型参数"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2, device=device),
                torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2, device=device))

    def _get_lstm_features(self, sentence, length):
        batch_size, seq_len = sentence.size(0), sentence.size(1)

        # idx->embedding
        embeds = self.word_embeds(sentence)
        embeds = self.dropout(embeds)  # 应用dropout
        embeds = pack_padded_sequence(embeds, length, batch_first=True)

        # LSTM forward
        self.hidden = self.init_hidden(batch_size, sentence.device)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.dropout(lstm_out)  # 应用dropout
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence, tags, mask, length):
        emissions = self._get_lstm_features(sentence, length)
        loss = -self.crf(emissions, tags, mask, reduction='mean')
        return loss

    def infer(self, sentence, mask, length):
        emissions = self._get_lstm_features(sentence, length)
        return self.crf.decode(emissions, mask)