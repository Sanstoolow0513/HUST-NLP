import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM_CRF(nn.Module):
    """通用的BiLSTM-CRF模型，可用于NER和CWS任务"""
    
    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim, task='cws'):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)
        self.task = task

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

        # CRF层 - 根据任务类型设置标签数量
        if task == 'ner':
            self.crf = CRF(21, batch_first=True)  # NER任务有21个标签
        else:  # cws
            self.crf = CRF(4, batch_first=True)   # CWS任务有4个标签

    def init_hidden(self, batch_size, device):
        """初始化LSTM的隐藏状态"""
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
        
        # 添加额外的处理层
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.linear(lstm_out)
        lstm_out = self.relu(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence, tags, mask, length):
        """模型前向传播"""
        emissions = self._get_lstm_features(sentence, length)
        loss = -self.crf(emissions, tags, mask, reduction='mean')
        return loss

    def infer(self, sentence, mask, length):
        """推理预测"""
        emissions = self._get_lstm_features(sentence, length)
        return self.crf.decode(emissions, mask)
        
    def load_pretrained_embeddings(self, pretrained_embeddings):
        """加载预训练词向量"""
        self.word_embeds.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
