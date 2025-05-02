import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CWS(nn.Module):
    """
    BiLSTM-CRF模型，可用于中文分词(CWS)和命名实体识别(NER)
    """
    # 在 __init__ 中接收 feature_dim
    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim, feature_dim=0, dropout=0.5): # 添加 feature_dim 参数
        super(CWS, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)
        self.feature_dim = feature_dim # 保存特征维度

        # 字符嵌入
        # 注意：词汇表大小可能需要+1来处理UNK或padding，取决于你的数据预处理
        self.word_embeds = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)

        # 添加dropout层
        self.dropout = nn.Dropout(dropout)

        # 如果有额外特征，增加特征嵌入的维度
        lstm_input_dim = embedding_dim + feature_dim # LSTM 输入维度包含特征

        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.crf = CRF(self.tagset_size, batch_first=True)

    def init_hidden(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_dim // 2, device=device),
                torch.randn(2, batch_size, self.hidden_dim // 2, device=device))

    # 修改 _get_lstm_features 以接收和处理 features
    def _get_lstm_features(self, sentence, length, features=None): # 添加 features 参数
        batch_size, seq_len = sentence.size(0), sentence.size(1)

        # idx->embedding
        embeds = self.word_embeds(sentence) # 直接使用 sentence，无需 view().reshape()
        # embeds = self.word_embeds(sentence.view(-1)).reshape(batch_size, seq_len, -1) # 旧代码

        # 应用dropout
        embeds = self.dropout(embeds)

        # 如果有额外特征，拼接到嵌入向量
        if features is not None and self.feature_dim > 0:
            # 确保 features 的数据类型和设备与 embeds 一致
            features = features.to(embeds.device, dtype=embeds.dtype)
            embeds = torch.cat([embeds, features], dim=2) # 在最后一个维度拼接

        # 注意：pack_padded_sequence 需要 length 是 CPU 上的列表或 Tensor
        length_cpu = length
        if isinstance(length, torch.Tensor) and length.device.type != 'cpu':
            length_cpu = length.cpu()

        packed = pack_padded_sequence(embeds, length_cpu, batch_first=True)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence, tags, mask, length, features=None):
        """计算损失"""
        feats = self._get_lstm_features(sentence, length, features)
        loss = -self.crf(feats, tags, mask=mask.bool(), reduction='mean')
        return loss

    def infer(self, sentence, mask, length, features=None):
        """推理，返回预测的标签序列"""
        # 获取LSTM特征
        feats = self._get_lstm_features(sentence, length, features)
        
        # 使用CRF解码
        mask_bool = mask.bool()  # 确保mask是布尔类型
        pred_tags = self.crf.decode(feats, mask_bool)
        
        # 返回预测的标签序列
        return pred_tags