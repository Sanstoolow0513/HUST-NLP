import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CWS(nn.Module):
    """Chinese Word Segmentation Model using BiLSTM-CRF"""
    
    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim, char_type_vocab_size, char_type_embedding_dim=20): 
        super(CWS, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)
        self.char_type_embedding_dim = char_type_embedding_dim 

        self.word_embeds = nn.Embedding(vocab_size + 1, embedding_dim) 
        self.char_type_embeds = nn.Embedding(char_type_vocab_size, char_type_embedding_dim)
        
        lstm_input_dim = embedding_dim + char_type_embedding_dim
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim // 2, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.2)
        
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True) 

    def init_hidden(self, batch_size, device):
        """初始化LSTM的隐藏状态
        
        Args:
            batch_size: 当前batch的大小
            device: 设备类型 (cpu/cuda)
            
        Returns:
            (h_0, c_0): 初始化后的隐藏状态和细胞状态
        """
        return (
            torch.zeros(4, batch_size, self.hidden_dim // 2, device=device),
            torch.zeros(4, batch_size, self.hidden_dim // 2, device=device)
        )

    def _get_lstm_features(self, sentence, char_types, length): # 添加 char_types 参数
        batch_size, seq_len = sentence.size(0), sentence.size(1)

        embeds = self.word_embeds(sentence) # No need for view/reshape if batch_first=True
        type_embeds = self.char_type_embeds(char_types)
        
        combined_embeds = torch.cat((embeds, type_embeds), dim=2)
        
        packed_embeds = pack_padded_sequence(combined_embeds, length, batch_first=True, enforce_sorted=False) # enforce_sorted=False if dataloader doesn't guarantee sorting

        self.hidden = self.init_hidden(batch_size, sentence.device)
        lstm_out, self.hidden = self.lstm(packed_embeds, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len) # Ensure output length matches input seq_len
        


        lstm_out = self.dropout(lstm_out)
        lstm_out = self.linear(lstm_out)
        lstm_out = self.relu(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence, tags, char_types, mask, length): 
        emissions = self._get_lstm_features(sentence, char_types, length) 
        mask = mask.bool()
        loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
        return loss

    def infer(self, sentence, char_types, mask, length): 
        emissions = self._get_lstm_features(sentence, char_types, length) 
        mask = mask.bool()
        return self.crf.decode(emissions, mask=mask)


class Attention(nn.Module):
    """注意力机制层"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, 1) 

    def forward(self, lstm_outputs): 
        energy = torch.tanh(self.attn(lstm_outputs)) 
        weights = F.softmax(energy, dim=1)
        context = weights * lstm_outputs 
        return weights 

