import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CWS(nn.Module):
    """Chinese Word Segmentation Model using BiLSTM-CRF"""
    
    # 修改 __init__ 接收 char_type 相关参数
    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim, char_type_vocab_size, char_type_embedding_dim=20): # 添加参数
        super(CWS, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)
        self.char_type_embedding_dim = char_type_embedding_dim # 保存字符类型嵌入维度

        self.word_embeds = nn.Embedding(vocab_size + 1, embedding_dim) # +1 for potential OOV handling if needed later
        # 添加字符类型嵌入层
        self.char_type_embeds = nn.Embedding(char_type_vocab_size, char_type_embedding_dim)
        
        # 调整LSTM输入维度
        lstm_input_dim = embedding_dim + char_type_embedding_dim
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim // 2, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.2)
        
        # 添加额外的线性层和dropout
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True) # Use self.tagset_size

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

    # 修改 _get_lstm_features 接收并使用 char_types
    def _get_lstm_features(self, sentence, char_types, length): # 添加 char_types 参数
        batch_size, seq_len = sentence.size(0), sentence.size(1)

        # 字嵌入
        embeds = self.word_embeds(sentence) # No need for view/reshape if batch_first=True
        # 字符类型嵌入
        type_embeds = self.char_type_embeds(char_types)
        
        # 拼接嵌入
        combined_embeds = torch.cat((embeds, type_embeds), dim=2)
        
        # 打包序列
        packed_embeds = pack_padded_sequence(combined_embeds, length, batch_first=True, enforce_sorted=False) # enforce_sorted=False if dataloader doesn't guarantee sorting

        # LSTM forward
        self.hidden = self.init_hidden(batch_size, sentence.device)
        lstm_out, self.hidden = self.lstm(packed_embeds, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len) # Ensure output length matches input seq_len
        
        # --- Attention Mechanism Application (Needs Correction/Refinement) ---
        # The current attention code calculates weights but doesn't use them.
        # This part needs to be fixed or redesigned in a later step.
        # For now, we bypass the faulty attention application.
        # attention = Attention(self.hidden_dim).to(sentence.device)
        # hidden_for_attn = self.hidden[0].transpose(0, 1).contiguous().view(batch_size, -1) # This likely needs change
        # attention_weights = attention(hidden_for_attn, lstm_out) # This call might be incorrect
        # --- End Attention ---

        # 后续处理层
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.linear(lstm_out)
        lstm_out = self.relu(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    # 修改 forward 接收并传递 char_types
    def forward(self, sentence, tags, char_types, mask, length): # 添加 char_types 参数
        emissions = self._get_lstm_features(sentence, char_types, length) # 传递 char_types
        # Ensure mask is boolean for CRF layer
        mask = mask.bool()
        loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
        return loss

    # 修改 infer 接收并传递 char_types
    def infer(self, sentence, char_types, mask, length): # 添加 char_types 参数
        emissions = self._get_lstm_features(sentence, char_types, length) # 传递 char_types
        # Ensure mask is boolean for CRF layer
        mask = mask.bool()
        return self.crf.decode(emissions, mask=mask)


class Attention(nn.Module):
    """注意力机制层"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        # Simplified attention: linear layer to compute alignment scores
        self.attn = nn.Linear(hidden_dim, 1) 
        # self.v = nn.Parameter(torch.rand(hidden_dim)) # Original complex attention parameter

    def forward(self, lstm_outputs): # Changed signature - simpler attention over outputs
        # lstm_outputs: [batch_size, seq_len, hidden_dim]
        
        # Calculate energy (alignment scores)
        # energy shape: [batch_size, seq_len, 1]
        energy = torch.tanh(self.attn(lstm_outputs)) 
        
        # Calculate weights (softmax over sequence length)
        # weights shape: [batch_size, seq_len, 1]
        weights = F.softmax(energy, dim=1)
        
        # Calculate context vector (weighted sum)
        # context shape: [batch_size, seq_len, hidden_dim] 
        # (element-wise multiplication broadcasts weights)
        context = weights * lstm_outputs 
        
        # Return weights or context depending on how you want to use it
        # Often, context is summed over seq_len or used alongside lstm_outputs
        return weights # Returning weights for now, application needs integration

    # Original _score method (part of a different attention type)
    # def _score(self, hidden, encoder_output):
    #     energy = self.attn(encoder_output)
    #     energy = torch.tanh(energy)
    #     energy = torch.sum(self.v * energy, dim=1)
    #     return energy
