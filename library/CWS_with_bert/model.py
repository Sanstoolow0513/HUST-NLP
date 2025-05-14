import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertTokenizer


class CWS(nn.Module):

    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim):
        super(CWS, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)

        self.word_embeds = nn.Embedding(vocab_size + 1, embedding_dim) # vocab_size + 1 for padding?

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.crf = CRF(self.tagset_size, batch_first=True) # Corrected line

    def init_hidden(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_dim // 2, device=device),
                torch.randn(2, batch_size, self.hidden_dim // 2, device=device))

    def _get_lstm_features(self, sentence, length):
        batch_size, seq_len = sentence.size(0), sentence.size(1)

        # idx->embedding
        embeds = self.word_embeds(sentence.view(-1)).reshape(batch_size, seq_len, -1)
        embeds = pack_padded_sequence(embeds, length, batch_first=True)

        # LSTM forward
        self.hidden = self.init_hidden(batch_size, sentence.device)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence, tags, mask, length):
        emissions = self._get_lstm_features(sentence, length)
        loss = -self.crf(emissions, tags, mask, reduction='mean')
        return loss

    def infer(self, sentence, mask, length):
        emissions = self._get_lstm_features(sentence, length)
        return self.crf.decode(emissions, mask)


class BertCWS(nn.Module):
    """使用BERT预训练模型的中文分词模型"""
    
    def __init__(self, tag2id, bert_model_name='bert-base-chinese', hidden_dim=768, dropout=0.1):
        super(BertCWS, self).__init__()
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)
        self.o_tag_id = tag2id.get('O') # 获取O标签的ID
        if self.o_tag_id is None:
            # 如果数据中没有'O'标签，则可能需要抛出错误或进行默认设置
            # 这里假设'O'标签必须存在于tag2id中，由data_u.py保证
            raise ValueError("'O' tag not found in tag2id. Please ensure it is added during data preprocessing.")
            
        self.hidden_dim = hidden_dim
        
        # 加载BERT模型
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 分类层
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, self.tagset_size)
        
        # CRF层
        self.crf = CRF(self.tagset_size, batch_first=True)
    
    def _get_bert_features(self, input_ids, attention_mask):
        """获取BERT特征"""
        # BERT forward
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        return emissions
    
    def forward(self, input_ids, tags, attention_mask, token_type_ids=None):
        """前向传播，计算损失"""
        emissions = self._get_bert_features(input_ids, attention_mask)
        
        crf_tags = tags.clone()
        # 将忽略标签（通常为-100）替换为'O'标签的ID
        crf_tags[crf_tags < 0] = self.o_tag_id 
        
        # 确保所有标签都在有效范围内 (钳位操作)
        crf_tags = torch.clamp(crf_tags, 0, self.tagset_size - 1)
        
        crf_mask = attention_mask.clone().byte()
        
        # 确保CLS令牌的掩码为1，如果它被传递给CRF的话
        # 根据CRF库的期望，可能不需要此操作，如果CRF能正确处理基于mask的序列长度
        # if crf_mask.shape[1] > 0:
        #     crf_mask[:, 0] = 1 

        loss = -self.crf(emissions, crf_tags, crf_mask, reduction='mean')
        return loss
    
    def infer(self, input_ids, attention_mask, token_type_ids=None):
        """推理，返回预测的标签序列"""
        emissions = self._get_bert_features(input_ids, attention_mask)
        return self.crf.decode(emissions, attention_mask.byte())


class BertBiLSTMCRF(nn.Module):
    """使用BERT+BiLSTM+CRF的中文分词模型"""
    def __init__(self, tag2id, bert_model_name='bert-base-chinese', 
                 lstm_hidden_dim=256, lstm_layers=1, dropout_rate=0.1,
                 ner_tagset_size=0, ner_embedding_dim=0): # Added NER params
        super(BertBiLSTMCRF, self).__init__()
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)
        self.o_tag_id = tag2id.get('O')
        if self.o_tag_id is None:
            raise ValueError("'O' tag not found in tag2id. Please ensure it is added during data preprocessing.")

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm_hidden_dim = lstm_hidden_dim
        self.bert_hidden_dim = self.bert.config.hidden_size

        # NER Embeddings (optional)
        self.ner_embedding_dim = ner_embedding_dim
        self.ner_tagset_size = ner_tagset_size
        if self.ner_tagset_size > 0 and self.ner_embedding_dim > 0:
            self.ner_embeddings = nn.Embedding(self.ner_tagset_size, self.ner_embedding_dim, padding_idx=0) # Assuming 0 can be a padding_idx for NER tags if needed
            lstm_input_size = self.bert_hidden_dim + self.ner_embedding_dim
        else:
            self.ner_embeddings = None
            lstm_input_size = self.bert_hidden_dim

        self.lstm = nn.LSTM(input_size=lstm_input_size, 
                              hidden_size=lstm_hidden_dim // 2, # BiLSTM所以hidden_size减半
                              num_layers=lstm_layers, 
                              bidirectional=True, 
                              batch_first=True)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden2tag = nn.Linear(lstm_hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)

    def _get_features(self, input_ids, attention_mask, ner_labels=None):
        """获取经过BERT和BiLSTM的特征"""
        # BERT特征
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, bert_hidden_size]
        sequence_output = self.dropout(sequence_output)
        
        # NER特征 (optional)
        if self.ner_embeddings is not None and ner_labels is not None:
            # Ensure ner_labels are non-negative before passing to embedding layer
            # Values like -100 (ignore_index) should be handled, e.g., mapped to a padding_idx or a specific 'O' NER tag index if not already.
            # Here, we assume ner_labels are already processed to be valid indices for nn.Embedding.
            # If -100 is present and padding_idx is not set or not -100, it will cause an error.
            # A common practice is to map -100 to 0 if 0 is the padding_idx for NER embeddings.
            ner_labels_for_embedding = ner_labels.clone()
            ner_labels_for_embedding[ner_labels_for_embedding < 0] = 0 # Map ignored NER labels to padding_idx 0
            ner_embeds = self.ner_embeddings(ner_labels_for_embedding) # [batch_size, seq_len, ner_embedding_dim]
            ner_embeds = self.dropout(ner_embeds) # Apply dropout to NER embeddings as well
            sequence_output = torch.cat([sequence_output, ner_embeds], dim=-1) # Concatenate

        # BiLSTM特征
        lstm_output, _ = self.lstm(sequence_output) # [batch_size, seq_len, lstm_hidden_dim]
        lstm_output = self.dropout(lstm_output)
        
        # 线性层转换到标签空间
        emissions = self.hidden2tag(lstm_output) # [batch_size, seq_len, tagset_size]
        return emissions

    def forward(self, input_ids, tags, attention_mask, token_type_ids=None, ner_labels=None): # Added ner_labels
        """前向传播，计算损失"""
        emissions = self._get_features(input_ids, attention_mask, ner_labels=ner_labels)
        
        crf_tags = tags.clone()
        crf_tags[crf_tags < 0] = self.o_tag_id # 处理忽略标签
        crf_tags = torch.clamp(crf_tags, 0, self.tagset_size - 1) # 确保标签有效
        
        crf_mask = attention_mask.clone().byte()
        loss = -self.crf(emissions, crf_tags, mask=crf_mask, reduction='mean')
        return loss

    def infer(self, input_ids, attention_mask, token_type_ids=None, ner_labels=None): # Added ner_labels
        """推理，返回预测的标签序列"""
        emissions = self._get_features(input_ids, attention_mask, ner_labels=ner_labels)
        return self.crf.decode(emissions, mask=attention_mask.byte())
