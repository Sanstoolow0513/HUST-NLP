import torch
import torch.nn as nn
# 导入 BertTokenizerFast
from transformers import BertModel, BertTokenizerFast

class BertSegmenter(nn.Module):
    def __init__(self, num_classes, pretrained_model="bert-base-chinese"):
        super(BertSegmenter, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        return logits

class BertTokenizerForSegmentation:
    def __init__(self, pretrained_model="bert-base-chinese"):
        # 使用 BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)

    def encode(self, text, max_length=128, return_tensors="pt"):
        # Fast tokenizer 默认支持 offset_mapping，这里不需要显式传递
        # 但在 Dataset 中调用时需要传递 return_offsets_mapping=True
        return self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors
            # 注意：这里不需要 return_offsets_mapping=True，是在 Dataset 的 __getitem__ 中调用时才需要
        )