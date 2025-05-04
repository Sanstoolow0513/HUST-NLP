import torch
import torch.nn as nn
import math
# import numpy as np # numpy 未在此文件中使用，可以移除

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerSegmenter(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, transformer_ff_dim=512, num_classes=4,
                 num_layers=4, num_heads=8, dropout=0.2, max_len=1024, use_crf=True, padding_idx=0): # 显式加入 padding_idx
        super(TransformerSegmenter, self).__init__()

        self.padding_idx = padding_idx # 使用传入的 padding_idx
        # self.unk_idx = 1 # unk_idx 似乎没有在模型逻辑中使用，可以考虑移除

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.padding_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len)

        # 增加层归一化
        self.layer_norm_emb = nn.LayerNorm(embedding_dim) # 重命名以示区分
        self.dropout_emb = nn.Dropout(dropout) # 在 Embedding 和 Positional Encoding 后增加 Dropout

        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=transformer_ff_dim, # 使用 transformer_ff_dim
            dropout=dropout,
            batch_first=True # !!! 设置 batch_first=True !!!
        )
        # 添加 Encoder 的 LayerNorm
        encoder_norm = nn.LayerNorm(embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers, norm=encoder_norm)

        # 分类层 (直接在 Transformer 输出上操作)
        self.dropout_out = nn.Dropout(dropout)
        # !!! FC 层的输入维度应为 embedding_dim !!!
        self.fc = nn.Linear(embedding_dim, num_classes)

        # 条件随机场层
        self.use_crf = use_crf
        if use_crf:
            # 确保 torchcrf 已安装: pip install pytorch-crf
            try:
                from torchcrf import CRF
                self.crf = CRF(num_classes, batch_first=True)
            except ImportError:
                print("错误：请安装 pytorch-crf 库 (pip install pytorch-crf) 以使用 CRF 功能。")
                self.use_crf = False # 如果未安装，则禁用 CRF

    def forward(self, x, mask=None, labels=None):
        # x: [batch_size, seq_len]

        # 创建 padding mask (如果未提供)
        # mask 应为 True 表示 padding 位置
        if mask is None:
            mask = (x == self.padding_idx)  # [batch_size, seq_len]

        # 嵌入和位置编码
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        x = self.layer_norm_emb(x)
        x = self.dropout_emb(x)

        # !!! TransformerEncoderLayer 设置了 batch_first=True, 无需 permute !!!
        # Transformer编码器
        # src_key_padding_mask 需要 True 表示 padding 位置
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        # x 的输出维度: [batch_size, seq_len, embedding_dim]

        # !!! 移除 LSTM 层 !!!
        # x, _ = self.lstm(x) # [batch_size, seq_len, hidden_dim]

        # 分类层
        x = self.dropout_out(x)
        emissions = self.fc(x)  # [batch_size, seq_len, num_classes]

        # CRF 处理
        if self.use_crf:
            mask_for_crf = ~mask # CRF 需要 True 表示非 padding 位置
            if labels is not None:  # 训练模式
                # 确保标签在有效范围内 (CRF 内部可能不需要，但以防万一)
                # labels = labels.clamp(0, self.crf.num_tags - 1) # torchcrf 似乎不需要 clamp
                # 计算负对数似然损失
                labels_for_crf = labels.clone() # 复制标签
                labels_for_crf[labels == -100] = 0

                log_likelihood = self.crf(emissions, labels_for_crf, mask=mask_for_crf, reduction='mean') # 使用 mean reduction
                return -log_likelihood
            else:  # 推理模式
                # 返回最佳标签序列列表 List[List[int]]
                return self.crf.decode(emissions, mask=mask_for_crf)
        else: # 非 CRF 模式
            # 训练时返回发射分数，由外部计算损失
            # 推理时也返回发射分数，由外部进行 argmax
            return emissions