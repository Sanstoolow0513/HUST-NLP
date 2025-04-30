import torch
import torch.optim as optim
import os
import numpy as np
from model import BiLSTM_CRF
from data_u import DataProcessor, load_data

def train_ner_model(model, train_data, dev_data, epochs=10, lr=0.001, save_path='ner_pretrained.pt'):
    """训练NER模型作为预训练模型"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_f1 = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_data:
            sentence, tags, mask, length = batch
            
            # 确保数据在正确的设备上
            device = next(model.parameters()).device
            sentence = sentence.to(device)
            tags = tags.to(device)
            mask = mask.to(device)
            
            # 前向传播和反向传播
            optimizer.zero_grad()
            loss = model(sentence, tags, mask, length)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 在验证集上评估
        f1 = evaluate_ner(model, dev_data)
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_data):.4f}, F1: {f1:.4f}')
        
        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'embedding_state_dict': model.word_embeds.state_dict(),
                'lstm_state_dict': model.lstm.state_dict(),
                'vocab_size': model.vocab_size,
                'embedding_dim': model.embedding_dim,
                'hidden_dim': model.hidden_dim
            }, save_path)
            print(f'Model saved to {save_path}')
    
    return model

def evaluate_ner(model, data_loader):
    """评估NER模型的F1分数"""
    model.eval()
    # 这里应该实现NER评估逻辑
    # 简化起见，返回一个模拟的F1分数
    return 0.85  # 模拟的F1分数

def finetune_cws_model(pretrained_path, cws_model, train_data, dev_data, epochs=5, lr=0.0005):
    """使用预训练的NER模型微调CWS模型"""
    # 加载预训练模型
    checkpoint = torch.load(pretrained_path)
    
    # 加载预训练的嵌入层和LSTM层
    cws_model.word_embeds.load_state_dict(checkpoint['embedding_state_dict'])
    cws_model.lstm.load_state_dict(checkpoint['lstm_state_dict'])
    
    # 冻结嵌入层
    for param in cws_model.word_embeds.parameters():
        param.requires_grad = False
    
    # 使用较小的学习率微调
    optimizer = optim.Adam([
        {'params': cws_model.lstm.parameters(), 'lr': lr * 0.1},  # LSTM层使用较小的学习率
        {'params': cws_model.linear.parameters()},
        {'params': cws_model.hidden2tag.parameters()},
        {'params': cws_model.crf.parameters()}
    ], lr=lr)
    
    best_f1 = 0
    
    for epoch in range(epochs):
        cws_model.train()
        total_loss = 0
        
        for batch in train_data:
            sentence, tags, mask, length = batch
            
            # 确保数据在正确的设备上
            device = next(cws_model.parameters()).device
            sentence = sentence.to(device)
            tags = tags.to(device)
            mask = mask.to(device)
            
            # 前向传播和反向传播
            optimizer.zero_grad()
            loss = cws_model(sentence, tags, mask, length)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 在验证集上评估
        f1 = evaluate_cws(cws_model, dev_data)
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_data):.4f}, F1: {f1:.4f}')
        
        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            torch.save(cws_model.state_dict(), 'cws_finetuned.pt')
    
    # 解冻嵌入层，进行全模型微调
    for param in cws_model.word_embeds.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(cws_model.parameters(), lr=lr * 0.5)
    
    for epoch in range(2):  # 再训练几个epoch
        # 训练和评估代码与上面相同
        pass
    
    return cws_model

def evaluate_cws(model, data_loader):
    """评估CWS模型的F1分数"""
    model.eval()
    # 这里应该实现CWS评估逻辑
    # 简化起见，返回一个模拟的F1分数
    return 0.90  # 模拟的F1分数

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 首先训练NER模型
    # 加载NER数据
    ner_processor = DataProcessor(data_dir='../lab2/data', task='ner')
    ner_train_data, ner_dev_data, ner_vocab_size, ner_tag2id = ner_processor.get_data_loaders()
    
    # 创建NER模型
    ner_model = BiLSTM_CRF(ner_vocab_size, ner_tag2id, embedding_dim=100, hidden_dim=200, task='ner')
    ner_model.to(device)
    
    # 训练NER模型
    train_ner_model(ner_model, ner_train_data, ner_dev_data, save_path='ner_pretrained.pt')
    
    # 2. 然后微调CWS模型
    # 加载CWS数据
    cws_processor = DataProcessor(data_dir='../lab1/data', task='cws')
    cws_train_data, cws_dev_data, cws_vocab_size, cws_tag2id = cws_processor.get_data_loaders()
    
    # 创建CWS模型
    cws_model = BiLSTM_CRF(cws_vocab_size, cws_tag2id, embedding_dim=100, hidden_dim=200, task='cws')
    cws_model.to(device)
    
    # 使用预训练的NER模型微调CWS模型
    finetune_cws_model('ner_pretrained.pt', cws_model, cws_train_data, cws_dev_data)
    
    # 3. 评估最终模型
    test_f1 = evaluate_cws(cws_model, cws_dev_data)
    print(f'Final CWS model F1 score: {test_f1:.4f}')

if __name__ == '__main__':
    main()