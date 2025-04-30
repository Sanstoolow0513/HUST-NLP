import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from model import BiLSTM_CRF
from data_u import load_data

def evaluate_model(model, data_loader):
    """评估模型性能"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            sentence, tags, mask, length = batch
            
            # 确保数据在正确的设备上
            device = next(model.parameters()).device
            sentence = sentence.to(device)
            tags = tags.to(device)
            mask = mask.to(device)
            
            # 预测
            pred = model.infer(sentence, mask, length)
            
            # 计算准确率
            for i, length_i in enumerate(length):
                correct += sum(p == t.item() for p, t in zip(pred[i][:length_i], tags[i][:length_i]))
                total += length_i
    
    return correct / total

def compare_models(pretrained_model_path, data_dir, embedding_dim=100, hidden_dim=200):
    """比较使用预训练和不使用预训练的模型性能"""
    # 加载数据
    _, dev_data, vocab_size, tag2id = load_data(data_dir, task='cws')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建不使用预训练的模型
    model_no_pretrain = BiLSTM_CRF(vocab_size, tag2id, embedding_dim, hidden_dim, task='cws')
    model_no_pretrain.to(device)
    
    # 创建使用预训练的模型
    model_pretrain = BiLSTM_CRF(vocab_size, tag2id, embedding_dim, hidden_dim, task='cws')
    
    # 加载预训练模型
    checkpoint = torch.load(pretrained_model_path)
    
    # 加载预训练的嵌入层和LSTM层
    model_pretrain.word_embeds.load_state_dict(checkpoint['embedding_state_dict'])
    model_pretrain.lstm.load_state_dict(checkpoint['lstm_state_dict'])
    
    model_pretrain.to(device)
    
    # 加载微调后的模型
    model_finetuned = BiLSTM_CRF(vocab_size, tag2id, embedding_dim, hidden_dim, task='cws')
    model_finetuned.load_state_dict(torch.load('cws_finetuned.pt'))
    model_finetuned.to(device)
    
    # 评估模型
    acc_no_pretrain = evaluate_model(model_no_pretrain, dev_data)
    acc_pretrain = evaluate_model(model_pretrain, dev_data)
    acc_finetuned = evaluate_model(model_finetuned, dev_data)
    
    print(f'Accuracy without pretraining: {acc_no_pretrain:.4f}')
    print(f'Accuracy with pretraining (no fine-tuning): {acc_pretrain:.4f}')
    print(f'Accuracy with pretraining and fine-tuning: {acc_finetuned:.4f}')
    
    # 绘制对比图
    labels = ['No Pretraining', 'With Pretraining\n(No Fine-tuning)', 'With Pretraining\nand Fine-tuning']
    accuracies = [acc_no_pretrain, acc_pretrain, acc_finetuned]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, accuracies, color=['blue', 'orange', 'green'])
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)
    
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

if __name__ == '__main__':
    compare_models('ner_pretrained.pt', '../lab1/data')