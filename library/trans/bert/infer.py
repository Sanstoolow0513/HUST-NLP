import os
import sys
import torch
import numpy as np
import pickle  # 添加pickle导入

# 添加父目录到路径，以便导入model.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 不再直接从data_u.py导入，而是从pickle文件加载
# from data.data_u import id2tag, tag2id, word2id, id2word
from model import BertSegmenter, BertTokenizerForSegmentation

class BertSegmenterInference:
    def __init__(self, model_path, data_pkl_path="../data/datasave.pkl", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # 从pkl文件加载映射关系
        try:
            with open(data_pkl_path, 'rb') as f:
                self.word2id = pickle.load(f)
                self.id2word = pickle.load(f)
                self.tag2id = pickle.load(f)
                self.id2tag = pickle.load(f)
                # 可以选择性加载 x_train, y_train, x_test, y_test，如果需要的话
                # x_train = pickle.load(f)
                # y_train = pickle.load(f)
                # x_test = pickle.load(f)
                # y_test = pickle.load(f)
        except FileNotFoundError:
            print(f"错误：无法找到数据文件 {data_pkl_path}。请确保路径正确并已运行数据处理脚本。")
            sys.exit(1)
        except Exception as e:
            print(f"加载数据文件 {data_pkl_path} 时出错: {e}")
            sys.exit(1)
            
        # 初始化模型
        num_classes = len(self.id2tag)
        self.model = BertSegmenter(num_classes=num_classes).to(self.device)
        
        # 加载模型权重
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"错误：无法找到模型文件 {model_path}。请确保路径正确并已成功训练模型。")
            sys.exit(1)
        except RuntimeError as e:
            print(f"加载模型状态字典时出错: {e}")
            print("请确保模型定义与保存的模型权重匹配，特别是 vocab_size 和 num_classes。")
            sys.exit(1)
        except Exception as e:
            print(f"加载模型文件 {model_path} 时发生未知错误: {e}")
            sys.exit(1)
            
        self.model.eval()
        
        # 初始化tokenizer
        self.tokenizer = BertTokenizerForSegmentation()
    
    def segment(self, text):
        """对文本进行分词"""
        # 使用BERT tokenizer处理文本
        encoding = self.tokenizer.encode(
            text,
            max_length=512,  # 使用更长的序列长度以适应更长的文本
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            preds = torch.argmax(outputs, dim=2)[0].cpu().numpy()
        
        # 根据预测标签进行分词，注意BERT的特殊标记
        words = []
        current_word = ""
        
        # 跳过[CLS]标记，从第一个实际字符开始
        for i, (token_id, tag_id) in enumerate(zip(input_ids[0][1:].cpu().numpy(), preds[1:])):
            # 如果是[SEP]标记或者padding，则停止处理
            if token_id == self.tokenizer.tokenizer.sep_token_id or token_id == self.tokenizer.tokenizer.pad_token_id:
                break
                
            # 获取字符和标签
            char = self.tokenizer.tokenizer.convert_ids_to_tokens([token_id])[0]
            
            # 检查tag_id是否在有效范围内
            if tag_id < 0 or tag_id >= len(self.id2tag):
                print(f"警告：无效的tag_id {tag_id}在位置{i}。跳过此字符。")
                continue
                
            tag = self.id2tag[tag_id]
            
            # 处理BERT分词器产生的特殊标记
            if char.startswith("##"):
                char = char[2:]  # 去掉##前缀
            
            if tag == 'B':  # 词的开始
                if current_word:
                    words.append(current_word)
                current_word = char
            elif tag == 'M':  # 词的中间
                # 确保current_word不是空的，M不应该出现在词首
                if current_word:
                    current_word += char
                else:
                    # 处理M出现在词首的异常情况，可以当作S处理
                    words.append(char)
                    current_word = ""
            elif tag == 'E':  # 词的结束
                # 确保current_word不是空的，E不应该出现在词首
                if current_word:
                    current_word += char
                    words.append(current_word)
                    current_word = ""
                else:
                    # 处理E出现在词首的异常情况，可以当作S处理
                    words.append(char)
                    current_word = ""
            elif tag == 'S':  # 单字词
                if current_word:
                    words.append(current_word)
                words.append(char)
                current_word = ""
            else:
                print(f"警告：未知的标签'{tag}'在位置{i}。")
                # 可以选择将此字符视为单字词
                if current_word:
                    words.append(current_word)
                words.append(char)
                current_word = ""
        
        # 处理最后一个词
        if current_word:
            words.append(current_word)
        
        return words

def main():
    # 加载模型
    model_path = "bert_segmenter.pth"
    data_pkl_path = "../data/datasave.pkl"
    segmenter = BertSegmenterInference(model_path, data_pkl_path)
    
    # 从文件读取测试文本
    test_file = "../infer_apply_test.txt"
    output_file = "segmentation_results_bert.txt"
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        with open(output_file, 'w', encoding='utf-8') as outf:
            # 对每行文本进行分词
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    outf.write("\n")
                    continue
                
                words = segmenter.segment(line)
                segmented_text = ' '.join(words)
                
                print(f"原文: {line}")
                print(f"分词结果: {segmented_text}")
                print("-" * 50)
                
                outf.write(segmented_text + "\n")
        
        print(f"分词结果已保存到 {output_file}")
        
    except FileNotFoundError as e:
        print(f"错误：处理文件时出错 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"处理文件时发生未知错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()