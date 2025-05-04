import os
import sys
import torch
import numpy as np
import pickle
import time
from tqdm import tqdm

# 添加父目录到路径，以便导入model.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TransformerSegmenter

class TransformerSegmenterInference:
    def __init__(self, model_path, data_pkl_path="../data/datasave.pkl", device=None, use_crf=True):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")

        # 从 pkl 文件加载映射关系和 pad_id
        try:
            with open(data_pkl_path, 'rb') as f:
                processed_data = pickle.load(f) # 加载整个字典
            self.word2id = processed_data['word2id']
            self.id2word = processed_data['id2word']
            self.tag2id = processed_data['tag2id']
            self.id2tag = processed_data['id2tag']
            self.pad_id = processed_data.get('pad_id', 0) # 获取 pad_id，默认为 0
        except FileNotFoundError:
            print(f"错误：无法找到数据文件 {data_pkl_path}。请确保路径正确并已运行数据处理脚本。")
            sys.exit(1)
        except KeyError as e:
             print(f"错误：数据文件 {data_pkl_path} 中缺少键 '{e}'。请确保 data_u.py 正确保存了所有需要的信息。")
             sys.exit(1)
        except Exception as e:
            print(f"加载数据文件 {data_pkl_path} 时出错: {e}")
            sys.exit(1)

        # 确保word2id中有<UNK>标记
        if '<UNK>' not in self.word2id:
            self.word2id['<UNK>'] = len(self.word2id)
            self.id2word.append('<UNK>')
            print("已添加<UNK>标记到词表")

        # 初始化模型
        vocab_size = len(self.word2id)
        num_classes = len(self.id2tag)
        self.use_crf = use_crf

        # 使用与训练时匹配的参数初始化模型
        # 注意：这里的超参数应该与训练时使用的最佳模型的超参数一致
        # 如果不确定，可以考虑将超参数也保存在 pkl 或模型文件名中
        embedding_dim = 256 # 假设与训练时一致
        transformer_ff_dim = 512 # 假设与训练时一致
        num_layers = 4 # 假设与训练时一致
        num_heads = 8 # 假设与训练时一致
        dropout = 0.2 # Dropout 在 eval 模式下通常无效，但保持一致性

        self.model = TransformerSegmenter(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            transformer_ff_dim=transformer_ff_dim, # 使用新名称
            num_classes=num_classes,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_crf=use_crf,
            padding_idx=self.pad_id # !!! 传递 pad_id !!!
        ).to(self.device)

        # 加载模型权重
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"成功加载模型权重: {model_path}")
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

    def preprocess(self, text):
        """将文本转换为模型输入格式"""
        # 将文本转换为ID序列
        char_ids = []
        for char in text:
            if char in self.word2id:
                char_ids.append(self.word2id[char])
            else:
                # 对于未知字符，使用<UNK>标记
                char_ids.append(self.word2id['<UNK>'])

        return torch.tensor([char_ids], dtype=torch.long).to(self.device)

    def segment(self, text):
        """对文本进行分词"""
        if not text:
            return []

        x = self.preprocess(text) # [1, seq_len]
        # 对于单句推理，mask 应全为 False (没有 padding)
        mask = torch.zeros(x.size(), dtype=torch.bool).to(self.device) # [1, seq_len], 全 False

        with torch.no_grad():
            # mask (True for padding) 正确传递 (这里是全 False)
            preds_or_emissions = self.model(x, mask=mask)

            if self.use_crf:
                # CRF decode 返回 List[List[int]], 取第一个 batch (即唯一的那个)
                preds = preds_or_emissions[0] # List[int]
            else:
                # 非 CRF 返回 emissions [1, seq_len, num_classes]
                preds = torch.argmax(preds_or_emissions, dim=2)[0].cpu().tolist() # List[int]

        # 根据预测标签进行分词
        words = []
        current_word = ""

        # 确保 preds 的长度与 text 匹配
        if len(preds) != len(text):
             print(f"警告：预测标签序列长度 ({len(preds)}) 与输入文本长度 ({len(text)}) 不匹配。可能存在模型或预处理问题。将尝试按最短长度处理。")
             min_len = min(len(preds), len(text))
             text = text[:min_len]
             preds = preds[:min_len]


        for i, char in enumerate(text):
            tag_id = preds[i]
            # 检查 tag_id 是否在有效范围内
            if tag_id < 0 or tag_id >= len(self.id2tag):
                print(f"警告：无效的 tag_id {tag_id} 在位置 {i}。使用'S'标签。")
                tag = 'S'  # 默认使用单字词标签
            else:
                tag = self.id2tag[tag_id]

            if tag == 'B':
                if current_word: words.append(current_word)
                current_word = char
            elif tag == 'M':
                if current_word: current_word += char
                else: words.append(char); current_word = "" # 异常处理
            elif tag == 'E':
                if current_word: current_word += char; words.append(current_word); current_word = ""
                else: words.append(char); current_word = "" # 异常处理
            elif tag == 'S':
                if current_word: words.append(current_word)
                words.append(char); current_word = ""
            else: # 未知标签
                print(f"警告：未知的标签 '{tag}' 在位置 {i}。")
                if current_word: words.append(current_word)
                words.append(char); current_word = ""

        if current_word:
            words.append(current_word)

        return words

    def batch_segment(self, texts, batch_size=32):
        """批量处理文本分词"""
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="批量分词"):
            batch_texts = texts[i:i+batch_size]
            for text in batch_texts:
                results.append(self.segment(text))
        return results

# 主程序入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="使用 Transformer 模型进行中文分词，并将结果保存到文件。")
    parser.add_argument("--model_path", type=str, default="transformer_segmenter_best_f1.pth", 
                        help="训练好的模型权重文件路径")
    parser.add_argument("--data_pkl_path", type=str, default="../data/datasave.pkl", 
                        help="包含 word2id 等映射的数据 pkl 文件路径")
    parser.add_argument("--input_file", type=str, default="../infer_apply_test.txt", 
                        help="待分词的输入文本文件路径")
    parser.add_argument("--output_file", type=str, default="segmentation_results_transformers.txt", 
                        help="保存分词结果的输出文本文件路径")
    parser.add_argument("--device", type=str, default=None, 
                        help="指定设备 (例如 'cpu' 或 'cuda:0')")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="批处理大小")
    parser.add_argument("--use_crf", action="store_true", 
                        help="是否使用CRF层进行解码")

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误：输入文件 {args.input_file} 不存在。")
        sys.exit(1)

    # 初始化推理器
    start_time = time.time()
    segmenter = TransformerSegmenterInference(
        model_path=args.model_path,
        data_pkl_path=args.data_pkl_path,
        device=args.device,
        use_crf=args.use_crf # 传递 use_crf 参数
    )
    init_time = time.time() - start_time
    print(f"模型初始化时间: {init_time:.2f}秒")

    print(f"正在从 {args.input_file} 读取文本...")
    print(f"分词结果将保存到 {args.output_file}...")

    try:
        # 读取所有文本
        with open(args.input_file, 'r', encoding='utf-8') as infile:
            lines = [line.strip() for line in infile]
        
        # 记录分词时间
        segment_start = time.time()
        
        # 批量处理
        if len(lines) > 1:
            print(f"使用批处理模式，批大小: {args.batch_size}")
            results = segmenter.batch_segment(lines, batch_size=args.batch_size)
            
            # 写入结果
            with open(args.output_file, 'w', encoding='utf-8') as outfile:
                for segmented_words in results:
                    if segmented_words:
                        outfile.write(" ".join(segmented_words) + "\n")
                    else:
                        outfile.write("\n")
        else:
            # 单行处理
            with open(args.output_file, 'w', encoding='utf-8') as outfile:
                for line in lines:
                    if not line:
                        outfile.write("\n")
                        continue
                    
                    segmented_words = segmenter.segment(line)
                    outfile.write(" ".join(segmented_words) + "\n")
        
        segment_time = time.time() - segment_start
        total_chars = sum(len(line) for line in lines if line)
        
        print(f"分词完成，结果已保存。")
        print(f"处理时间: {segment_time:.2f}秒")
        print(f"处理速度: {total_chars / segment_time:.2f} 字符/秒")
        print(f"总字符数: {total_chars}")
        print(f"总行数: {len(lines)}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        sys.exit(1)