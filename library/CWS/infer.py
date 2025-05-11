import torch
import pickle
import argparse
import os
import logging
from transformers import BertTokenizerFast
from model import BertCWS, BertBiLSTMCRF

def set_logger():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m%d %H:%M:%S',
    )

def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='save/best_model_bert_bilstm_crf.pkl', help='模型路径')
    parser.add_argument('--model_type', type=str, default='bert', choices=['bert', 'bert_bilstm_crf'], help='模型类型: bert或bert_bilstm_crf')
    parser.add_argument('--bert_model_name', type=str, default='bert-base-chinese', help='BERT预训练模型名称')
    parser.add_argument('--output_file', type=str, default='cws_result.txt', help='输出文件路径')
    parser.add_argument('--input_file', type=str, default='../test_data.txt', help='输入文件路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')
    parser.add_argument('--max_length', type=int, default=128, help='最大序列长度')
    parser.add_argument('--cuda', action='store_true', default=False, help='是否使用CUDA')
    parser.add_argument('--datasave_path', type=str, default='data/datasave.pkl', help='训练数据保存路径')
    return parser.parse_args()

def load_data(args):
    """加载标签映射和词表"""
    logging.info(f"Loading data from {args.datasave_path}")
    with open(args.datasave_path, 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        _ = pickle.load(inp)  # x_train
        _ = pickle.load(inp)  # y_train
        _ = pickle.load(inp)  # x_val
        _ = pickle.load(inp)  # y_val
        _ = pickle.load(inp)  # x_test
        _ = pickle.load(inp)  # y_test
    return word2id, id2word, tag2id, id2tag

def load_model(args, tag2id):
    """加载模型"""
    logging.info(f"Loading model from {args.model_path}")
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    # 根据模型类型实例化模型
    if args.model_type == 'bert':
        model = BertCWS(tag2id, args.bert_model_name)
    elif args.model_type == 'bert_bilstm_crf':
        model = BertBiLSTMCRF(tag2id, args.bert_model_name)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")
    
    # 加载模型参数
    if os.path.exists(args.model_path):
        # 检查文件是否存在，并根据文件类型加载
        if args.model_path.endswith('.pkl'):
            try:
                # 尝试加载整个模型
                model = torch.load(args.model_path, map_location=device)
                logging.info("Loaded entire model")
            except Exception as e:
                # 如果失败，尝试加载state_dict
                logging.info(f"Failed to load entire model: {e}. Trying to load state_dict...")
                try:
                    model.load_state_dict(torch.load(args.model_path, map_location=device))
                    logging.info("Loaded model state_dict")
                except Exception as e2:
                    logging.error(f"Failed to load model state_dict: {e2}")
                    raise e2
    else:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    model.to(device)
    model.eval()
    return model, device

MAX_DIAGNOSTIC_LINES = 3 # 控制打印诊断日志的行数

def infer_text(text, model, tokenizer, id2tag, device, max_length=128, line_number_for_logging=float('inf')):
    """分析单段文本并返回分词结果"""
    if not text.strip():
        return ""
    
    if line_number_for_logging <= MAX_DIAGNOSTIC_LINES:
        print(f"\n--- Diagnostic for input line {line_number_for_logging} (text chunk: '{text[:50]}...') ---")

    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        predictions = model.infer(input_ids, attention_mask) # List[List[int]]
    
    predicted_tag_ids_with_special_tokens = predictions[0]

    if line_number_for_logging <= MAX_DIAGNOSTIC_LINES:
        print(f"Raw predicted tag IDs (incl. CLS/SEP): {predicted_tag_ids_with_special_tokens}")
        # 同时打印出对应的标签名，方便对照
        try:
            raw_predicted_tags_named = [id2tag[tid] for tid in predicted_tag_ids_with_special_tokens]
            print(f"Raw predicted tags (named): {raw_predicted_tags_named}")
        except IndexError as e:
            print(f"Error converting raw tag IDs to names: {e}. Some tag IDs might be out of bounds for id2tag.")

    words = []
    current_word = ""
    
    # 遍历原始文本的每个字符
    for char_idx, char_val in enumerate(text):
        # 原始文本中 char_idx 位置的字符，其对应的标签在 predicted_tag_ids_with_special_tokens 中的索引应该是 char_idx + 1
        # 因为 predicted_tag_ids_with_special_tokens[0] 对应 [CLS] 标记
        tag_idx_for_char = char_idx + 1
        
        # 检查计算出的标签索引是否有效：
        # 1. 不应超出 predicted_tag_ids_with_special_tokens 的范围
        # 2. 不应取到末尾 [SEP] 标记对应的标签（通常是 len - 1 的位置）
        # 因此，有效的标签索引范围是 1 到 len(predicted_tag_ids_with_special_tokens) - 2
        if tag_idx_for_char < len(predicted_tag_ids_with_special_tokens) - 1:
            tag_id = predicted_tag_ids_with_special_tokens[tag_idx_for_char]
            try:
                tag_name = id2tag[tag_id]
            except IndexError:
                # 如果tag_id无效（例如模型预测出范围外的ID），作为鲁棒性处理，视为'O'
                # logging.warning(f"Invalid tag_id {tag_id} for char '{char_val}'. Treating as 'O'.")
                tag_name = 'O' 
            
            if line_number_for_logging <= MAX_DIAGNOSTIC_LINES:
                print(f"  Char: '{char_val}' (idx {char_idx}) -> Model Token Idx {tag_idx_for_char} -> Tag ID: {tag_id}, Tag Name: '{tag_name}'")
        else:
            # 如果原始文本字符超出了模型能提供的有效标签范围 (例如文本被截断，或已到[SEP]标记之后)
            # 则结束当前词（如果正在构建中），并停止处理后续字符
            if current_word:
                words.append(current_word)
                current_word = ""
            break # 停止处理，因为没有更多有效标签了

        # 根据BME/S标签构建词语
        if tag_name == 'B': # Begin
            if current_word: # 如果前一个词因为其他原因（如遇到'O'或'S'）未结束，则先添加
                words.append(current_word)
            current_word = char_val
        elif tag_name == 'M': # Middle
            current_word += char_val
        elif tag_name == 'E': # End
            current_word += char_val
            words.append(current_word)
            current_word = ""
        elif tag_name == 'S': # Single
            if current_word: # 如果前一个词因为其他原因（如遇到'O'）未结束，则先添加
                words.append(current_word)
            words.append(char_val)
            current_word = ""
        else:  # 'O' (Outside) 或其他未知标签
            if current_word: # 如果一个词正在构建中，遇到'O'则表示该词结束
                words.append(current_word)
                current_word = ""
            # 'O' 标签的字符本身不计入词中，所以这里不需要 current_word = char_val 或 words.append(char_val)
    
    # 循环结束后，如果仍有未完成的词（例如文本以B或B-M结尾），则将其添加
    if current_word:
        words.append(current_word)
    
    if line_number_for_logging <= MAX_DIAGNOSTIC_LINES:
        print(f"Resulting words for this chunk: {words}")
        print(f"--- End Diagnostic for input line {line_number_for_logging} (chunk: '{text[:50]}...') ---")
    
    return ' '.join(words)

def process_long_text(text, model, tokenizer, id2tag, device, max_length=128, overlap=10, line_number_for_logging=float('inf')):
    """处理长文本，将其分割成较小的块并进行推理，并正确合并结果。"""
    if not text.strip():
        return ""

    full_text_len = len(text)
    # BERT处理时会加入[CLS]和[SEP]，所以有效处理长度是max_length-2
    effective_chunk_len = max_length - 2

    if full_text_len <= effective_chunk_len:
        # 文本不长于模型的单次处理上限，直接调用infer_text
        return infer_text(text, model, tokenizer, id2tag, device, max_length, line_number_for_logging=line_number_for_logging)

    all_words_final = []
    
    # current_pos_in_text 指向当前块在原始文本中的起始位置
    current_pos_in_text = 0
    # chars_effectively_covered 指向原始文本中已经被all_words_final覆盖到的字符的末尾后一个位置
    chars_effectively_covered = 0

    is_first_chunk = True

    while current_pos_in_text < full_text_len:
        chunk_start_abs = current_pos_in_text
        chunk_end_abs = min(current_pos_in_text + effective_chunk_len, full_text_len)
        current_chunk_text = text[chunk_start_abs:chunk_end_abs]

        if not current_chunk_text:  # 理论上不应发生
            break

        # 调用infer_text获取当前块的分词结果（字符串形式）
        raw_chunk_segmented_string = infer_text(current_chunk_text, model, tokenizer, id2tag, device, max_length, line_number_for_logging=line_number_for_logging)
        segmented_words_in_chunk = [w for w in raw_chunk_segmented_string.split(' ') if w] # 分割并去除空字符串

        if not segmented_words_in_chunk:
            # 如果当前块没有产生任何词语，需要确保指针前进以避免死循环
            advance_by = effective_chunk_len - overlap
            if advance_by <= 0: advance_by = 1 # 最小步进
            current_pos_in_text += advance_by
            if current_pos_in_text > chars_effectively_covered : # 更新覆盖范围
                 chars_effectively_covered = current_pos_in_text
            continue
            
        # 对于第一个块，我们取所有分词结果
        if is_first_chunk:
            all_words_final.extend(segmented_words_in_chunk)
            # 更新已覆盖的字符数：这里我们假设分词结果准确反映了块内所有字符
            # 更准确的计算应该是 `len("".join(segmented_words_in_chunk))` 但这不包含原始空格
            # 我们用块的实际结束位置来标记覆盖
            chars_effectively_covered = chunk_end_abs
            is_first_chunk = False
        else:
            # 对于后续的块，我们需要确定哪些词是新的（即不属于上一个块的重叠部分）
            # `start_char_offset_in_current_chunk` 指的是当前块中，新内容的起始字符相对于块首的偏移
            start_char_offset_in_current_chunk = 0
            if chunk_start_abs < chars_effectively_covered: # 存在重叠
                start_char_offset_in_current_chunk = chars_effectively_covered - chunk_start_abs
            
            # 遍历当前块的分词结果，只添加新部分的词
            temp_char_counter_in_chunk = 0
            for word in segmented_words_in_chunk:
                # 如果这个词的起始位置在当前块的新内容区域
                if temp_char_counter_in_chunk >= start_char_offset_in_current_chunk:
                    all_words_final.append(word)
                temp_char_counter_in_chunk += len(word) # 简单地用词长累加，忽略原始空格
            
            # 更新已覆盖字符的绝对位置
            chars_effectively_covered = chunk_end_abs

        # 决定下一个块的起始位置
        if chunk_end_abs == full_text_len:
            break # 已处理完整个文本
        
        next_chunk_start = chunk_end_abs - overlap
        # 确保指针前进，避免在overlap >= effective_chunk_len时卡住
        if next_chunk_start <= current_pos_in_text:
            next_chunk_start = current_pos_in_text + 1 # 最小前进一个字符
        
        current_pos_in_text = next_chunk_start
        if current_pos_in_text >= full_text_len : # 如果计算出的下一个起始点超出了文本长度
             break


    return ' '.join(all_words_final)

def main():
    set_logger()
    args = get_param()
    
    # 加载数据和标签映射
    _, _, tag2id, id2tag = load_data(args)
    
    # 加载模型
    model, device = load_model(args, tag2id)
    
    # 加载tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_name)
    
    # 读取输入文件并处理
    with open(args.input_file, 'r', encoding='utf-8') as f_in, \
         open(args.output_file, 'w', encoding='utf-8') as f_out:
        
        total_lines = sum(1 for _ in open(args.input_file, 'r', encoding='utf-8'))
        processed_lines = 0
        
        for line in f_in:
            processed_lines += 1
            if processed_lines % 100 == 0:
                logging.info(f"Processing line {processed_lines}/{total_lines}")
            
            line = line.strip()
            if not line:
                f_out.write("\n")
                continue
            
            segmented_text = process_long_text(line, model, tokenizer, id2tag, device, args.max_length, line_number_for_logging=processed_lines)
            
            f_out.write(segmented_text + "\n")
    
    logging.info(f"分词结果已保存至 {args.output_file}")

if __name__ == '__main__':
    main()
