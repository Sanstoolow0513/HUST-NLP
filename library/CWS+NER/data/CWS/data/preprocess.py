import re
import os
import codecs

def clean_text(text):
    """
    清洗文本数据
    1. 去除多余空格
    2. 标准化标点符号
    3. 处理特殊字符
    """
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text)
    
    # 标准化标点符号（全角转半角）
    punctuation_map = {
        '，': ',', '。': '.', '！': '!', '？': '?',
        '；': ';', '：': ':', '（': '(', '）': ')',
        '"': '"', '"': '"', ''': "'", ''': "'",
        '【': '[', '】': ']', '《': '<', '》': '>',
        '——': '-', '……': '...', '、': ',', '　': ' '
    }
    for k, v in punctuation_map.items():
        text = text.replace(k, v)
    
    # 处理特殊字符
    text = re.sub(r'[^\w\s,.!?;:()\[\]<>\'\"]+', '', text)
    
    return text.strip()

def normalize_digits(text):
    """
    将文本中的数字标准化
    例如：将"１２３４５"转换为"12345"
    """
    digit_map = {
        '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
        '５': '5', '６': '6', '７': '7', '８': '8', '９': '9'
    }
    for k, v in digit_map.items():
        text = text.replace(k, v)
    return text

def process_file(input_file, output_file):
    """处理单个文件"""
    with codecs.open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            processed_lines.append('')
            continue
        
        # 清洗和标准化
        line = clean_text(line)
        line = normalize_digits(line)
        
        processed_lines.append(line)
    
    with codecs.open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(processed_lines))

def main():
    # 处理训练数据
    process_file('train.txt', 'train_clean.txt')
    
    # 处理测试数据
    if os.path.exists('test.txt'):
        process_file('test.txt', 'test_clean.txt')
    
    print("数据预处理完成！")

if __name__ == "__main__":
    main()