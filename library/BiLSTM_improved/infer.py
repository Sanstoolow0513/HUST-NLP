import torch
import pickle
# 导入 CWS 类，以便 torch.load 可以正确反序列化模型对象
# 假设 model.py 在同一目录下或 Python 路径中
from model import CWS 

# 定义字符类型映射 (与 data_u.py 中一致)
char_type_map = {'CN': 0, 'EN': 1, 'NUM': 2, 'PUNC': 3, 'OTHER': 4}

def get_char_type(char):
    """根据字符返回其类型ID"""
    if '\u4e00' <= char <= '\u9fff':
        return char_type_map['CN']
    elif 'a' <= char.lower() <= 'z': # 统一处理大小写
        return char_type_map['EN']
    elif '0' <= char <= '9':
        return char_type_map['NUM']
    # 更全面的标点符号判断 (示例)
    elif char in ',.!?;:，。！？；：\'"()[]{}<>《》“”‘’': 
        return char_type_map['PUNC']
    else:
        return char_type_map['OTHER']

if __name__ == '__main__':
    # 确保模型文件路径正确
    model_path = 'save/best_model.pkl' # 或者 'save/model_epochX.pkl'
    # 注意：这里假设 'save/best_model.pkl' 保存的是模型状态字典 (state_dict)
    # 如果保存的是整个模型对象，加载方式不变，但需要确保 CWS 类已导入
    
    # --- 加载数据 ---
    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        # 加载 char_type_vocab_size (虽然这里不直接用，但保持加载完整性)
        char_type_vocab_size = pickle.load(inp) 
        # 不再需要加载训练/测试数据
        # x_train = pickle.load(inp)
        # y_train = pickle.load(inp)
        # x_char_types_train = pickle.load(inp)
        # x_test = pickle.load(inp)
        # y_test = pickle.load(inp)
        # x_char_types_test = pickle.load(inp)

    # --- 实例化模型并加载状态 ---
    # 使用与训练时相同的参数
    embedding_dim = 500 # 改为与训练时一致
    hidden_dim = 1024   # 改为与训练时一致
    
    # 实例化模型结构
    model = CWS(len(word2id), tag2id, embedding_dim, hidden_dim, char_type_vocab_size)
    
    # 加载模型状态字典
    # map_location 确保在没有 GPU 时也能加载
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # 设置为评估模式
    
    # --- 推理过程 ---
    output_file = 'cws_result.txt'
    input_file = 'data/test_data.txt' # 假设测试数据在此文件

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            test_sentence = line.strip()
            if not test_sentence:
                continue

            # 准备输入张量
            sentence_len = len(test_sentence)
            x = torch.LongTensor(1, sentence_len)
            char_types = torch.LongTensor(1, sentence_len) # 创建 char_types 张量
            
            for i, char in enumerate(test_sentence):
                # 处理字 ID
                x[0, i] = word2id.get(char, len(word2id)) # 使用 get 处理 OOV
                # 处理字符类型 ID
                char_types[0, i] = get_char_type(char)

            # 创建 mask (dtype=torch.bool)
            mask = torch.ones(1, sentence_len, dtype=torch.bool) 
            length = [sentence_len] # 句子实际长度

            # 执行推理
            # 使用 torch.no_grad() 避免计算梯度
            with torch.no_grad():
                predict_tags_list = model.infer(x, char_types, mask, length) 
            
            # predict_tags_list 是一个列表，包含一个元素的列表（因为 batch_size=1）
            # 每个元素是对应句子的预测标签 ID 列表
            if predict_tags_list:
                predict_tags = predict_tags_list[0] # 获取第一个（也是唯一一个）句子的预测结果
                
                # 输出分词结果
                for i in range(sentence_len):
                    f_out.write(test_sentence[i])
                    # 根据预测标签决定是否加空格
                    if id2tag[predict_tags[i]] in ['E', 'S']:
                        f_out.write(' ')
                f_out.write('\n') # 每个句子后换行
            else:
                # 处理解码失败或空结果的情况
                print(f"Warning: No prediction for sentence: {test_sentence}", file=sys.stderr) # 打印警告到标准错误流
                f_out.write(test_sentence + '\n') # 原样输出

    print(f"分词结果已写入 {output_file}")
