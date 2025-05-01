import torch
import pickle
import argparse
import os

def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cws_model', type=str, default='save/CWS/model_epoch9.pkl', help='CWS模型路径')
    parser.add_argument('--ner_model', type=str, default='save/NER/model_epoch9.pkl', help='NER模型路径')
    parser.add_argument('--test_file', type=str, default='data/CWS/data/test_data.txt', help='测试文件路径')
    parser.add_argument('--output_file', type=str, default='combined_result.txt', help='输出文件路径')
    parser.add_argument('--use_ner', action='store_true', default=False, help='是否使用NER结果辅助分词')
    return parser.parse_args()

def load_model_and_data(model_path, data_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    
    with open(data_path, 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
    
    return model, word2id, id2word, tag2id, id2tag

def cws_infer(text, cws_model, cws_word2id, cws_id2tag):
    """中文分词推理"""
    x = torch.LongTensor(1, len(text))
    mask = torch.ones_like(x, dtype=torch.uint8)
    length = [len(text)]
    
    for i in range(len(text)):
        if text[i] in cws_word2id:
            x[0, i] = cws_word2id[text[i]]
        else:
            x[0, i] = len(cws_word2id)
    
    predict = cws_model.infer(x, mask, length)[0]
    
    words = []
    word = ""
    for i in range(len(text)):
        word += text[i]
        if cws_id2tag[predict[i]] in ['E', 'S']:
            words.append(word)
            word = ""
    
    if word:  # 处理最后可能的残余字符
        words.append(word)
    
    return words, predict

def ner_infer(text, ner_model, ner_word2id, ner_id2tag):
    """命名实体识别推理"""
    x = torch.LongTensor(1, len(text))
    mask = torch.ones_like(x, dtype=torch.uint8)
    length = [len(text)]
    
    for i in range(len(text)):
        if text[i] in ner_word2id:
            x[0, i] = ner_word2id[text[i]]
        else:
            x[0, i] = len(ner_word2id)
    
    predict = ner_model.infer(x, mask, length)[0]
    
    entities = []
    start, end = -1, -1
    entity_type = None
    
    for i in range(len(text)):
        tag = ner_id2tag[predict[i]]
        if tag.startswith('B-'):
            start = i
            entity_type = tag[2:]
        elif tag.startswith('I-') and start != -1 and entity_type == tag[2:]:
            continue
        elif tag.startswith('E-') and start != -1 and entity_type == tag[2:]:
            end = i
            entities.append((start, end, entity_type))
            start, end = -1, -1
            entity_type = None
        elif tag.startswith('S-'):
            entities.append((i, i, tag[2:]))
        elif tag == 'O':
            start, end = -1, -1
            entity_type = None
    
    return entities, predict

def main():
    args = get_param()
    
    # 加载CWS模型
    cws_model, cws_word2id, cws_id2word, cws_tag2id, cws_id2tag = load_model_and_data(
        args.cws_model, 'data/CWS/data/datasave.pkl')
    
    # 加载NER模型
    ner_model, ner_word2id, ner_id2word, ner_tag2id, ner_id2tag = load_model_and_data(
        args.ner_model, 'data/NER/data/ner_datasave.pkl')
    
    output = open(args.output_file, 'w', encoding='utf-8')
    
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                print(file=output)
                continue
            
            # 中文分词
            words, cws_predict = cws_infer(line, cws_model, cws_word2id, cws_id2tag)
            
            # 命名实体识别
            entities, ner_predict = ner_infer(line, ner_model, ner_word2id, ner_id2tag)
            
            # 输出分词结果
            print(' '.join(words), file=output)
            
            # 输出实体信息
            # debug
            # if entities:
            #     print("实体信息:", file=output)
            #     for start, end, entity_type in entities:
            #         entity_text = line[start:end+1]
            #         print(f"{entity_text} ({entity_type})", file=output)
    
    output.close()
    print(f"结果已保存到 {args.output_file}")

if __name__ == '__main__':
    main()