import torch
import pickle

if __name__ == '__main__':
    # 注意：模型路径是 'save/model_epoch9.pkl'。
    # 如果您有 'save/best_model.pkl' 或希望使用特定epoch的模型，请确保路径正确。
    model = torch.load('save/best_model.pkl', map_location=torch.device('cpu'))
    # 注意：输出文件路径是 'cws_result.txt'，它将在当前工作目录（即 library/CWS_base/）创建。
    output = open('cws_result.txt', 'w', encoding='utf-8')

    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        # x_train, y_train, x_test, y_test 是加载了但在此脚本中未使用，这是正常的，因为infer.py用于新数据
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    # 获取未登录词的ID，根据data_u.py的逻辑，id2word[0] 应该是 UNK_TOKEN
    # 并且 word2id[id2word[0]] 应该是 UNK_TOKEN 对应的ID (通常是0)
    unk_token_id = word2id[id2word[0]]

    # 注意：输入文件路径是 '../test_data.txt'。
    # 这意味着脚本期望在 'library/' 目录下找到 'test_data.txt' 文件。
    # 即 c:\Users\Sanstoolow\Desktop\HUST-NLP\library\test_data.txt
    # 如果您的测试文件在其他位置（例如项目根目录或 data/ 目录），请相应调整此路径。
    with open('../test_data.txt', 'r', encoding='utf-8') as f:
        for test in f:
            test = test.strip()
            if not test: # 跳过空行
                print(file=output) # 在输出文件中也打印一个空行以保持对应
                continue

            x = torch.LongTensor(1, len(test))
            mask = torch.ones_like(x, dtype=torch.uint8) # 在 CRF 中，mask 类型通常是 torch.bool 或者 torch.uint8
            length = [len(test)]
            for i in range(len(test)):
                if test[i] in word2id:
                    x[0, i] = word2id[test[i]]
                else:
                    x[0, i] = unk_token_id # 使用预定义的UNK_TOKEN ID

            predict = model.infer(x, mask, length)[0]
            for i in range(len(test)):
                print(test[i], end='', file=output)
                if id2tag[predict[i]] in ['E', 'S']:
                    print(' ', end='', file=output)
            print(file=output)

    output.close()
    print("分词结果已保存到 cws_result.txt")
