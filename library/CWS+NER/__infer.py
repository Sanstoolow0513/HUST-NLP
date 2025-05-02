import torch
import pickle
import argparse

def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='save/CWS/model_epoch9.pkl', help='模型路径')
    parser.add_argument('--data', type=str, default='data/CWS/data/datasave.pkl', help='数据路径')
    parser.add_argument('--test_file', type=str, default='data/CWS/data/test.txt', help='测试文件路径')
    parser.add_argument('--output_file', type=str, default='cws_result.txt', help='输出文件路径')
    return parser.parse_args()

def main():
    args = get_param()
    
    model = torch.load(args.model, map_location=torch.device('cpu'))
    output = open(args.output_file, 'w', encoding='utf-8')

    with open(args.data, 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    with open(args.test_file, 'r', encoding='utf-8') as f:
        for test in f:
            test = test.strip()
            if not test:
                print(file=output)
                continue

            x = torch.LongTensor(1, len(test))
            mask = torch.ones_like(x, dtype=torch.uint8)
            length = [len(test)]
            for i in range(len(test)):
                if test[i] in word2id:
                    x[0, i] = word2id[test[i]]
                else:
                    x[0, i] = len(word2id)

            predict = model.infer(x, mask, length)[0]
            for i in range(len(test)):
                print(test[i], end='', file=output)
                if id2tag[predict[i]] in ['E', 'S']:
                    print(' ', end='', file=output)
            print(file=output)
    
    output.close()
    print(f"结果已保存到 {args.output_file}")

if __name__ == '__main__':
    main()
