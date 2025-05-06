class Tokenizer(object):
    def __init__(self, words, max_len):
        self.words = words
        self.max_len = max_len

    def fmm_split(self, text):
        '''
        正向最大匹配分词算法
        :param text: 待分词字符串
        :return: 分词结果，以list形式存放，每个元素为分出的词
        '''
        result = []
        i = 0
        while i < len(text):
            # 取当前位置开始的最大长度子串
            max_match_len = min(self.max_len, len(text) - i)
            word = text[i:i + max_match_len]
            
            # 逐渐缩短子串长度，直到找到匹配的词或长度为1
            while max_match_len > 0:
                if word in self.words or max_match_len == 1:
                    result.append(word)
                    break
                max_match_len -= 1
                word = text[i:i + max_match_len]
            
            # 移动指针
            i += max_match_len
        
        return result

    def rmm_split(self, text):
        '''
        逆向最大匹配分词算法
        :param text: 待分词字符串
        :return: 分词结果，以list形式存放，每个元素为分出的词
        '''
        result = []
        i = len(text)
        
        while i > 0:
            # 取当前位置结束的最大长度子串
            max_match_len = min(self.max_len, i)
            word = text[i - max_match_len:i]
            
            # 逐渐缩短子串长度，直到找到匹配的词或长度为1
            while max_match_len > 0:
                if word in self.words or max_match_len == 1:
                    result.insert(0, word)  # 在结果列表前端插入
                    break
                max_match_len -= 1
                word = text[i - max_match_len:i]
            
            # 移动指针
            i -= max_match_len
        
        return result

    def bimm_split(self, text):
        '''
        双向最大匹配分词算法
        :param text: 待分词字符串
        :return: 分词结果，以list形式存放，每个元素为分出的词
        '''
        # 分别使用正向和逆向最大匹配
        fmm_result = self.fmm_split(text)
        rmm_result = self.rmm_split(text)
        
        # 如果两种方法结果词数不同，返回分词数量较少的那个
        if len(fmm_result) != len(rmm_result):
            return fmm_result if len(fmm_result) < len(rmm_result) else rmm_result
        
        # 如果分词数量相同，返回单字词数较少的那个
        fmm_single = sum(1 for word in fmm_result if len(word) == 1)
        rmm_single = sum(1 for word in rmm_result if len(word) == 1)
        
        return fmm_result if fmm_single < rmm_single else rmm_result


def load_dict(path):
    tmp = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split(' ')[0]
            tmp.add(word)
    return tmp


if __name__ == '__main__':
    words = load_dict('dict.txt')
    max_len = max(map(len, [word for word in words]))
    tokenizer = Tokenizer(words, max_len)
    
    # 读取 ../test_data.txt 文件中的句子
    input_file_path = '../test_data.txt'
    fmm_output_path = 'fmm_result.txt'
    rmm_output_path = 'rmm_result.txt'
    bimm_output_path = 'bimm_result.txt'
    
    try:
        # 读取输入文件
        with open(input_file_path, 'r', encoding='utf-8') as f_in:
            sentences = [line.strip() for line in f_in if line.strip()]
        
        # 打开三个输出文件
        with open(fmm_output_path, 'w', encoding='utf-8') as f_fmm, \
             open(rmm_output_path, 'w', encoding='utf-8') as f_rmm, \
             open(bimm_output_path, 'w', encoding='utf-8') as f_bimm:
            
            # 处理每个句子
            for sentence in sentences:
                # 使用三种分词方法
                fmm_result = tokenizer.fmm_split(sentence)
                rmm_result = tokenizer.rmm_split(sentence)
                bimm_result = tokenizer.bimm_split(sentence)
                
                # 将结果写入对应的文件，每个词之间用空格分隔
                f_fmm.write(' '.join(fmm_result) + '\n')
                f_rmm.write(' '.join(rmm_result) + '\n')
                f_bimm.write(' '.join(bimm_result) + '\n')
                
        print(f'分词完成，结果已保存至：')
        print(f'前向最大匹配结果：{fmm_output_path}')
        print(f'后向最大匹配结果：{rmm_output_path}')
        print(f'双向最大匹配结果：{bimm_output_path}')
    
    except FileNotFoundError:
        print(f'错误：找不到文件 {input_file_path}')
    except Exception as e:
        print(f'处理过程中发生错误: {str(e)}')
