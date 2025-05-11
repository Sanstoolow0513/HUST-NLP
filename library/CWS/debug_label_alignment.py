import pickle
import torch
from transformers import BertTokenizerFast
# 假设 run.py 和 model.py 在同一目录下或PYTHONPATH可找到
from run import BertSentence # BertSentence定义在run.py中
from data.data_u import id2tag, tag2id, id2word as data_u_id2word # 从data_u.py导入相关映射

# --- 配置参数 ---
DATASAVE_PATH = 'data/datasave.pkl'
BERT_MODEL_NAME = 'bert-base-chinese' # 与 run.py 和 infer.py 中使用的模型一致
MAX_LENGTH = 128 # 与 run.py 中 BertSentence 的默认值一致

# 加载BERT的tokenizer
tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

# 加载处理好的数据
with open(DATASAVE_PATH, 'rb') as inp:
    word2id = pickle.load(inp) # 这个word2id是data_u.py生成的
    id2word = pickle.load(inp) # 这个id2word是data_u.py生成的，确保与BertSentence内部一致
    # 注意：BertSentence内部可能重新定义tag2id, o_tag_id，这里加载的tag2id_from_data_u主要用于对比
    tag2id_from_data_u = pickle.load(inp)
    id2tag_from_data_u = pickle.load(inp)
    x_train = pickle.load(inp)
    y_train = pickle.load(inp)
    # 其他数据（x_val, y_val等）对于此调试脚本不是必需的

print(f"--- Debugging Label Alignment using {DATASAVE_PATH} ---")
print(f"Loaded {len(x_train)} samples from x_train.")

if not x_train or not y_train:
    print("Error: x_train or y_train is empty. Make sure data_u.py ran correctly on debug_train.txt.")
else:
    # 我们只检查第一个样本 (来自 "中国 人民")
    sample_idx = 0
    original_char_ids = x_train[sample_idx]
    original_tag_ids = y_train[sample_idx]

    print(f"\n--- Processing Sample {sample_idx} ---")
    
    # 从data_u.py的id2word转换原始字符ID回字符
    try:
        original_chars = [id2word[char_id] for char_id in original_char_ids]
        print(f"Original Chars (from data_u's id2word): {''.join(original_chars)}")
    except IndexError as e:
        print(f"Error converting original_char_ids to chars: {e}")
        print(f"original_char_ids: {original_char_ids}, id2word length: {len(id2word)}")
        original_chars = ['ERROR']

    # 从data_u.py的id2tag转换原始标签ID回标签名
    try:
        original_tags_named = [id2tag_from_data_u[tag_id] for tag_id in original_tag_ids]
        print(f"Original Tags (from data_u's id2tag):  {original_tags_named}")
    except IndexError as e:
        print(f"Error converting original_tag_ids to tags: {e}")
        original_tags_named = ['ERROR']

    # 实例化BertSentence
    # BertSentence 需要 x_data, y_data, tokenizer, tag2id, id2word
    # 注意：BertSentence内部对tag2id的处理，特别是'O'标签的添加
    # 为了模拟真实情况，我们传递从data_u加载的tag2id和id2word
    print(f"Initializing BertSentence with id2word (len {len(id2word)}) and tag2id (from data_u: {tag2id_from_data_u})")
    
    # BertSentence 在 run.py 中定义，它的构造函数需要 tag2id 和 id2word
    # 它内部可能会修改 tag2id (例如添加 'O' 如果不存在)
    # 我们使用从 datasave.pkl 加载的 id2word 和 tag2id_from_data_u
    dataset_for_debug = BertSentence(
        x_data=[original_char_ids], # BertSentence期望列表形式的x_data和y_data
        y_data=[original_tag_ids],
        tokenizer=tokenizer,
        tag2id=tag2id_from_data_u.copy(), # 传递副本以防内部修改影响后续打印
        id2word=id2word, # 使用从datasave加载的id2word
        max_length=MAX_LENGTH
    )

    # 获取BertSentence内部（可能已调整的）tag2id 和 o_tag_id，用于后续解码
    internal_tag2id = dataset_for_debug.tag2id
    internal_o_tag_id = dataset_for_debug.o_tag_id
    # 构建一个内部的id2tag映射，用于解码BertSentence输出的labels
    internal_id2tag = {v: k for k, v in internal_tag2id.items()}
    # 如果-100被用作忽略标签，也加入到可读映射中
    if -100 not in internal_id2tag : internal_id2tag[-100] = 'IGN'


    print(f"BertSentence internal tag2id: {internal_tag2id}")
    print(f"BertSentence internal o_tag_id: {internal_o_tag_id}")
    print(f"BertSentence internal id2tag (for decoding its output): {internal_id2tag}")


    # 调用 __getitem__ 获取处理后的数据
    print(f"\nCalling BertSentence.__getitem__({sample_idx})...")
    try:
        processed_sample = dataset_for_debug.__getitem__(sample_idx)
    except Exception as e:
        print(f"Error during BertSentence.__getitem__({sample_idx}): {e}")
        processed_sample = None

    if processed_sample:
        input_ids_tensor = processed_sample['input_ids']
        attention_mask_tensor = processed_sample['attention_mask']
        aligned_labels_tensor = processed_sample['labels']

        # 将 PyTorch张量转换为列表
        input_ids_list = input_ids_tensor.tolist()
        aligned_labels_list = aligned_labels_tensor.tolist()

        # 将input_ids转换回token字符串
        tokens = tokenizer.convert_ids_to_tokens(input_ids_list, skip_special_tokens=False)

        print("\n--- Results from BertSentence ---")
        print(f"Input Tokens (from input_ids, {len(tokens)} tokens):          {tokens}")
        
        # 将对齐后的标签ID转换回标签名，使用BertSentence内部的id2tag映射
        try:
            aligned_labels_named = [internal_id2tag.get(label_id, f'ERR_ID:{label_id}') for label_id in aligned_labels_list]
            print(f"Aligned Labels (ID->Name via internal map, {len(aligned_labels_named)} labels): {aligned_labels_named}")
        except Exception as e:
            print(f"Error converting aligned_labels_list to names: {e}")
            aligned_labels_named = ['ERROR'] * len(aligned_labels_list)

        print("\n--- Comparison ---")
        print(f"Original Chars:          {''.join(original_chars)} ({len(original_chars)} chars)")
        print(f"Original Tags (BME/S):   {original_tags_named} ({len(original_tags_named)} tags)")
        print(f"BERT Tokens:             {tokens}")
        print(f"Aligned Labels (BME/S):  {aligned_labels_named}")

        print("\n--- Detailed Per-Token Alignment ---")
        max_len_display = min(len(tokens), len(aligned_labels_named))
        print(f"{'Token':<15} | {'Label ID':<10} | {'Label Name':<10}")
        print("-" * 40)
        for i in range(max_len_display):
            token_str = tokens[i]
            label_id_str = str(aligned_labels_list[i])
            label_name_str = aligned_labels_named[i]
            print(f"{token_str:<15} | {label_id_str:<10} | {label_name_str:<10}")
            
        # 检查是否有长度不匹配的情况（不应该发生，因为标签是根据token对齐的）
        if len(tokens) != len(aligned_labels_named):
            print(f"WARNING: Length mismatch between tokens ({len(tokens)}) and aligned_labels ({len(aligned_labels_named)})")

print("\n--- Script Finished ---") 