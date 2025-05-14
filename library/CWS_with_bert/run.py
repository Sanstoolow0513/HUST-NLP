import pickle
import logging
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizerFast
from model import CWS, BertCWS, BertBiLSTMCRF
from dataloader import Sentence
import os
from NER.model import CWS as NERModel # For loading NER model

def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--max_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=200, help="LSTM hidden dim for CWS model")
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--early_stopping', type=int, default=3, help='提前停止训练的耐心值')
    parser.add_argument('--model_type', type=str, default='bert_bilstm_crf', choices=['lstm', 'bert', 'bert_bilstm_crf'], help='模型类型: lstm, bert, bert_bilstm_crf')
    parser.add_argument('--bert_model_name', type=str, default='bert-base-chinese', help='BERT预训练模型名称')
    parser.add_argument('--bert_lr', type=float, default=2e-5, help='BERT模型学习率')
    parser.add_argument('--warmup_steps', type=int, default=0, help='预热步数')
    parser.add_argument('--bert_lstm_hidden_dim', type=int, default=256, help='BertBiLSTMCRF中LSTM层的隐藏层维度')
    parser.add_argument('--bert_lstm_layers', type=int, default=1, help='BertBiLSTMCRF中LSTM层的层数')
    parser.add_argument('--bert_dropout_rate', type=float, default=0.1, help='BertBiLSTMCRF中的dropout率')
    parser.add_argument('--ner_model_path', type=str, default='c:\\Users\\Sanstoolow\\Desktop\\HUST-NLP\\library\\NER\\save\\model_epoch9.pkl', help='Path to the trained NER model .pkl file (optional, uses default if None)')
    parser.add_argument('--ner_data_path', type=str, default='c:\\Users\\Sanstoolow\\Desktop\\HUST-NLP\\library\\NER\\data\\ner_datasave.pkl', help='Path to the NER data save file (ner_datasave.pkl) (optional, uses default if None)')
    parser.add_argument('--ner_embedding_dim', type=int, default=50, help='Dimension of NER tag embeddings') # New argument
    return parser.parse_args()


def set_logger():
    log_file = os.path.join('save', 'log.txt')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m%d %H:%M:%S',
        filename=log_file,
        filemode='w',
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO) # Changed to INFO for cleaner console output
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def entity_split(x, y, id2tag, entities, cur):
    """
    提取实体边界
    x: token序列
    y: 标签序列
    id2tag: 标签ID到标签名的映射
    entities: 用于存储提取的实体边界的集合
    cur: 当前处理的文本在整个数据集中的起始位置
    """
    if len(x) != len(y):
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
    
    start, end = -1, -1
    for j in range(len(y)):
        try:
            # 跳过特殊标记（-100）
            if y[j] == -100:
                continue
                
            # 检查标签ID是否在有效范围内
            if y[j] < 0 or y[j] >= len(id2tag):
                logging.warning(f"Invalid tag ID {y[j]} at position {j}, skipping...")
                start, end = -1, -1
                continue
                
            tag = id2tag[y[j]]
            if tag == 'B':
                start = cur + j
            elif tag == 'M' and start != -1:
                continue
            elif tag == 'E' and start != -1:
                end = cur + j
                entities.add((start, end))
                start, end = -1, -1
            elif tag == 'S':
                entities.add((cur + j, cur + j))
                start, end = -1, -1
            else:
                start, end = -1, -1
        except (KeyError, IndexError) as e:
            logging.warning(f"Error processing tag at position {j}: {str(e)}")
            start, end = -1, -1
            continue


class BertSentence(Dataset):
    """BERT数据集类"""
    def __init__(self, x_data, y_data, tokenizer, tag2id, id2word, max_length=128, ner_model_path=None, ner_data_path=None):
        # ner_model_path: path to the trained NER model (.pkl)
        # ner_data_path: path to the NER data save file (ner_datasave.pkl)
        self.x_data = x_data
        self.y_data = y_data
        self.tokenizer = tokenizer
        self.tag2id = tag2id # CWS tag2id
        self.id2word = id2word
        self.max_length = max_length
        
        # 确保CWS tag2id中包含'O'标签
        if 'O' not in self.tag2id:
            self.tag2id = self.tag2id.copy()
            self.tag2id['O'] = len(self.tag2id)
            logging.info(f"添加CWS 'O'标签，ID为: {self.tag2id['O']}")
        
        self.o_tag_id = self.tag2id['O'] # CWS 'O' tag ID

        # Load NER model and data
        self.ner_model = None
        self.ner_word2id = None
        self.ner_tag2id = None # NER tag2id
        self.ner_id2tag = None # NER id2tag
        self.ner_tagset_size = 0
        self.ner_o_tag_id = None # NER 'O' tag ID, if exists
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if ner_model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            ner_model_path = os.path.join(current_dir, '..', 'NER', 'save', 'model_epoch9.pkl') 
        if ner_data_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            ner_data_path = os.path.join(current_dir, '..', 'NER', 'data', 'ner_datasave.pkl')

        try:
            if os.path.exists(ner_model_path) and os.path.exists(ner_data_path):
                logging.info(f"Loading NER model from: {ner_model_path}")
                self.ner_model = torch.load(ner_model_path, map_location=self.device)
                self.ner_model.eval()
                logging.info(f"Loading NER data from: {ner_data_path}")
                with open(ner_data_path, 'rb') as f:
                    self.ner_word2id = pickle.load(f)
                    _ = pickle.load(f)  # ner_id2word, not strictly needed here
                    self.ner_tag2id = pickle.load(f) # tag2id from NER data
                    self.ner_id2tag = pickle.load(f)   # id2tag from NER data
                    self.ner_tagset_size = len(self.ner_id2tag)
                if 'O' in self.ner_tag2id:
                    self.ner_o_tag_id = self.ner_tag2id['O']
                else:
                    # If NER model doesn't have 'O', we might need a placeholder or handle it in the model
                    logging.warning("'O' tag not found in NER tag2id. Special tokens for NER features will be -100.")
                logging.info(f"NER model and data loaded successfully. NER tagset size: {self.ner_tagset_size}")
            else:
                logging.warning(f"NER model path ({ner_model_path}) or NER data path ({ner_data_path}) does not exist. NER integration will be disabled.")
                self.ner_model = None
        except Exception as e:
            logging.error(f"Error loading NER model or data: {e}. NER integration will be disabled.", exc_info=True)
            self.ner_model = None
        
        # 记录标签统计信息
        self.label_stats = {
            'B': 0, 'M': 0, 'E': 0, 'S': 0, 'O': 0,
            'total': 0, 'invalid': 0, 'total_samples_processed': 0, 'samples_with_errors': 0
        }
    
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx): # 添加这个方法
        return self._getitem_single(idx) # 调用已有的 _getitem_single 方法
    
    def _update_label_stats(self, label_id):
        """更新标签统计信息"""
        if label_id == -100:
            self.label_stats['invalid'] += 1
            return
            
        try:
            tag = self.tag2id[label_id]
            if tag in self.label_stats:
                self.label_stats[tag] += 1
            self.label_stats['total'] += 1
        except (KeyError, IndexError):
            self.label_stats['invalid'] += 1
    
    def _getitem_single(self, idx):
        try:
            original_char_ids = self.x_data[idx] # 原始字符ID序列
            original_char_labels = self.y_data[idx] # 原始字符对应的CWS标签ID序列
            
            text_chars = [self.id2word[char_id] for char_id in original_char_ids]
            
            encoding = self.tokenizer(
                text_chars,
                is_split_into_words=True, 
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze() 
            attention_mask = encoding['attention_mask'].squeeze()
            word_ids = encoding.word_ids() # 获取每个token对应的原始词（字符）索引
            
            aligned_cws_labels = torch.full((self.max_length,), -100, dtype=torch.long)
            aligned_ner_labels = torch.full((self.max_length,), -100, dtype=torch.long) # For NER features
            ner_char_pred_tag_ids = None # Predicted NER tag IDs for original characters

            if self.ner_model and self.ner_word2id and self.ner_id2tag:
                try:
                    ner_unk_id = self.ner_word2id.get('<UNK>', 0) 
                    ner_input_char_ids = [self.ner_word2id.get(char, ner_unk_id) for char in text_chars]
                    
                    if not ner_input_char_ids:
                        logging.warning(f"Sample {idx}: text_chars is empty. Skipping NER prediction.")
                    else:
                        ner_input_tensor = torch.tensor([ner_input_char_ids], dtype=torch.long).to(self.device)
                        ner_mask_tensor = torch.ones_like(ner_input_tensor, dtype=torch.bool).to(self.device)
                        ner_length_tensor = torch.tensor([len(ner_input_char_ids)], dtype=torch.long)

                        with torch.no_grad():
                            ner_predictions_batch = self.ner_model.infer(ner_input_tensor, ner_mask_tensor, ner_length_tensor)
                            if ner_predictions_batch and len(ner_predictions_batch) > 0:
                                ner_char_pred_tag_ids = ner_predictions_batch[0]
                                if len(ner_char_pred_tag_ids) != len(text_chars):
                                    logging.warning(f"Sample {idx}: NER prediction length ({len(ner_char_pred_tag_ids)}) mismatch with text_chars length ({len(text_chars)}). Disabling NER features for this sample.")
                                    ner_char_pred_tag_ids = None 
                            else:
                                logging.warning(f"Sample {idx}: NER model returned empty predictions. Disabling NER features for this sample.")
                                ner_char_pred_tag_ids = None
                except Exception as e:
                    logging.error(f"Error during NER prediction for sample {idx}: {e}. Disabling NER features for this sample.", exc_info=True)
                    ner_char_pred_tag_ids = None

            for token_idx, word_idx in enumerate(word_ids):
                if not attention_mask[token_idx].item(): # Padded token
                    aligned_cws_labels[token_idx] = -100 
                    aligned_ner_labels[token_idx] = -100
                    continue

                if word_idx is None: # CLS, SEP tokens
                    aligned_cws_labels[token_idx] = self.o_tag_id # CWS 'O' tag for special tokens
                    aligned_ner_labels[token_idx] = -100 # Or ner_o_tag_id if defined and appropriate for CLS/SEP
                    continue
                
                # Align CWS labels
                if 0 <= word_idx < len(original_char_labels):
                    aligned_cws_labels[token_idx] = original_char_labels[word_idx]
                else:
                    logging.warning(
                        f"Sample {idx}, token {token_idx}: CWS word_idx {word_idx} "
                        f"out of bounds for original_char_labels (len {len(original_char_labels)}). "
                        f"Text: {''.join(text_chars)}. "
                        f"Token: {self.tokenizer.convert_ids_to_tokens([input_ids[token_idx].item()])}"
                    )
                    aligned_cws_labels[token_idx] = -100 # Mark as ignore

                # Align NER labels (features)
                if ner_char_pred_tag_ids and 0 <= word_idx < len(ner_char_pred_tag_ids):
                    aligned_ner_labels[token_idx] = ner_char_pred_tag_ids[word_idx]
                else:
                    # If NER prediction failed or word_idx is out of bounds for NER preds
                    aligned_ner_labels[token_idx] = -100 # Mark as ignore or a default NER 'O' tag if available
                    if ner_char_pred_tag_ids and not (0 <= word_idx < len(ner_char_pred_tag_ids)):
                         logging.warning(
                            f"Sample {idx}, token {token_idx}: NER word_idx {word_idx} "
                            f"out of bounds for ner_char_pred_tag_ids (len {len(ner_char_pred_tag_ids)})."
                        )

            self.label_stats['total_samples_processed'] += 1
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': aligned_cws_labels, # CWS target labels
                'ner_labels': aligned_ner_labels # NER feature labels
            }
        except Exception as e:
            logging.error(f"Error processing sample {idx} in _getitem_single: {str(e)}", exc_info=True)
            self.label_stats['samples_with_errors'] += 1


if __name__ == '__main__':
    set_logger()
    args = get_param()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # 加载数据
    with open(os.path.join('data', 'datasave.pkl'), 'rb') as f:
        word2id = pickle.load(f)
        id2word = pickle.load(f)
        tag2id = pickle.load(f)
        id2tag = pickle.load(f)
        x_train = pickle.load(f)
        y_train = pickle.load(f)
        x_val = pickle.load(f)
        y_val = pickle.load(f)
        x_test = pickle.load(f)
        y_test = pickle.load(f)

    # 选择模型和数据集
    if args.model_type == 'lstm':
        train_dataset = Sentence(x_train, y_train)
        val_dataset = Sentence(x_val, y_val)
        model = CWS(len(word2id), tag2id, args.embedding_dim, args.hidden_dim)
        collate_fn = Sentence.collate_fn
    else:
        tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_name)
        train_dataset = BertSentence(x_train, y_train, tokenizer, tag2id, id2word, max_length=128, ner_model_path=args.ner_model_path, ner_data_path=args.ner_data_path)
        val_dataset = BertSentence(x_val, y_val, tokenizer, tag2id, id2word, max_length=128, ner_model_path=args.ner_model_path, ner_data_path=args.ner_data_path)
        # Pass ner_tagset_size to the model if it's BertBiLSTMCRF
        ner_tagset_size_for_model = train_dataset.ner_tagset_size if hasattr(train_dataset, 'ner_tagset_size') else 0
        
        if args.model_type == 'bert':
            model = BertCWS(tag2id, args.bert_model_name)
        else: # bert_bilstm_crf
            model = BertBiLSTMCRF(tag2id, args.bert_model_name, 
                                  lstm_hidden_dim=args.bert_lstm_hidden_dim, 
                                  lstm_layers=args.bert_lstm_layers, 
                                  dropout_rate=args.bert_dropout_rate,
                                  ner_tagset_size=ner_tagset_size_for_model, # Pass NER tagset size
                                  ner_embedding_dim=args.ner_embedding_dim # Pass NER embedding dim (new arg)
                                  )
        def bert_collate_fn(batch):
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            labels = torch.stack([item['labels'] for item in batch])
            ner_labels = torch.stack([item['ner_labels'] for item in batch])
            return input_ids, attention_mask, labels, ner_labels
        collate_fn = bert_collate_fn

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.bert_lr if 'bert' in args.model_type else args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_epoch * len(train_loader))

    from tqdm import tqdm
    best_val_loss = float('inf')
    patience = 0
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}/{args.max_epoch}')
        for step, batch in pbar:
            optimizer.zero_grad()
            if args.model_type == 'lstm':
                input_ids, labels, mask, lengths = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                mask = mask.to(device)
                lengths = torch.tensor(lengths, dtype=torch.long).to(device)
                loss = model(input_ids, labels, mask, lengths)
            else:
                input_ids, attention_mask, labels, ner_labels = batch # Correctly unpack 4 items
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                ner_labels = ner_labels.to(device) # Move ner_labels to device

                if args.model_type == 'bert_bilstm_crf':
                    loss = model(input_ids, labels, attention_mask, ner_labels=ner_labels)
                elif args.model_type == 'bert':
                    loss = model(input_ids, labels, attention_mask)
                # No else needed as model_type is one of these or 'lstm'
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                pbar.set_postfix({'batch_loss': loss.item(), 'avg_loss': train_loss / (step + 1)})
        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if args.model_type == 'lstm':
                    input_ids, labels, mask, lengths = batch
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    mask = mask.to(device)
                    lengths = torch.tensor(lengths, dtype=torch.long).to(device)
                    loss = model(input_ids, labels, mask, lengths)
                else:
                    input_ids, attention_mask, labels, ner_labels = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    ner_labels = ner_labels.to(device)
                    loss = model(input_ids, labels, attention_mask, ner_labels=ner_labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f'Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}')
        # print(f'Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}') # Removed redundant print

        # Early stopping & best model保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join('save', f'best_model_{args.model_type}.pkl'))
            logging.info(f'Best model saved at epoch {epoch}')
        else:
            patience += 1
            if patience >= args.early_stopping:
                logging.info(f'Early stopping at epoch {epoch}')
                # print(f'Early stopping at epoch {epoch}') # Removed redundant print
                break

    # print('训练结束，最优模型已保存。') # Removed redundant print
    logging.info('训练结束，最优模型已保存。')
