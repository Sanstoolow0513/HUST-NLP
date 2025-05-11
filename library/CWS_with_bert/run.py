import pickle
import logging
import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizerFast
from model import CWS, BertCWS, BertBiLSTMCRF
from dataloader import Sentence

def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--max_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=200, help="LSTM hidden dim for CWS model")
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--early_stopping', type=int, default=3, help='提前停止训练的耐心值')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'bert', 'bert_bilstm_crf'], help='模型类型: lstm, bert, bert_bilstm_crf')
    parser.add_argument('--bert_model_name', type=str, default='bert-base-chinese', help='BERT预训练模型名称')
    parser.add_argument('--bert_lr', type=float, default=2e-5, help='BERT模型学习率')
    parser.add_argument('--warmup_steps', type=int, default=0, help='预热步数')
    parser.add_argument('--bert_lstm_hidden_dim', type=int, default=256, help='BertBiLSTMCRF中LSTM层的隐藏层维度')
    parser.add_argument('--bert_lstm_layers', type=int, default=1, help='BertBiLSTMCRF中LSTM层的层数')
    parser.add_argument('--bert_dropout_rate', type=float, default=0.1, help='BertBiLSTMCRF中的dropout率')
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
    console.setLevel(logging.DEBUG)
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
    def __init__(self, x_data, y_data, tokenizer, tag2id, id2word, max_length=128):
        self.x_data = x_data
        self.y_data = y_data
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.id2word = id2word
        self.max_length = max_length
        
        # 确保tag2id中包含'O'标签
        if 'O' not in tag2id:
            self.tag2id = tag2id.copy()
            self.tag2id['O'] = len(tag2id)
            logging.info(f"添加'O'标签，ID为: {self.tag2id['O']}")
        else:
            self.tag2id = tag2id
            
        self.o_tag_id = self.tag2id['O']
        
        # 记录标签统计信息
        self.label_stats = {
            'B': 0, 'M': 0, 'E': 0, 'S': 0, 'O': 0,
            'total': 0, 'invalid': 0, 'total_samples_processed': 0, 'samples_with_errors': 0
        }
    
    def __len__(self):
        return len(self.x_data)
    
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
            original_char_labels = self.y_data[idx] # 原始字符对应的标签ID序列
            
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
            
            aligned_labels = torch.full((self.max_length,), -100, dtype=torch.long)

            for token_idx, word_idx in enumerate(word_ids):
                if not attention_mask[token_idx].item(): # Padded token
                    aligned_labels[token_idx] = -100 
                    continue

                # CLS, SEP tokens (word_idx is None for these)
                if word_idx is None: 
                    aligned_labels[token_idx] = self.o_tag_id # 使用 'O' 标签 for CLS/SEP
                    continue
                
                # word_idx 直接是原始字符在 original_char_labels 中的索引
                if word_idx < len(original_char_labels):
                    label_id_for_token = original_char_labels[word_idx]
                    aligned_labels[token_idx] = label_id_for_token
                else:
                    # This case should ideally not happen if word_ids are generated correctly
                    # and original_char_labels is valid for the given original_char_ids.
                    logging.warning(f"Sample {idx}, token {token_idx}: word_idx {word_idx} out of bounds for original_char_labels (len {len(original_char_labels)}). Text: {''.join(text_chars)}. Token: {self.tokenizer.convert_ids_to_tokens([input_ids[token_idx].item()])}")
                    aligned_labels[token_idx] = -100 

            self.label_stats['total_samples_processed'] += 1
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': aligned_labels
            }
        except Exception as e:
            logging.error(f"Error processing sample {idx} in _getitem_single: {str(e)}", exc_info=True)
            self.label_stats['samples_with_errors'] += 1
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.full((self.max_length,), -100, dtype=torch.long)
            }
    
    def __getitem__(self, idx):
        try:
            return self._getitem_single(idx)
        except Exception as e:
            logging.error(f"处理样本 {idx} 时出错: {str(e)}")
            # 返回一个有效的空样本
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.zeros(self.max_length, dtype=torch.long)
            }
    
    @staticmethod
    def collate_fn(batch):
        """整理批次数据"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return input_ids, labels, attention_mask
    
    def print_label_stats(self):
        """打印标签统计信息"""
        total_valid = self.label_stats['total']
        if total_valid > 0:
            logging.info("标签统计信息:")
            for tag in ['B', 'M', 'E', 'S', 'O']:
                count = self.label_stats[tag]
                percentage = (count / total_valid) * 100
                logging.info(f"{tag}: {count} ({percentage:.2f}%)")
            logging.info(f"无效标签: {self.label_stats['invalid']}")
            logging.info(f"总有效标签: {total_valid}")


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_val = pickle.load(inp)
        y_val = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    logging.debug(f"id2tag type: {type(id2tag)}, content: {id2tag}")
    # 根据模型类型选择不同的模型和数据加载方式
    if args.model_type == 'lstm':
        # 使用原始LSTM+CRF模型
        model = CWS(len(word2id), tag2id, args.embedding_dim, args.hidden_dim)
        
        train_dataset = Sentence(x_train, y_train)
        val_dataset = Sentence(x_val, y_val)
        test_dataset = Sentence(x_test[:1000], y_test[:1000])
        
        train_data = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=Sentence.collate_fn,
            drop_last=False,
            num_workers=6
        )
        
        val_data = DataLoader(
            dataset=val_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=Sentence.collate_fn,
            drop_last=False,
            num_workers=6
        )
        
        test_data = DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=Sentence.collate_fn,
            drop_last=False,
            num_workers=6
        )
        
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = None
        
    else:  # bert模型
        # 使用BERT+CRF模型
        logging.info(f"使用BERT模型: {args.bert_model_name}")
        tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_name)
        model = BertCWS(tag2id, args.bert_model_name)
        
        # 创建BERT数据集
        train_dataset = BertSentence(x_train, y_train, tokenizer, tag2id, id2word)
        val_dataset = BertSentence(x_val, y_val, tokenizer, tag2id, id2word)
        test_dataset = BertSentence(x_test[:1000], y_test[:1000], tokenizer, tag2id, id2word)
        
        # 打印训练集标签统计信息
        logging.info("训练集标签统计:")
        train_dataset.print_label_stats()
        
        # 打印验证集标签统计信息
        logging.info("验证集标签统计:")
        val_dataset.print_label_stats()
        
        # 创建数据加载器
        train_data = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=BertSentence.collate_fn,
            drop_last=False,
            num_workers=6
        )
        
        val_data = DataLoader(
            dataset=val_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=BertSentence.collate_fn,
            drop_last=False,
            num_workers=6
        )
        
        test_data = DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=BertSentence.collate_fn,
            drop_last=False,
            num_workers=6
        )
        
        # 为BERT模型设置优化器和学习率调度器
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.bert_lr)
        
        # 创建学习率调度器
        total_steps = len(train_data) * args.max_epoch
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
    
    if use_cuda:
        model = model.to(device)
    
    for name, param in model.named_parameters():
        logging.debug('%s: %s, require_grad=%s' % (name, str(param.shape), str(param.requires_grad)))

    best_fscore = 0
    patience = args.early_stopping
    patience_counter = 0

    for epoch in range(args.max_epoch):
        step = 0
        log = []
        model.train()
        
        for batch in train_data:
            if args.model_type == 'lstm':
                sentence, label, mask, length = batch
                if use_cuda:
                    sentence = sentence.to(device)
                    label = label.to(device)
                    mask = mask.to(device)
                
                # forward
                loss = model(sentence, label, mask, length)
            else:  # bert模型
                input_ids, labels, attention_mask = batch
                if use_cuda:
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    attention_mask = attention_mask.to(device)
                
                # forward
                loss = model(input_ids, labels, attention_mask)
            
            log.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()

            step += 1
            if step % 1000 == 0:
                logging.debug('epoch %d-step %d loss: %f' % (epoch, step, sum(log)/len(log)))
                log = []
                
                # 每100步进行一次验证
                logging.info(f"Step {step}: 开始验证...")
                model.eval()
                if args.model_type == 'lstm':
                    val_precision, val_recall, val_fscore = evaluate(model, val_data, id2tag, use_cuda, args.model_type, device)
                else:
                    val_precision, val_recall, val_fscore = evaluate(model, val_data, id2tag, use_cuda, args.model_type, device, tokenizer_for_bert=tokenizer)
                logging.info(f"Step {step} 验证结果 - precision: {val_precision:.4f}, recall: {val_recall:.4f}, fscore: {val_fscore:.4f}")
                model.train()  # 切换回训练模式

        # 在验证集上评估
        if args.model_type == 'lstm':
            logging.info("在验证集上评估...")
            val_precision, val_recall, val_fscore = evaluate(model, val_data, id2tag, use_cuda, args.model_type, device)
            logging.info("验证集 - precision: %f, recall: %f, fscore: %f" % (val_precision, val_recall, val_fscore))
            logging.info("在测试集上评估...")
            test_precision, test_recall, test_fscore = evaluate(model, test_data, id2tag, use_cuda, args.model_type, device)
            logging.info("测试集 - precision: %f, recall: %f, fscore: %f" % (test_precision, test_recall, test_fscore))
        else: # bert
            logging.info("在验证集上评估...")
            val_precision, val_recall, val_fscore = evaluate(model, val_data, id2tag, use_cuda, args.model_type, device, tokenizer_for_bert=tokenizer)
            logging.info("验证集 - precision: %f, recall: %f, fscore: %f" % (val_precision, val_recall, val_fscore))
            logging.info("在测试集上评估...")
            test_precision, test_recall, test_fscore = evaluate(model, test_data, id2tag, use_cuda, args.model_type, device, tokenizer_for_bert=tokenizer)
            logging.info("测试集 - precision: %f, recall: %f, fscore: %f" % (test_precision, test_recall, test_fscore))
        
        # 保存最佳模型
        if val_fscore > best_fscore:
            best_fscore = val_fscore
            patience_counter = 0
            best_model_path = f"./save/best_model_{args.model_type}.pkl"
            torch.save(model, best_model_path)
            logging.info("发现更好的模型，已保存至 %s" % best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("早停：验证集性能 %d 个epoch未提升，停止训练" % patience)
                break
        
        # 每个epoch都保存一个模型
        path_name = f"./save/model_{args.model_type}_epoch{epoch}.pkl"
        torch.save(model, path_name)
        logging.info("模型已保存至 %s" % path_name)


def evaluate(model, data_loader, id2tag, use_cuda, model_type='lstm', device=None,tokenizer_for_bert=None):
    """评估模型性能"""
    entity_predict = set()
    entity_label = set()
    
    logging.debug(f"id2tag type: {type(id2tag)}, content: {id2tag}")

    # 在BERT模式下，从dataloader获取tokenizer实例 (如果dataloader是BertSentence的实例)
    # 或者直接通过参数传入。这里我们选择通过参数传入。
    if model_type == 'bert' and tokenizer_for_bert is None and hasattr(data_loader.dataset, 'tokenizer'):
        tokenizer_for_bert = data_loader.dataset.tokenizer


    with torch.no_grad():
        model.eval()
        cur = 0
        for batch in data_loader:
            if model_type == 'lstm':
                sentence, label, mask, length = batch
                if use_cuda:
                    sentence = sentence.to(device)
                    label = label.to(device)
                    mask = mask.to(device)
                predict = model.infer(sentence, mask, length) # predict is list of lists
                
                for i in range(len(length)): # iterate over batch
                    # sentence[i, :length[i]] is tensor of word_ids
                    # predict[i] is list of tag_ids
                    # label[i, :length[i]] is tensor of tag_ids
                    entity_split(sentence[i, :length[i]].tolist(), predict[i], id2tag, entity_predict, cur)
                    entity_split(sentence[i, :length[i]].tolist(), label[i, :length[i]].tolist(), id2tag, entity_label, cur)
                    cur += length[i]
            else:  # bert模型
                input_ids, labels, attention_mask = batch # all are tensors [batch_size, max_length]
                if use_cuda:
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    attention_mask = attention_mask.to(device)
                
                # predict is a list of lists (list of tag_id sequences for each item in batch)
                predict_batch = model.infer(input_ids, attention_mask) 
                
                for i in range(input_ids.size(0)): # Iterate over each sentence in the batch

                    current_input_ids_tensor = input_ids[i]
                    current_labels_tensor = labels[i]
                    current_attention_mask_tensor = attention_mask[i]
                    current_predictions_list = predict_batch[i] # This is already a list of tag IDs from CRF decode


                    tokens_for_processing = []
                    predictions_for_processing = []
                    true_labels_for_processing = []
                    
                    # Assuming tokenizer_for_bert is available
                    cls_token_id = tokenizer_for_bert.cls_token_id
                    sep_token_id = tokenizer_for_bert.sep_token_id
                    pad_token_id = tokenizer_for_bert.pad_token_id

                    pred_idx = 0 
                    for k in range(current_input_ids_tensor.size(0)):
                        if current_attention_mask_tensor[k].item() == 0: # PADDING
                            break # Stop at first padding token

                        token_id = current_input_ids_tensor[k].item()
                        
                        # Skip [CLS] and [SEP] for entity splitting
                        if token_id == cls_token_id or token_id == sep_token_id:
                            # If CRF included CLS/SEP in its output, advance pred_idx
                            if pred_idx < len(current_predictions_list): # Check boundary
                                pass # We will simply not add CLS/SEP to _for_processing lists
                            else: # Should not happen if pred_idx tracks correctly
                                pass
                        else: # Actual content tokens
                            if pred_idx < len(current_predictions_list): # Ensure pred_idx is valid
                                tokens_for_processing.append(token_id)
                                predictions_for_processing.append(current_predictions_list[pred_idx])
                                true_labels_for_processing.append(current_labels_tensor[k].item())
                            # Always advance pred_idx for any token that was part of the CRF input (i.e., attention_mask=1)
                    
                    unmasked_len = current_attention_mask_tensor.sum().item()
                    if len(current_predictions_list) != unmasked_len:
                        pass

                    valid_len = current_attention_mask_tensor.sum().item() # Number of non-padded tokens
                    
                    input_ids_slice = current_input_ids_tensor[:valid_len].tolist()
                    labels_slice = current_labels_tensor[:valid_len].tolist()
                    # predict_batch[i] is already a list, hopefully of length valid_len
                    predictions_slice = current_predictions_list # Assuming len(current_predictions_list) == valid_len

                    # Now, filter out CLS and SEP from these slices
                    final_tokens, final_preds, final_labels = [], [], []
                    for token_k_id, pred_k, label_k in zip(input_ids_slice, predictions_slice, labels_slice):
                        if token_k_id != cls_token_id and token_k_id != sep_token_id:
                            final_tokens.append(token_k_id)
                            final_preds.append(pred_k)
                            final_labels.append(label_k)
                    
                    if final_tokens: # If there's anything left after removing CLS/SEP
                        entity_split(final_tokens, final_preds, id2tag, entity_predict, cur)
                        # For true labels, entity_split handles -100 by KeyError -> maps to 'O' like behavior
                        entity_split(final_tokens, final_labels, id2tag, entity_label, cur)
                        cur += len(final_tokens)

        right_predict = [item for item in entity_predict if item in entity_label] # Corrected this line
        # Calculation of P, R, F seems okay
        precision, recall, fscore = 0, 0, 0
        if len(entity_label) > 0:
            recall = float(len(right_predict)) / len(entity_label)
        if len(entity_predict) > 0:
            precision = float(len(right_predict)) / len(entity_predict)
        
        if (precision + recall) > 0:
            fscore = (2 * precision * recall) / (precision + recall)
        
        return precision, recall, fscore


if __name__ == '__main__':
    set_logger()
    main(get_param())
