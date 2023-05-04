# -*- coding: utf-8 -*-

# @Author : Eason_Chen

# @Time : 2023/4/17 下午 02:19

from transformers import BertTokenizer, BertModel, BertForTokenClassification
from model import BertCrfForNer
from tools import finetuning_argparse, data_helpers, post_process, code_annotation
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

import os
import datetime
import time
import torch
import logging
import torch.nn as nn



class TrainTestPredict:

    def __init__(self):
        self.config = finetuning_argparse.ModelConfig()
        self.code_ann = code_annotation.ProcessingData()

    def train(self):
        model = BertCrfForNer.BertNER.from_pretrained(self.config.pretrain_path, num_labels=len(self.config.num_labels))
        now_time = datetime.datetime.now().strftime('%Y-%m-%d')
        model_save_path = os.path.join(self.config.model_save_dir, "ner_{}_model.pt".format(now_time))
        model = model.to(self.config.device)
        model.train()
        bert_tokenize = BertTokenizer.from_pretrained(self.config.pretrain_path).tokenize
        data_loader = data_helpers.LoadNameEntityRecognitionDataset(vocab_path=self.config.vocab_path,
                                                                    tokenizer=bert_tokenize,
                                                                    batch_size=self.config.batch_size,
                                                                    label2id=self.config.label2id
                                                                    )
        train_word, train_label, test_word, test_label = self.code_ann.main()
        train_iter, train_data, val_iter = data_loader.load_train_val_test_data(train_word, train_label, test_word, test_label)

        # Prepare optimizer
        if self.config.full_fine_tuning:
            # model.named_parameters(): [bert, classifier, crf]
            bert_optimizer = list(model.bert.named_parameters())
            classifier_optimizer = list(model.classifier.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.config.weight_decay},
                {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0},
                {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
                 'lr': self.config.learning_rate * 5, 'weight_decay': self.config.weight_decay},
                {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
                 'lr': self.config.learning_rate * 5, 'weight_decay': 0.0},
                {'params': model.crf.parameters(), 'lr': self.config.learning_rate * 5}
            ]
        # only fine-tune the head classifier
        else:
            param_optimizer = list(model.classifier.named_parameters())
            optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, correct_bias=False)
        train_steps_per_epoch = len(train_data) // self.config.batch_size
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=(self.config.epochs // 10) * train_steps_per_epoch,
                                                    num_training_steps=self.config.epochs * train_steps_per_epoch)

        # 畫曲線圖用
        train_loss_all = []
        train_acc_all = []
        val_loss_all = []
        val_acc_all = []

        # 保存最優模型用
        max_acc = 0
        for epoch in range(self.config.epochs):
            # 畫混淆矩陣用
            train_truel_all = []
            train_prel_all = []
            val_truel_all = []
            val_prel_all = []

            # 真實/預測/原文
            train_true_tags = []
            train_pred_tags = []
            train_sent_data = []
            start_time = time.time()
            losses, accs, precisions, recalls, f1scores = 0, 0, 0, 0, 0
            model.zero_grad()
            for idx, batch_samples in enumerate(train_iter):
                batch_data, batch_token_starts, batch_labels, batch_original_contents = batch_samples
                batch_data = batch_data.to(self.config.device)
                batch_token_starts = batch_token_starts.to(self.config.device)
                batch_labels = batch_labels.to(self.config.device)
                # print("batch_labels", batch_labels)
                # 返回原字樣
                train_sent_data.extend([[idx for idx in indices] for indices in batch_original_contents])
                batch_masks = batch_data.gt(0)  # get padding mask
                # print(batch_masks)
                label_masks = batch_labels.gt(-1)  # get padding mask, gt(x): get index greater than x

                # compute model output and loss
                loss, logits = model((batch_data, batch_token_starts),
                                     token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
                losses += loss.item()
                # clear previous gradients, compute gradients of all variables wrt loss
                model.zero_grad()
                loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.config.clip_grad)
                # performs updates using calculated gradients
                optimizer.step()
                scheduler.step()
                # (batch_size, max_len, num_labels)
                batch_output = model((batch_data, batch_token_starts),
                                     token_type_ids=None, attention_mask=batch_masks)[0]
                # (batch_size, max_len - padding_label_len)
                batch_output = model.crf.decode(batch_output, mask=label_masks)
                # print("batch_out:",batch_output)
                # (batch_size, max_len)
                batch_tags = batch_labels.to('cpu').numpy()
                train_pred_tags.extend([[self.config.id2label.get(idx) for idx in indices] for indices in batch_output])
                for train_pred_tag in train_pred_tags:
                    train_prel_all.extend(train_pred_tag)
                # (batch_size, max_len - padding_label_len)
                train_true_tags.extend(
                    [[self.config.id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])
                for train_true_tag in train_true_tags:
                    train_truel_all.extend(train_true_tag)

                # precision/recall/f1-score/support
                acc = accuracy_score(train_true_tags, train_pred_tags)
                precision = precision_score(train_true_tags, train_pred_tags)
                recall = recall_score(train_true_tags, train_pred_tags)
                f1score = f1_score(train_true_tags, train_pred_tags)
                accs += acc
                precisions += precision
                recalls += recall
                f1scores += f1score

                if idx % 10 == 0:
                    logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                                 f"Train loss : {loss.item():.3f}, Train acc: {acc:.3f}, "
                                 f"\nPrecision : {precision:.3f}, Recall: {recall:.3f}, "
                                 f"f1-score: {f1score:.3f}, ")
            end_time = time.time()

            train_loss = losses / len(train_iter)
            train_acc = accs / len(train_iter)
            train_precision = precisions / len(train_iter)
            train_recall = recalls / len(train_iter)
            train_f1 = f1scores / len(train_iter)

            # 畫曲線圖用
            train_loss_all.append(train_loss)
            train_acc_all.append(train_acc)

            logging.info(f"Epoch: {epoch}, "
                         f"Train loss: {train_loss:.3f}, Train acc: {train_acc:.3f}, "
                         f"\nPrecision : {train_precision:.3f}, Recall: {train_recall:.3f}, "
                         f"f1-score: {train_f1:.3f}, "
                         f"Epoch time = {(end_time - start_time):.3f}s")

            # precision/recall/f1-score/support
            # prfs = classification_report(train_true_tags, train_pred_tags)
            prfs = classification_report(train_truel_all, train_prel_all)
            logging.info(f"prfs on train:\n{prfs}")

            # 驗證
            model.eval()
            with torch.no_grad():
                val_true_tags = []
                val_pred_tags = []
                val_sent_data = []
                losses, accs, precisions, recalls, f1scores = 0, 0, 0, 0, 0
                start_time = time.time()
                for idx, batch_samples in enumerate(val_iter):
                    batch_data, batch_token_starts, batch_labels, batch_original_contents = batch_samples
                    batch_data = batch_data.to(self.config.device)
                    batch_token_starts = batch_token_starts.to(self.config.device)
                    batch_labels = batch_labels.to(self.config.device)

                    # 返回原字樣
                    val_sent_data.extend([[idx for idx in indices] for indices in batch_original_contents])
                    batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
                    label_masks = batch_labels.gt(-1)  # get padding mask, gt(x): get index greater than x

                    # compute model output and loss
                    loss = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
                    losses += loss.item()
                    # (batch_size, max_len, num_labels)
                    batch_output = model((batch_data, batch_token_starts),
                                         token_type_ids=None, attention_mask=batch_masks)[0]
                    # (batch_size, max_len - padding_label_len)
                    batch_output = model.crf.decode(batch_output, mask=label_masks)
                    # (batch_size, max_len)
                    batch_tags = batch_labels.to('cpu').numpy()
                    val_pred_tags.extend([[self.config.id2label.get(idx) for idx in indices] for indices in batch_output])
                    for val_pred_tag in val_pred_tags:
                        val_prel_all.extend(val_pred_tag)
                    # (batch_size, max_len - padding_label_len)
                    val_true_tags.extend(
                        [[self.config.id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])
                    for val_true_tag in val_true_tags:
                        val_truel_all.extend(val_true_tag)

                    acc = accuracy_score(val_true_tags, val_pred_tags)
                    precision = precision_score(val_true_tags, val_pred_tags)
                    recall = recall_score(val_true_tags, val_pred_tags)
                    f1score = f1_score(val_true_tags, val_pred_tags)
                    accs += acc
                    precisions += precision
                    recalls += recall
                    f1scores += f1score
                    if idx % 10 == 0:
                        logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(val_iter)}], "
                                     f"Val loss :{loss.item():.3f}, Val acc: {acc:.3f}, "
                                     f"\nPrecision : {precision:.3f}, Recall: {recall:.3f}, "
                                     f"f1-score: {f1score:.3f}, ")
                end_time = time.time()

            val_loss = losses / len(val_iter)
            val_acc = accs / len(val_iter)
            val_precision = precisions / len(val_iter)
            val_recall = recalls / len(val_iter)
            val_f1 = f1scores / len(val_iter)

            # 畫曲線圖用
            val_loss_all.append(val_loss)
            val_acc_all.append(val_acc)

            logging.info(f"Epoch: {epoch}, "
                         f"Val loss: {val_loss:.3f}, Val acc: {val_acc:.3f}, "
                         f"\nPrecision : {val_precision:.3f}, Recall: {val_recall:.3f}, "
                         f"f1-score: {val_f1:.3f}, "
                         f"Epoch time = {(end_time - start_time):.3f}s")

            # precision/recall/f1-score/support
            # prfs = classification_report(val_true_tags, val_pred_tags)
            prfs = classification_report(val_truel_all, val_prel_all)
            logging.info(f"prfs on val:\n{prfs}")

            # 保存最優模型
            if val_acc > max_acc:
                max_acc = val_acc
                torch.save(model.state_dict(), model_save_path)

    def evaluate(self, pre_iter, model, device, id2label):
        model.eval()
        with torch.no_grad():
            # 真實/預測/原文
            pre_true_tags = []
            pre_pred_tags = []
            pre_sent_data = []
            losses = 0
            for idx, batch_samples in enumerate(pre_iter):
                batch_data, batch_token_starts, batch_labels, batch_original_contents = batch_samples
                batch_data = batch_data.to(device)
                batch_token_starts = batch_token_starts.to(device)
                batch_labels = batch_labels.to(device)

                # 返回原字样
                pre_sent_data.extend([[idx for idx in indices] for indices in batch_original_contents])
                batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
                label_masks = batch_labels.gt(-1)  # get padding mask, gt(x): get index greater than x

                # compute model output and loss
                loss = model((batch_data, batch_token_starts),
                             token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
                losses += loss.item()
                # (batch_size, max_len, num_labels)
                batch_output = model((batch_data, batch_token_starts),
                                     token_type_ids=None, attention_mask=batch_masks)[0]
                # (batch_size, max_len - padding_label_len)
                batch_output = model.crf.decode(batch_output, mask=label_masks)
                # (batch_size, max_len)
                batch_tags = batch_labels.to('cpu').numpy()
                pre_pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
                # (batch_size, max_len - padding_label_len)
                pre_true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])
        return pre_true_tags, pre_pred_tags, pre_sent_data

    def predict(self, content):
        model = BertCrfForNer.BertNER.from_pretrained(self.config.pretrain_path, num_labels=len(self.config.num_labels))
        model_save_path = os.path.join(self.config.model_save_dir, "ner_2023-04-25_model.pt")
        loaded_paras = torch.load(model_save_path, map_location='cpu')
        model.load_state_dict(loaded_paras)
        model.to(self.config.device)

        bert_tokenize = BertTokenizer.from_pretrained(self.config.pretrain_path).tokenize
        # 如果要返回預標註功能，batch_size只能 = 1
        data_loader = data_helpers.LoadNameEntityRecognitionDataset(vocab_path=self.config.vocab_path,
                                                                    tokenizer=bert_tokenize,
                                                                    batch_size=self.config.batch_size,
                                                                    label2id=self.config.label2id
                                                                    )
        predict_word, predict_label, all_lentexts = self.code_ann.main(content=content)
        predict_iter = data_loader.load_train_val_test_data(predict_word=predict_word, predict_label=predict_label)

        pre_true_tags, pre_pred_tags, pre_sent_data = self.evaluate(predict_iter, model, self.config.device, self.config.id2label)
        all_map_dict, all_texts, all_labels, all_coordinates_names = post_process.get_ner(pre_pred_tags, pre_sent_data,
                                                                                          all_lentexts,self.config.labels)
        new_all_map_dict = post_process.translation(all_map_dict)
        return new_all_map_dict










if __name__ == '__main__':
    Main = TrainTestPredict()
    config = finetuning_argparse.ModelConfig()
    code_ann = code_annotation.ProcessingData()
    # Main.train()
    print(Main.predict(content='洼田吞咽功能障碍评价 2级.mRS,3级。颜面部痛温触觉正常，张口无偏斜，咬颞肌有力，角膜反射正常。'))