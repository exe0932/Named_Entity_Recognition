# -*- coding: utf-8 -*-

# @Author : Eason_Chen

# @Time : 2023/4/17 下午 04:57


# from .layers.crf import CRF
from torchcrf import CRF
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from torch.nn.utils.rnn import pad_sequence

import torch
import torch.nn as nn

# '''導入育訓練tokenizer 和 model'''
# tokenizers = BertTokenizer.from_pretrained(r'D:\Unicom\competition\gitlab_test\202304\Single-disease-cerebral-infarction\pretrain_model\bert-base-chinese')
# model = BertModel.from_pretrained(r'D:\Unicom\competition\gitlab_test\202304\Single-disease-cerebral-infarction\pretrain_model\bert-base-chinese')
#
# # 將句子進行分詞
# inputs = tokenizers("Hello, my dog is cute", return_tensors="pt")
# # 模型輸出
# outputs = model(**inputs)
# # 提取輸出最後一個隱藏層狀態量
# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states)

'''使用CRF'''
class BertNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNER, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        # print(input_ids)
        # print(input_token_starts)
        # print("----------*----------")
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        # 去除[CLS]標籤等位置，獲得與label對齊的pre_label表示
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # 將sequence_output的pred_label維度padding到最大長度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)
        # 得到判別值
        logits = self.classifier(padded_sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs
        # contain: (loss), scores
        return outputs

# '''使用CRF'''
# class BertCrfForNer(BertPreTrainedModel):
#     def __init__(self, config):
#         super(BertCrfForNer, self).__init__(config)
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.crf = CRF(num_tags=config.num_labels, batch_first=True)
#         self.init_weights()
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None):
#         outputs =self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         sequence_output = outputs[0]
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)
#         outputs = (logits,)
#         if labels is not None:
#             loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
#             outputs =(-1*loss,)+outputs
#         return outputs # (loss), scores