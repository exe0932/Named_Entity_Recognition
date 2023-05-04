# -*- coding: utf-8 -*-

# @Author : Eason_Chen

# @Time : 2023/4/17 下午 12:43


import os
import torch
import logging
from .log_helper import logger_init

# import argparse
# def get_argparse():
#     parser = argparse.ArgumentParser()
#     return parser

'''加載模型序要的參數'''
class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(self.project_dir , 'data', 'V2.0.txt')
        self.pretrain_path = os.path.join(self.project_dir, 'pretrain_model', 'bert-base-chinese')
        self.vocab_path = os.path.join(self.pretrain_path, 'vocab.txt')
        self.model_save_dir = os.path.join(self.project_dir, 'cache', 'Ner')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.figure_save_dir = os.path.join(self.project_dir, 'picture', 'Ner')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')

        self.batch_size = 1
        self.epochs = 1
        # hyper-parameter
        self.full_fine_tuning = True  # 是否對整個BERT進行fine tuning
        self.weight_decay = 0.01
        self.learning_rate = 3e-5
        self.clip_grad = 5

        self.labels = ['score']
        self.class_label = ["O", "B-score", "I-score"]
        self.label2id = {"O": 0, "B-score": 1, "I-score": 2}
        self.num_labels = self.label2id
        self.id2label = {_id: _label for _label, _id in list(self.label2id.items())}

        logger_init(log_file_name='NER', log_level=logging.INFO, log_dir=self.logs_save_dir)

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        if not os.path.exists(self.figure_save_dir):
            os.makedirs(self.figure_save_dir)

if __name__ == '__main__':
    a = ModelConfig()

