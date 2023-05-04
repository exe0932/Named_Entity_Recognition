# -*- coding: utf-8 -*-

# @Author : Eason_Chen

# @Time : 2023/4/7 下午 05:33

import re
import os
from tools import finetuning_argparse, regular_expression

class ProcessingData:

    def __init__(self):
        # cur_dir = '\\'.join(os.path.abspath(__file__).split('\\')[:-1])
        self.regular_expression = regular_expression.Regularexpression()
        config = finetuning_argparse.ModelConfig()

        # 文本路徑
        self.data_path = config.data_path

        # 載入文本內容
        self.texts = [i.strip() for i in open(self.data_path, encoding="utf-8") if i.strip()]

        print('Download Data ...')
        return

    '''建立訓練數據格式'''
    def build_data_format(self):
        out_list = list()
        for text in self.texts:
            in_dict = dict()
            in_list = list()
            in_dict['text'] = text
            for pattern in self.regular_expression.pattern():
                score_entity = re.search(pattern, text)
                # print(score_entity)
                if score_entity:
                    # # 打印評分實體在text中的起始位置和中止位置
                    # print("Start index: ", score_entity.start())
                    # print("End index: ", score_entity.end())
                    # print("Score entity: ", score_entity.group())
                    in_list.append([score_entity.start(), score_entity.end(), 'score'])
                    in_dict['labels'] = in_list
                else:
                    continue
            out_list.append(in_dict)
        print('Build Data Format  ...')
        return out_list

    '''训訓練集測試集劃分'''
    def data_split(self, datasets):
        trainset, testset = list(), list()
        for i in range(len(datasets)):
            if i <= len(datasets)*0.8:
                trainset.append(datasets[i])
            else:
                testset.append(datasets[i])
        print('Data Split  ...')
        return trainset, testset

    '''標註數據bio標籤代碼'''
    def creat_BIO(self, text, labels):
        tag_list = ['O' for i in range(len(text))]
        for start_index, end_index, key in labels:
            tag_list[start_index] = 'B-' + str(key)
            k = start_index + 1
            while k < end_index:
                tag_list[k] = 'I-' + str(key)
                k += 1
        return tag_list

    '''Bert最大長度512，用510字做切分數據'''
    def all_list(self, text, step=510):
        word_lis = list()
        for i, list_name in enumerate(text):
            if i % step == 0:
                word_lis.append(list())
            word_lis[-1].append(list_name)
        return word_lis

    '''標註數據bio標籤過程'''
    def data_preprocess(self, datasets=None, predict=None):
        if predict:
            word_lis, label_lis, all_lentexts = list(), list(), list()
            texts = list(predict)
            text = [text.replace('\xa0', ',').replace('\t', ',').replace('\ue0e7', ',').replace('\u3000', ',').replace('\n', ',').replace(' ', '-').replace('\u200b', ',') for text in texts]
            all_lentexts.append(len(text))
            tag_list = ['O' for i in range(len(text))]
            word_lis.extend(self.all_list(text))
            label_lis.extend(self.all_list(tag_list))
            return word_lis, label_lis, all_lentexts
        else:
            word_lis, label_lis = list(), list()
            for i in range(len(datasets)):
                label_entities = datasets[i].get('labels', None)
                text = datasets[i]['text']
                texts = list(text)
                text = [text.replace('\xa0', ',').replace('\t', ',').replace('\ue0e7', ',').replace('\u3000', ',').replace('\n',',').replace(' ', '-').replace('\u200b', ',') for text in texts]
                tag_list = self.creat_BIO(text, label_entities)
                label_lis.extend(self.all_list(tag_list))
                word_lis.extend(self.all_list(text))
            print('Creat BIO   ...')
            return word_lis, label_lis




    def main(self, content=None):
        if content:
            predict_word, predict_label, all_lentexts = self.data_preprocess(predict=content)
            return predict_word, predict_label, all_lentexts
        else:
            datasets = self.build_data_format()
            trainset, testset = self.data_split(datasets)
            train_word, train_label = self.data_preprocess(trainset)
            test_word, test_label = self.data_preprocess(testset)
            return train_word, train_label, test_word, test_label





if __name__ == '__main__':
    # text = '神清，构音清晰，时间、地点、人物定向力正常及远、近记忆力、判断力、理解力、计算力正常，自知力存在，情绪正常。颈无抵抗，Kernig、 Brudzinski征阴性。头颈面部及脊柱、四肢无畸形，头部无压痛，无强迫头位，听诊无血管杂音。粗测嗅觉及远、近视力正常，视野无缺损。眼底未查。眼睑无下垂，眼球位著居中，各向活动无受限，未见眼球震颤，双侧瞳孔等大等圆，直径3mm，直、间接对光反射灵敏，调节、辐辏反射存在。颜面部痛温触觉正常，张口无偏斜，咬颞肌有力，角膜反射正常。未见面肌抽搐，眼裂对称，双侧额纹、鼻唇沟无变浅，舌前2/3味觉存在。粗测听力正常，Rinne试验双侧均阴性，weber试验居中。洼田吞咽能力评定 5级、CHA2DS2-VASc  0分。发音正常，无饮水呛咳、吞咽困难，软腭上抬有力，悬雍垂居中，咽反射正常，舌后1/3味觉存在。转颈、耸肩对称有力。口腔中舌位居中，伸舌无偏斜，未见舌肌萎縮及纤颤，未见不自主运动，右利手，无肌肉萎缩，肌张力正常，在上肢近端肌力II+级，近端肌力IV级，余肢体肌力V-级。右侧指鼻试验、轮替试验不能配合完成，跟膝胜试验、Romberg’s征阴性，姿势、步态正常。躯干和四肢痛温触觉及关节位置觉、音叉振动觉、皮层觉正常。双侧桡骨膜、肱二头肌、肱三头肌反射及膝、踝反射 （++），Hoffmann、Rossolimo征（-），上、中、下腹壁反射及提睾反射、跖反射、肛门反射存在。双侧Babinski、Chaddock、 Oppenheim Gordon征（-），无强握反射及脊髓自主反射。皮肤色泽及皮温正常，无汗腺分泌障碍，大小便正常，皮肤划痕征阴性。'
    # data = '粗测听力正常，Rinne试验双侧均阴性，weber试验居中。洼田吞咽能力评定 5级、CHA2DS2-VASc 0分。'

    Pd = ProcessingData()
    a = Pd.main()
    # Pd.main()

