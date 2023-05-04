# -*- coding: utf-8 -*-

# @Author : Eason_Chen

# @Time : 2023/4/18 下午 07:47

def new_extend(datas, data):
    """
    将切分的文本长度拼接
    :param data:[['3', '2', '1', '0', ',', '女',......, '1', '2', '月', '2', '7', '日'],......,['3', '2', '1', '1', ',', '男']]
    :return:['3', '2', '1', '0', ',', '女', '，', '5',......,'4', '0', '5', '6',.......,'病', '例', '。']
    """
    for x in data:
        datas.extend(x)
    return datas

def new_data(all_indexs, all_datas):
    """
    还原原文本长度，例如原文本长度超过512个字
    :param all_indexs:[162, 510, 206, 99, 207, 182, 126, 159, 148, 177, 184, 145, 373, 391,......,643, 97, 644]
    :param all_datas:['3', '2', '1', '0', ',', '女', '，', '5',......,'4', '0', '5', '6',.......,'病', '例', '。']
    :return:[['3', '2', '1', '0', ',', '女',......, '1', '2', '月', '2', '7', '日'],......,['3', '2', '1', '1', ',', '男']]
    """
    b = 0
    new_test_sent_datas = list()
    for all_index in all_indexs:
        a = all_datas[b:b+int(all_index)]
        b += int(all_index)
        new_test_sent_datas.append(a)
    return new_test_sent_datas

def get_ner(pre_pred_tags, pre_sent_data, all_lentexts, labels):
    pre_prel_all = []
    all_pre_sent_datas = []
    pre_prel_all = new_extend(pre_prel_all, pre_pred_tags)
    all_pre_sent_datas = new_extend(all_pre_sent_datas, pre_sent_data)
    new_pre_sent_datas = new_data(all_lentexts, all_pre_sent_datas)
    new_pre_prel_datas = new_data(all_lentexts, pre_prel_all)
    all_map_dict, all_texts, all_labels, all_coordinates_names = pre_label(new_pre_prel_datas, new_pre_sent_datas, labels)
    return all_map_dict, all_texts, all_labels, all_coordinates_names

def pre_label(y_pred, data, labels):
    """
    用途 命名实体识别结果输出 与 生成预标注功能
    :param y_pred:

    [['B-district', 'I-district', 'I-district', 'B-district', 'I-district', 'I-district',
    'B-district', 'I-district', 'I-district', 'B-district', 'I-district', 'I-district',
    'B-place', 'I-place', 'I-place', 'I-place', 'I-district', 'I-district', 'I-district',
    'I-district']]

    :param data:

    [['广', '东', '省', '中', '山', '市', '三', '乡', '镇', '古', '鹤', '村', '四', '队', '福', '群',
     '东', '街', '9', '号']]

    :param labels:

    ['time', 'person', 'district', 'place', 'vehicle', 'mask']

    :return:
    all_map_dicts:

    [{'text': ['广东省中山市三乡镇古鹤村四队福群东街9号'], 'time': [], 'person': [],
     'district': ['广东省', '中山市', '三乡镇', '古鹤村'], 'place': ['四队福群东街9号'],
      'vehicle': [], 'mask': []}]

    all_texts:

    [['广东省中山市三乡镇古鹤村四队福群东街9号']]

    all_coordinates:

    [[[0, 3, 'district'], [3, 6, 'district'], [6, 9, 'district'], [9, 12, 'district'],
     [12, 20, 'place']]]

    all_coordinates_names:

    [{'time': [], 'person': [],
     'district': [[0, 3, '广东省'], [3, 6, '中山市'], [6, 9, '三乡镇'], [9, 12, '古鹤村']],
      'place': [[12, 20, '四队福群东街9号']], 'vehicle': [], 'mask': []}]
    """

    sentences = []
    for idx, p in enumerate(y_pred):
        sentence = []
        sentence.append(''.join(data[idx]))
        sentences.append(sentence)
    entity_name = ""
    flag = []
    all_coordinates = []
    all_coordinates_names = []
    all_texts = []
    all_map_dicts = []
    visit = False
    for i, text in enumerate(sentences):
        count, start, end = 0, 0, 0
        coordinates = []
        coordinates_name = dict()
        map_dict = dict()
        map_dict['text'] = text
        for label in labels:
            map_dict[label] = []
            coordinates_name[label] = []
        for char, tag in zip(sentences[i][0], y_pred[i]):
            # print(char,tag)
            if tag[0] == "B":
                if entity_name != "" and len(entity_name) == 1:
                    x = dict((a, flag.count(a)) for a in flag)
                    y = [k for k, v in x.items() if max(x.values()) == v]
                    for label in labels:
                        if y[0] in label:
                            map_dict[label].append(entity_name)
                            coordinates.append([start, start + 1, y[0]])
                            coordinates_name[label].append([start, start + 1, entity_name])
                            flag.clear()
                            entity_name = ""
                elif entity_name != "":
                    x = dict((a, flag.count(a)) for a in flag)
                    y = [k for k, v in x.items() if max(x.values()) == v]
                    for label in labels:
                        if y[0] in label:
                            map_dict[label].append(entity_name)
                            coordinates.append([start, end, y[0]])
                            coordinates_name[label].append([start, end, entity_name])
                            flag.clear()
                            entity_name = ""
                visit = True
                entity_name += char
                flag.append(tag[2:])
                start = count
                count += 1
            elif tag[0] == "I" and visit:
                entity_name += char
                flag.append(tag[2:])
                end = count + 1
                count += 1
            else:
                if entity_name != "" and len(entity_name) == 1:
                    x = dict((a, flag.count(a)) for a in flag)
                    y = [k for k, v in x.items() if max(x.values()) == v]
                    for label in labels:
                        if y[0] in label:
                            map_dict[label].append(entity_name)
                            coordinates.append([start, start + 1, y[0]])
                            coordinates_name[label].append([start, start + 1, entity_name])
                elif entity_name != "":
                    x = dict((a, flag.count(a)) for a in flag)
                    y = [k for k, v in x.items() if max(x.values()) == v]
                    for label in labels:
                        if y[0] in label:
                            map_dict[label].append(entity_name)
                            coordinates.append([start, end, y[0]])
                            coordinates_name[label].append([start, end, entity_name])
                flag.clear()
                visit = False
                entity_name = ""
                count += 1
        if entity_name != "":
            x = dict((a, flag.count(a)) for a in flag)
            y = [k for k, v in x.items() if max(x.values()) == v]
            for label in labels:
                if y[0] in label:
                    map_dict[label].append(entity_name)
                    coordinates.append([start, end, y[0]])
                    coordinates_name[label].append([start, end, entity_name])
        all_coordinates.append(coordinates)
        all_coordinates_names.append(coordinates_name)
        all_texts.append(text)
        all_map_dicts.append(map_dict)
    return all_map_dicts, all_texts, all_coordinates, all_coordinates_names

def translation(all_map_dicts):
    for all_map_dict in all_map_dicts:
        scores = all_map_dict['score']
        try:
            scores_1 = [score.replace('-', ',').replace('，', ',') for score in scores]
            new_scores = [{score_1.split(',')[0] : score_1.split(',')[1]} for score_1 in scores_1]
            all_map_dict['score']= new_scores
        except:
            return all_map_dicts
    return all_map_dicts