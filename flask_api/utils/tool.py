# -*- coding: utf-8 -*-

# @Author : Eason_Chen

# @Time : 2023/4/24 下午 05:54

def get_key(json_data):
    """
    输入 多个json 格式的文本内容，
    返回 存储文本内容的 key
    :param json_data: json key-value
    :return: key
    """
    keys_names = []
    for record in json_data:
        keys_name = []
        for keys in record:
            keys_name.append(keys)
        keys_names.append(keys_name)
    return keys_names

def get_value(json_data, keys_names):
    """
    输入 多个json 格式的文本内容 与 存储文本内容的 key，
    返回 多个指定内容 (列表) ，并在 (列表) 存储多个信息
    :param json_data: json key-value
    :param keys_names: value
    :return: 存储内容指定列表
    """
    get_text = list()
    for i, record in enumerate(json_data):
        if "text" in keys_names[i]:
            text = record["text"]
            get_text.append(text)
        else:
            text = ""
            get_text.append(text)
    return get_text

def get_key_value(get_text, model):
    """
    输入 承接 get_value 函数 的输出
    返回 接口二 指定内容
    :param get_travelContent: 挖掘的文本内容(列表存储后)
    :param model: 预训练模型
    :return: 重点场所信息平台接口
    """
    interface = []
    for i, text in enumerate(get_text):
        try:
            new_all_map_dict = model.predict(text)
            print(new_all_map_dict)
        except:
            return {'code': 402, 'msg': 'Failed to Mining data', 'data': []}
    #     # try:
    #     #     n = focus_place(df_data, get_travelContent[i], i)   # 输出接口2
    #     # except:
    #     #     return {'code': 403, 'msg': 'Failed to Post-processing', 'data': []}
        interface.append(new_all_map_dict)
    return {'code': 200, 'msg': 'succeed', 'data': interface}