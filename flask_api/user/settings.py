# -*- coding: utf-8 -*-

# @Author : Eason_Chen

# @Time : 2023/4/24 下午 05:53

from flask import Blueprint, jsonify, request
from flask_api.utils import tool
import run

user_bp = Blueprint('user', __name__)
model = run.TrainTestPredict()

@user_bp.route('/disease_cerbral', methods=['post'])
def disease_cerbral():


    '''獲取 json 資料'''
    ''' 401 請求不到資料 格是錯誤'''
    try:
        json_data = request.get_json()
    except:
        info = {'code': 401, 'msg': 'Failed to request data', 'data': []}
        return jsonify(info)

    '''獲取 json key'''
    keys_names = tool.get_key(json_data)
    '''將 json 資料變成批量 存成 list'''
    get_text = tool.get_value(json_data, keys_names)
    ''' 定製化接口輸出 '''
    interface = tool.get_key_value(get_text, model)
    return jsonify(interface)