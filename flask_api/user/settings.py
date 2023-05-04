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


    '''获取 json 资料'''
    ''' 401 请求不到资料 格式错误'''
    try:
        json_data = request.get_json()
    except:
        info = {'code': 401, 'msg': 'Failed to request data', 'data': []}
        return jsonify(info)

    '''获取 json key'''
    keys_names = tool.get_key(json_data)
    '''将 json 资料变成批量 存成 list'''
    get_text = tool.get_value(json_data, keys_names)
    ''' 定制化接口输出 '''
    interface = tool.get_key_value(get_text, model)
    return jsonify(interface)