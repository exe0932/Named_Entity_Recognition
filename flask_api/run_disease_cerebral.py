# -*- coding: utf-8 -*-

# @Author : Eason_Chen

# @Time : 2023/4/21 下午 03:01

from flask import request, jsonify

# 服务器用根目录
# import sys
# sys.path.append('../../')
import argparse
from user import views
from config import cfg

app = views.disease_cerebral()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='test123456789')
    parser.add_argument('--host', default=cfg.HOST, type=str, help='server ip')
    parser.add_argument('--port', default=cfg.PORT, type=int, help='server port')
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=True)