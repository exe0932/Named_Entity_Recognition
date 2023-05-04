# -*- coding: utf-8 -*-

# @Author : Eason_Chen

# @Time : 2023/4/24 下午 05:53

from flask import Flask
from flask_api.user import settings
# import flask_api

def disease_cerebral():
    app = Flask(__name__)

    app.register_blueprint(settings.user_bp)

    return app