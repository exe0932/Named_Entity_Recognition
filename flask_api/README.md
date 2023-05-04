# 部署至服务器专用接口 (Deploy to server private interface)

## 工程结构 (Engineering structure)
- `config` 参数设置 (parameter settings)
  - `cfg.py` 端口与地址设置 (Port and Address Settings)
- `user`  蓝图与路由选择 (Blueprints and Routing)
  - `settings.py` 定制化不同的路由模块 (Customize different routing modules)
  - `views.py` 定制化不同的蓝图模块 (Customize different blueprint modules)
- `utils` 后处里工具模块集合 (A collection of tool modules in the backend)
  - `tool.py` 提取信息的后处里代码 (The code behind the extraction information)
- `run_disease_cerebral.py` 运行接口代码 (Run the interface code)

## 使用方法 (Instructions)
- **运行流程 (run process)**
  -  step.1 在 `cfg.py`修改自己服务器的地址与端口`host/port`。 (Modify your server address and port `host/port` in `cfg.py`)
  -  step.2 在服务器上激活相关环境，`conda activate xxx`。 (Activate the relevant environment on the server, `conda activate xxx`)
  -  step.3 将项目运行至后台 `nohup python run_disease_cerebral.py &` (Run the project to the background `nohup python run_disease_cerebral.py &`)
- **请求的数据结构 (request data structure)**
  - [
    {
        "text" : "STAF 4分，神经功能缺损评分，26分。粗测听力正常，Rinne试验双侧均阴性，weber试验居中。"
    }
]
- **输出的数据结构 (output data structure)**
  - {
    "code": 200,
    "data": [
        [
            {
                "score": [
                    {
                        "STAF": "4分"
                    },
                    {
                        "神经功能缺损评分": "26分"
                    }
                ],
                "text": [
                    "STAF-4分，神经功能缺损评分，26分。粗测听力正常，Rinne试验双侧均阴性，weber试验居中。"
                ]
            }
        ]
    ],
    "msg": "succeed"
}
- **错误编码 (wrong code)**
  - `401`  请求不到资料；格式错误 (Data not requested; format error)
  - `402`  1.模型预测不到，请优化模型与标注 2. 该输入数据并没有需要预测字段 (1. The model cannot be predicted, please optimize the model and label 2. The input data does not need to predict the field)
  - `200`  模型预测成功，返回值 (The model prediction is successful, and the return value)