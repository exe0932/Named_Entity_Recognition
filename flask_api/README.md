# 部署至服務器專用接口 (Deploy to server private interface)

## 工程結構 (Engineering structure)
- `config` 參數設置 (parameter settings)
  - `cfg.py` 端口與地址設置 (Port and Address Settings)
- `user`  藍圖與路由選擇 (Blueprints and Routing)
  - `settings.py` 定制化不同的路由模塊 (Customize different routing modules)
  - `views.py` 定制化不同的藍圖模塊 (Customize different blueprint modules)
- `utils` 後處里工具模塊集合 (A collection of tool modules in the backend)
  - `tool.py` 提取信息的後處裡代碼 (The code behind the extraction information)
- `run_disease_cerebral.py` 運行接口代碼 (Run the interface code)

## 使用方法 (Instructions)
- **運行流程 (run process)**
  -  step.1 在 `cfg.py`修改自己服務器的地址與端口`host/port`。 (Modify your server address and port `host/port` in `cfg.py`)
  -  step.2 在服務器上激活相關環境，`conda activate xxx`。 (Activate the relevant environment on the server, `conda activate xxx`)
  -  step.3 將項目運行至後台 `nohup python run_disease_cerebral.py &` (Run the project to the background `nohup python run_disease_cerebral.py &`)
- **請求的數據結構 (request data structure)**
  - [
    {
        "text" : "STAF 4分，神经功能缺损评分，26分。粗测听力正常，Rinne试验双侧均阴性，weber试验居中。"
    }
]
- **輸出的數據結構 (output data structure)**
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
- **錯誤編碼 (wrong code)**
  - `401`  請求不到資料；格式錯誤 (Data not requested; format error)
  - `402`  1.模型預測不到，請優化模型與標註 2. 該輸入數據並沒有需要預測字段 (1. The model cannot be predicted, please optimize the model and label 2. The input data does not need to predict the field)
  - `200`  模型預測成功，返回值 (The model prediction is successful, and the return value)