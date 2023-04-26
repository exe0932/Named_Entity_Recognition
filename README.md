# 病種信息 (Disease information)
不使用手動標註或開源工具標註，使用代碼進行標註。
提取單一病種的相關信息。
使用pytorch框架，調用bert模型，訓練微調任務，簡稱NER模型
No manual labeling or tool labeling is used. The code is marked, and the relevant information of a single disease type* (cerebral infarction) is extracted. Use the pytorch framework, call the bert model, and train the fine-tuning task, NER for short.

## 工程結構 (Engineering structure)
- `cache` 訓練後的模型保存的位置 (The location where the trained model is saved)
- `data` 數據集 (dataset)
- `pretrain_model` 預訓練模型資料夾，加載不同版本，bert預訓練模型 (Pre-trained model folder, load different versions, bert pre-trained model)
  - `bert-base-chinese` 在huggingface官網下載中文版預訓練模型 (Download the Chinese version of the pre-training model on the official website of huggingface)
- `flask_api`部署至服務器專用接口 (Deploy to server private interface)
- `logs` 保存訓練日誌 (save training log)
- `model`目錄中是各個模塊的實現 (The directory is the implementation of each module)
  - `layers` 自實現並可優化的模塊 (Self-implementing and optimizeable modules)
    - `crf.py` crf.py模塊 (crf.py module)
  - `BertCrfForNer.py` 實現NER的下游訓練結構 (Implementing a downstream training structure for NER)
- `tools` 各工具類的實現 (Implementation of various tools)
  - `data_helpers.py` 各個下游任務的數據預處理及數據集構建模塊 (Data preprocessing and dataset building blocks for each downstream task)
  - `finetuning_argparse.py` 模型參數配置 (Model parameter configuration)
  - `log_helper.py` 日誌模塊 (log module)
  - `post_process.py` 後處裡模塊；用於定制化輸出 (Postprocessing module; for custom output)
  - `code_annotation.py` 自動標註與數據預處裡 (Automatic labeling and data preprocessing)
  - `regular_expression.py` 需標註的內容有哪些字樣 (What words should be marked)
- `run.py ` 運行訓練代碼/預測代碼 (Run the training code/prediction code)



