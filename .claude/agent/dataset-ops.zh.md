# Dataset Ops (自動化排程與營運子代理) - 人類參考手冊

## 職責簡介
此 Agent 扮演自動化管線與部署的角色，負責一鍵自動化任務、定時更新與資料搬移。

## 常用指令
- `python pipeline.py`：啟動全自動工作流。
- `python scheduler.py --run-now`：啟動定時任務。
- `python transfer_datasets.py`：將資料同步至模型訓練目錄。

## 使用說明
- **操作核心**：當您需要從零開始更新整個資料庫，只需召喚此 Agent 執行 pipeline 即可。
- **設定檔**：主要變動將發生在 `transfer_config.yaml`。
