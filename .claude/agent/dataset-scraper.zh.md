# Dataset Scraper (資料爬蟲子代理) - 人類參考手冊

## 職責簡介
此 Agent 專注於處理網頁數據獲取任務。它負責發掘新站點 (`discovery_agent.py`) 並自動抓取內容至 `data/raw/` 目錄。

## 常用指令
- `python scraper.py`：啟動主爬蟲。
- `python discovery_agent.py`：自動發掘潛在語料網站。

## 使用說明
- **設定檔**：主要修改 `scraper_config.yaml` 來定義爬蟲規則。
- **擋爬處理**：遇到 403 報錯時，此 Agent 會自動更新 Header 並嘗試重連。
