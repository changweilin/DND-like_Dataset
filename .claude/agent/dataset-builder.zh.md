# Dataset Builder (資料建構暨驗證子代理) - 人類參考手冊

## 職責簡介
此 Agent 專職將原始文本轉換為機器學習格式（Alpaca, ShareGPT），並負責生成驗證報告。

## 常用指令
- `python build_dataset.py`：轉換 Alpaca 格式。
- `python convert_to_sharegpt.py`：轉換對話格式。
- `python validate_dataset.py`：生成品質報告。

## 使用說明
- **資料處理**：本 Agent 會檢查 `data/raw/` 並產出至 `data/finetune/`。
- **品質管控**：產出後的 JSONL 檔案會自動經過靜態分析，確保沒有空值或亂碼。
