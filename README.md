# DND-like Dataset

本專案旨在建立與維護專攻於跑團（TRPG, 如 Dungeons & Dragons, Pathfinder, Warhammer 等）、奇幻文學與各類世界觀設定的文字資料集。專案包含了從**網路爬蟲（Scraping）**、**資料萃取建置（Building）**，到**資料集轉換與驗證（Conversion & Validation）**，以及**自動化排程與搬移（Scheduling & Transferring）** 的完整管線，能協助您快速準備給 LLM 微調（Fine-tuning）用的訓練資料集。

## 系統架構與自動化流程圖

以下流程圖說明了本專案中各個腳本與工具的交互運作方式：

```mermaid
graph TD
    %% 資料來源與獲取
    A[Web Sources<br/>Wiki / Fanfic / Forums] -->|scraper.py<br/>(& discovery_agent.py)| B[(data/raw/<br/>原始文本)]
    
    %% 第一階段建置
    B -->|build_dataset.py| C[(data/finetune/<br/>Alpaca 格式資料集)]
    
    %% 驗證與後續轉換
    C -->|validate_dataset.py| V[Validation Report<br/>資料品質報告]
    C -.->|export_hf.py| H[HuggingFace Hub<br/>開源發布]
    C -->|convert_to_sharegpt.py| S[(data/finetune/sharegpt/<br/>ShareGPT 格式資料集)]
    
    %% 模型專案轉移
    C -->|transfer_datasets.py| M[Model Training Project<br/>模型訓練專案目錄]
    S -->|transfer_datasets.py| M
    
    %% 自動化工具標記
    subgraph 自動化與排程生態 (Automation Ecosystem)
        SCHED[scheduler.py<br/>定期爬蟲與建置] -.->|定時觸發| A
        PIPE[pipeline.py<br/>端到端一鍵工作流] -.->|按序管理| B
        TRANS[transfer_datasets.py --loop<br/>自動資料同步] -.->|定時同步| M
    end
```

## 自動化工具 (Automation Tools) 使用說明

本專案提供了三種強大的自動化管理工具，可用於簡化資料集的生命週期維護：

### 1. `pipeline.py` (一鍵端到端工作流)
設計用來順序執行核心的資料處理流程：**抓取 (Scrape)** -> **建置 (Build)** -> **驗證 (Validate)** -> **後處理 (Postprocess, 可選)** -> **輸出 (Export, 可選)**。

**用法範例：**
- 執行完整標準流程：`python pipeline.py`
- 跳過抓取，僅重新建置與驗證：`python pipeline.py --skip-scrape`
- 只更新抓取新資料，不重新建置資料集：`python pipeline.py --skip-build`
- 僅針對特定分類（例如 `trpg`）執行管線：`python pipeline.py --category trpg`
- 以 Fail-Fast 模式執行（遇到錯誤立刻中止）：`python pipeline.py --fail-fast`
- 在驗證後執行 RL 後處理：`python pipeline.py --postprocess`
- 僅執行 RL 後處理（跳過前置步驟）：`python pipeline.py --skip-scrape --skip-build --postprocess`

### 2. `scheduler.py` (自動排程抓取與建置)
根據 `scraper_config.yaml` 內的設定，定期執行網路爬蟲（可設定同時啟動資料建置），非常適合放置於伺服器以 24 小時運作來持續更新知識庫。

**設定檔配置示例 (`scraper_config.yaml`):**
```yaml
schedule:
  interval_hours: 24   # 設定每 24 小時啟動一次執行週期
  auto_build: true     # 抓取完畢後是否自動觸發 build_dataset.py
```

**用法範例：**
- 依照設定檔週期運作：`python scheduler.py`
- 覆寫週期為每 6 小時一次：`python scheduler.py --interval 6`
- 立即執行一次並繼續進入排程循環：`python scheduler.py --run-now`
- 僅執行一次就不再反覆循環（適合排程軟體如 crontab 使用）：`python scheduler.py --once`

### 3. `transfer_datasets.py` (資料集自動搬移/同步)
專門用於將處理完成的資料集（如 Alpaca 或轉好的 ShareGPT 格式）自動搬移至外部的模型訓練專案。支援自動化排程並檢查版本覆寫。

**設定檔配置 (`transfer_config.yaml`):**
負責定義各種資料集對應的搬移目的地，以及搬移行為（如 `copy` 或 `move`）。

**用法範例：**
- 單次執行資料集同步：`python transfer_datasets.py`
- 在背景定期連續同步（依 `transfer_config.yaml` 內設定的 `interval_hours` 週期）：`python transfer_datasets.py --loop`

## 逐一資料集處理操作

如果您需要手動干預每個階段，可以單獨執行以下核心腳本：

1. **抓取原始資料**:
   `python scraper.py`
   依據 `scraper_config.yaml` 定義的資源列表進行爬取，結果存放於 `data/raw`。

2. **建構基礎資料集 (Alpaca 格式)**:
   `python build_dataset.py`
   將 `data/raw` 內的原始文本，依照不同資料提供策略重新包裝為包含 `instruction`, `input`, `output` 欄位的 JSONL 資料。結果存放於 `data/finetune/`。

3. **轉換為對話資料集 (ShareGPT 格式)**:
   `python convert_to_sharegpt.py`
   為了適應多輪對話及角色微調，此腳本將單輪的 Alpaca 指令格式轉換為 ShareGPT 格式（對應角色翻譯、分析、故事生成等用途）。輸出將位於 `data/finetune/sharegpt/` 下。

4. **驗證資料品質**:
   `python validate_dataset.py`
   可讀取生成的資料集，對文字長度、中英佔比等品質指標進行靜態分析並輸出報告，協助在實際訓練前排除無效資料。

5. **RL 訓練數據後處理 (postprocess_rl.py)**:
   `python postprocess_rl.py`
   對 ShareGPT 數據集進行清洗、過濾、格式轉換，為 GRPO/DPO 訓練做準備。結果輸出至 `data/finetune/sharegpt/cleaned/`。

   | 指令 | 說明 |
   |------|------|
   | `python postprocess_rl.py` | 處理全部四個資料集 |
   | `python postprocess_rl.py --task analyst` | 只處理 NER 實體清洗 |
   | `python postprocess_rl.py --task translator` | 只過濾超長翻譯記錄 |
   | `python postprocess_rl.py --task storyteller` | 只準備 DPO 偏好對格式 |
   | `python postprocess_rl.py --task reasoning` | 只標準化 JSON key |
   | `python postprocess_rl.py --stats` | 僅顯示統計，不寫入檔案 |
   | `python postprocess_rl.py --report report.md` | 輸出 markdown 處理報告 |
   | `python postprocess_rl.py --max-tokens 900` | 自訂 translator token 上限（預設 900）|

   也可整合進 pipeline：`python pipeline.py --skip-scrape --skip-build --postprocess`
