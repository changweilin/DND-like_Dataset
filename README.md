# DND-like Dataset

本專案旨在建立與維護專攻於跑團（TRPG, 如 Dungeons & Dragons, Pathfinder, Warhammer 等）、奇幻文學與各類世界觀設定的文字資料集。專案包含了從**網路爬蟲（Scraping）**、**資料萃取建置（Building）**，到**資料集轉換與驗證（Conversion & Validation）**，以及**自動化排程與搬移（Scheduling & Transferring）** 的完整管線，為 LLM 的 SFT 與 RL (DPO/GRPO) 階段提供高品質訓練資料。

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
    
    %% RL 後處理
    S -->|postprocess_rl.py| R[(data/finetune/cleaned/<br/>RLVR 專用數據集)]
    
    %% 模型專案轉移
    C -->|transfer_datasets.py| M[Model Training Project<br/>模型訓練專案目錄]
    S -->|transfer_datasets.py| M
    R -->|transfer_datasets.py| M
    
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
設計用來順序執行核心的資料處理流程：**抓取 (Scrape)** -> **建置 (Build)** -> **驗證 (Validate)** -> **RL 後處理 (Postprocess)** -> **搬移 (Transfer)**。

**用法範例：**
- 執行完整標準流程：`python pipeline.py`
- 包含自動搬移的完整流程：`python pipeline.py --transfer` (搬移前會詢問)
- 跳過抓取，僅重新建置、驗證並搬移：`python pipeline.py --skip-scrape --transfer`
- 在驗證後執行 RL 後處理：`python pipeline.py --postprocess`
- 僅執行 RL 後處理並搬移：`python pipeline.py --skip-scrape --skip-build --postprocess --transfer`
- 以 Fail-Fast 模式執行（遇到錯誤立刻中止）：`python pipeline.py --fail-fast`

### 2. `scheduler.py` (自動排程抓取與建置)
根據 `scraper_config.yaml` 內的設定，定期執行網路爬蟲（可設定同時啟動資料建置），適合放置於伺服器 24 小時運作。

**用法範例：**
- 依照設定檔週期運作：`python scheduler.py`
- 立即執行一次並繼續進入排程循環：`python scheduler.py --run-now`
- 僅執行一次就不再反覆循環：`python scheduler.py --once`

### 3. `transfer_datasets.py` (資料集自動搬移/同步)
將處理完成的資料集自動同步至外部的模型訓練專案。支援路徑自動對應與版本覆寫檢查。

**用法範例：**
- 單次執行資料集同步：`python transfer_datasets.py`
- 在背景定期連續同步：`python transfer_datasets.py --loop`

---

## 核心腳本說明

如果您需要手動干預每個階段，可以單獨執行以下腳本：

1. **scraper.py (抓取原始資料)**:
   依據 `scraper_config.yaml` 定義的資源（Wiki, AO3, Fanfiction 等）進行爬取，支援自動發現新頁面與更新檢查。

2. **build_dataset.py (建構 Alpaca 資料集)**:
   將原始文本轉換為包含 `instruction`, `input`, `output` 欄位的 JSONL 資料。支援多種採樣策略（如 `RandomChunkStrategy`）。

3. **convert_to_sharegpt.py (轉換 ShareGPT 格式)**:
   將單輪指令轉換為多輪對話風格的 ShareGPT 格式，以適應現代 LLM 訓練框架。

4. **validate_dataset.py (驗證資料品質)**:
   對生成的資料集進行靜態分析（文字長度、中英佔比、JSON 結構等）並輸出 `dataset_format_report.md`。

5. **postprocess_rl.py (RL 訓練數據後處理)**:
   針對 GRPO/DPO 強化學習進行特定的數據清洗、過濾與格式轉換。
   
   | 指令 | 說明 |
   |------|------|
   | `python postprocess_rl.py --task analyst` | NER 實體清洗與標註優化 |
   | `python postprocess_rl.py --task translator` | 過濾長度不匹配與異常翻譯 |
   | `python postprocess_rl.py --task storyteller` | 準備 DPO 偏好對前置資料 |
   | `python postprocess_rl.py --task reasoning` | 標準化思維鏈/JSON 輸出格式 |

   *提示：也可以透過 `pipeline.py --postprocess` 整合執行。*
