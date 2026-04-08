"""
review_standards.py — 通用 Code Review 標準模板

新專案使用方式：
  1. 複製此檔案至專案的 scripts/ 目錄。
  2. 在 ai_reviewer.py 中 from review_standards import build_review_prompt。
  3. 呼叫 build_review_prompt(diff, focus_area, project_context) 取得完整 prompt。

可自訂：
  - 修改 REVIEW_DIMENSIONS 新增或移除審核維度。
  - 呼叫時傳入 project_context 注入專案特定知識。
"""

# ── 審核維度定義 ──────────────────────────────────────────────────────────────
# 每個維度包含 name（標題）與 checks（細項清單）。
# 可自由新增、移除或調整順序。
REVIEW_DIMENSIONS = [
    {
        "name": "🔐 安全性",
        "checks": [
            "硬編碼金鑰、Token 或密碼（應改用環境變數 os.getenv()）",
            "Prompt 注入：diff 或外部資料是否直接拼接進 LLM prompt，而未使用邊界標籤（如 <diff>...</diff>）",
            "命令注入：subprocess 呼叫是否使用 shell=True 或字串拼接；建議改用 list 參數",
            "路徑遍歷：使用者輸入是否未驗證就用於檔案路徑（建議 Path.resolve() + 白名單）",
            "敏感資料洩露：API 金鑰、個資是否可能出現在日誌、錯誤訊息或 git diff 中",
            "依賴安全：是否引入已知有漏洞的套件版本（可用 pip audit 檢查）",
        ],
    },
    {
        "name": "🛡️ 可靠性與錯誤處理",
        "checks": [
            "裸 except 或 except Exception: pass（吞掉所有錯誤，掩蓋真正問題）",
            "資源保證清理：檔案、DB 連線、GPU memory 應使用 try/finally 或 with 語法",
            "重試邏輯合理性：是否有最大重試次數上限；建議指數退避而非固定等待",
            "超時機制：外部 API 呼叫、subprocess、網路請求是否設定 timeout",
            "優雅降級：主要流程失敗時是否有明確 fallback（而非直接 sys.exit 或靜默忽略）",
            "並發安全：多執行緒或多進程場景下是否有競態條件（shared state、文件鎖）",
        ],
    },
    {
        "name": "🤖 AI / Agent 特有",
        "checks": [
            "Token 使用效率：Prompt 是否有冗餘文字；長文本是否已截斷（建議 diff[:30000]）",
            "模型選擇適當性：任務是否真的需要 Pro/Opus？輕量任務改用 Flash/Haiku 可節省大量配額",
            "LLM 輸出驗證：JSON 回傳是否有結構驗證與預設值，避免幻覺造成 KeyError/AttributeError",
            "Agent 無限循環：遞迴或 while True 是否有最大迭代次數或逾時保護",
            "配額管理：是否整合 RPM/RPD 限流機制（api_tracker 或等效方案）",
            "並行任務防衝突：.ai_working 或等效鎖定機制是否在 mid-task 期間阻擋其他寫入",
            "Prompt 版本管理：Prompt 變動是否可追蹤（避免靜默改變模型行為）",
        ],
    },
    {
        "name": "⚡ 效能",
        "checks": [
            "大型檔案（模型權重、資料集）是否串流讀取，而非一次載入記憶體",
            "GPU/VRAM 管理：模型使用完畢後是否呼叫 unload / del + torch.cuda.empty_cache()",
            "重複 I/O：是否在迴圈內反覆讀取同一檔案或呼叫同一 API（應快取結果）",
            "批次處理：可批次的操作是否已批次化，而非逐筆呼叫 API",
            "演算法複雜度：是否有 O(n²) 以上的瓶頸（例如雙重迴圈做全局搜尋）",
            "不必要的 subprocess 呼叫：能用 Python 原生完成的操作不應 fork 子進程",
        ],
    },
    {
        "name": "🧹 程式碼品質",
        "checks": [
            "命名一致性：snake_case 函式/變數、PascalCase 類別、語意明確不用無意義縮寫",
            "魔法數字/字串：30000、600、300 等常數是否已提取為命名常數（方便日後維護）",
            "死碼與冗餘 import：未使用的變數、函式、import 是否已清理",
            "單一職責：單一函式是否超過 60 行或承擔超過一項職責",
            "潛在 Bug：off-by-one 錯誤、型別假設（str vs bytes）、None 未處理、整數除法",
            "全域狀態：是否過度使用 global 變數（難以測試、易產生隱性副作用）",
        ],
    },
    {
        "name": "💾 資料處理",
        "checks": [
            "檔案讀寫是否明確指定 encoding='utf-8'（Windows 預設 cp950，跨平台時常出錯）",
            "大型資料集是否分批/分塊處理（避免 OOM）",
            "格式轉換（JSON/JSONL/Parquet/CSV）是否有錯誤容忍與格式驗證",
            "輸出檔案路徑是否避免覆蓋重要資料（建議加時間戳或版本號）",
            "資料清洗邏輯是否冪等（重複執行不會產生重複資料或損壞資料）",
        ],
    },
    {
        "name": "🔧 DevOps / Git 衛生",
        "checks": [
            "大型二進位檔是否不慎納入 diff（.safetensors .bin .gguf .pt .ckpt .parquet .csv）",
            "新增腳本或目錄是否需要對應的 .gitignore 規則",
            "環境變數管理：設定值是否透過 os.getenv() 讀取，而非硬編碼在程式碼中",
            "Secrets 防洩漏：.env、credentials、API key 是否可能被 commit",
            "CI/CD 工作流：新功能是否需要更新 GitHub Actions workflow 或觸發條件",
        ],
    },
]


def build_review_prompt(diff: str, focus_area: str, project_context: str = "") -> str:
    """
    建立標準化 Code Review Prompt。

    Args:
        diff:            git diff 內容（建議先截斷至 30000 字元）
        focus_area:      本次 PR 主要領域（例如 "LLM 訓練流程與參數設置"）
        project_context: 選填，專案特定背景（技術棧、架構約定、已知限制等）

    Returns:
        完整可直接傳給 LLM 的 prompt 字串。
    """
    dimensions_text = ""
    for i, dim in enumerate(REVIEW_DIMENSIONS, 1):
        checks = "\n".join(f"   - {c}" for c in dim["checks"])
        dimensions_text += f"{i}. **{dim['name']}**\n{checks}\n\n"

    project_section = (
        f"\n**專案背景**：\n{project_context}\n" if project_context else ""
    )

    return f"""你是一位資深的 Python AI 系統架構師與 Code Reviewer。請審核以下 Pull Request 的差異。
本次 PR 主要領域：【{focus_area}】
{project_section}
請逐一檢查下列 {len(REVIEW_DIMENSIONS)} 個維度，對**有問題的項目**給出具體說明與修正建議；若該維度無問題則以一句話帶過。

{dimensions_text}
**輸出格式要求**：
- 使用繁體中文
- 每個維度使用粗體標題 + 條列式說明
- 🚨 標記嚴重問題（安全漏洞、資料遺失風險、服務中斷風險）
- 💡 標記建議改進（非必要但推薦）
- ✅ 標記無問題的維度（一句話即可）
- 結尾提供 1-2 句整體評語與優先修正建議

<diff>
{diff}
</diff>"""
