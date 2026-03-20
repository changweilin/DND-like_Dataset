# Dataset 格式品質報告

> 生成日期：2026-03-20
> 目的：評估現有四個 dataset 對 RL 訓練的適用性，並提出修正建議。

---

## 一、總覽

| Dataset | 數量（含val） | 適用 RL | 緊急問題 | 優先修正 |
|---|---|---|---|---|
| `lora_analyst` | 1610 | GRPO | 實體幻覺、英文文本品質差 | ★★★ |
| `lora_reasoning` | 1375 | GRPO | 無重大問題 | ★ |
| `lora_storyteller` | 1190 | DPO | 訓練材料非遊戲對話 | ★★ |
| `lora_translator` | 1606 | 暫緩 | 嚴重截斷（89%被截） | ★★★ |

---

## 二、lora_analyst — 嚴重品質問題

### 問題 1：假實體（幻覺來源）
**現象**：原始資料中將「Age」、「Campaign Four」等詞彙當作角色名稱標記。

**根因**：訓練文本來自 Critical Role Wiki 的百科式英文段落，例如：
```
"Shortly before the beginning of Campaign Four , Matthew Mercer clarified..."
```
此處 "Campaign Four" 是活動名稱，但資料若將其標記為角色/組織，就會教模型做出錯誤的 NER。

**建議修正**：
1. 過濾規則：排除以下類型的 token 被標記為實體：
   - 純數字或「Campaign + 數字」型字串
   - 常見英文一般名詞（Age, Era, City, Town, Guild...）
   - 長度 ≤ 2 字元的單字
2. 建議增加「驗證步驟」：每筆資料的 `gpt` 輸出中所有實體，必須能在 `human` 的輸入文本中找到對應字串。

**修正腳本示意**：
```python
import json, re

BLACKLIST = {"age", "era", "campaign", "chapter", "arc", "book", "season"}

def is_valid_entity(entity: str, source_text: str) -> bool:
    e = entity.strip().lower()
    if len(e) <= 2:
        return False
    if any(bl in e for bl in BLACKLIST):
        return False
    if e not in source_text.lower():
        return False  # 定位失敗
    return True
```

---

### 問題 2：訓練文本與實際遊戲場景脫節

**現象**：大量 `human` 欄位內容是 Wikipedia 風格的百科描述：
```
"Exandria is the name of the world on which the events of the first three
campaigns of Critical Role take place..."
```

**問題**：NER 任務的目標是解析**玩家輸入的遊戲對話**（如「艾莉絲拔出長劍衝向黑龍騎士團」），而不是靜態的百科知識文章。兩者語域完全不同，用百科文本訓練後，模型遇到動態遊戲對話時 NER 準確率會下降。

**建議修正**：
- 補充 **500+ 筆遊戲對話式 NER 資料**，格式為：
  - `human`: 玩家行動描述、NPC 對話、戰鬥旁白等短句（50-150 字）
  - `gpt`: 對應的 JSON 實體列表
- 可從 `lora_storyteller` 中抽取故事段落，再人工標記 NER，兩份資料形成互補。

---

## 三、lora_translator — 截斷嚴重問題

### 問題：89% 驗證集被 1024 token 截斷

**現象**（已知問題）：translator 資料的 p50 長度約 1366 tokens，遠超訓練上限 1024。

**後果**：
- SFT 訓練時，模型學到的是「被截斷的翻譯」等於正確答案
- GRPO 訓練時，獎勵函數（BLEU）計算的是截斷版本，無法反映完整翻譯品質
- 推理時模型可能養成「翻到一半就停」的壞習慣

**修正方案（擇一）**：

| 方案 | 作法 | 優缺點 |
|---|---|---|
| A. 切分長段 | 把每筆長資料切成 ≤ 700 token 的小段，各自對齊翻譯 | 資料量增加，但需人工重新對齊，工作量大 |
| B. 過濾短段 | 只保留 `human` + `gpt` 合計 ≤ 900 token 的資料 | 簡單易行，但資料量可能大幅縮水 |
| C. 增加 context length | 訓練時改用 `max_seq_length=2048`，需更多 VRAM | RTX 3060 可能 OOM，需測試 |

**推薦**：先做方案 B 過濾，評估剩餘資料量；若不足 500 筆再補充新資料。

---

## 四、lora_storyteller — 語域問題

### 問題：訓練素材為百科內容，非 RPG 敘事對話

**現象**：`human` 欄位為 Critical Role Wiki 的段落（如地理、曆法等），`gpt` 回應是同樣的百科延續。

**問題**：RPG 遊戲的故事接龍場景是：
- 玩家輸入行動 → 模型生成 GM 敘事回應
- 模型需要製造懸念、描寫戰鬥、回應角色選擇

而百科式的「繼續這個段落」與此完全不同。

**建議**：
1. 現有 storyteller 資料可保留用於**語言風格對齊**（文風、詞彙）
2. 補充 **GM 回應式資料**（格式為：玩家行動 → GM 敘事），來源可考慮：
   - 手工撰寫的 DND session log
   - AI 生成 + 人工審核的 GM 回應
3. 為 DPO 準備偏好對時，以 GM 回應資料為 `chosen`，百科延續為 `rejected`

---

## 五、lora_reasoning — 整體品質良好

### 現有優點
- 結構清晰：`【玩家行動】` + `【當前遊戲狀態】` → `【推理步驟】` + `【結論】`
- 數值可驗證：好感度計算有明確的算術依據
- 中英混合實體（NPC 名如 Gorin、地點如 The Obsidian Vault）符合遊戲語域

### 小幅改進建議

1. **增加任務觸發多樣性**：目前資料大量集中在「好感度 + 任務觸發」場景。建議補充：
   - 陣營關係變動（聯盟/敵對）
   - 隱藏事件解鎖（需滿足多個條件）
   - 失敗懲罰場景（好感度下降）

2. **新增邊界條件資料**：
   - 好感度已達上限（100），繼續贈禮的處理
   - 多個任務條件同時達成的優先權判斷

3. **JSON key 標準化**：不同資料筆的 JSON key 名稱有些許差異（如「新好感度」vs「NPC好感度」）。建議統一 schema。

---

## 六、修正優先順序建議

```
立即修正（影響 RL 訓練品質）：
  1. lora_analyst：加入實體定位驗證過濾器，移除假實體資料
  2. lora_translator：過濾超過 900 token 的資料，評估剩餘量

中期補充（提升場景適用性）：
  3. lora_analyst：補充 500+ 筆遊戲對話式 NER 資料
  4. lora_storyteller：補充 GM 回應式資料，並構造 DPO 偏好對

長期優化（深度改善）：
  5. lora_reasoning：補充多樣化場景（陣營、失敗懲罰）
  6. lora_translator：切分長段資料，重新對齊（需大量人工）
```

---

## 七、新 Task C（語意推理）資料集說明

`lora_reasoning.jsonl` 目前有 1375 筆，可以作為 Task C 的基礎。
原先 memory 中標記「No Task C dataset yet」已可更新。

雖然筆數已達訓練門檻（1000-3000 筆），但多樣性仍需擴充（見第五節）。
