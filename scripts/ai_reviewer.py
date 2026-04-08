"""
ai_reviewer.py — Gemini 動態 Code Reviewer (Dataset 專屬版)

功能：
  1. 呼叫 Gemini 2.0 Flash 進行程式碼審核（7 個維度全面標準）。
  2. Gemini 頻繁失敗時等 10 分鐘後重試。
  3. 自動將評論發送回 PR 頁面。
"""

import os
import sys
import json
import time
import subprocess
import requests
from datetime import datetime
from pathlib import Path
import google.generativeai as genai

# 審核標準模板
_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    from review_standards import build_review_prompt
except ImportError:
    build_review_prompt = None

# 配置
API_KEY = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO = os.getenv("GITHUB_REPOSITORY")
EVENT_PATH = os.getenv("GITHUB_EVENT_PATH")
BASE_REF = os.getenv("GITHUB_BASE_REF", "main")

if not GITHUB_TOKEN or not REPO or not EVENT_PATH:
    print("[ERROR] Missing environment variables (GITHUB_TOKEN, GITHUB_REPOSITORY, GITHUB_EVENT_PATH).")
    sys.exit(1)

if not API_KEY:
    print("[WARNING] GEMINI_API_KEY not set, will attempt Claude fallback only.")
else:
    genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash") if API_KEY else None

MODEL_ID = "gemini-2.0-flash"

def _wait_for_quota():
    """RPM 超限 → 等到下一分鐘；RPD 超限 → 回傳 False（跳過今日）。"""
    try:
        from api_tracker import LIMITS, _load_log, _today, _now_minute
    except ImportError:
        return True

    limit = LIMITS.get(MODEL_ID, {})
    log = _load_log()
    today, minute = _today(), _now_minute()
    day_count = log.get(today, {}).get(MODEL_ID, {}).get("count", 0)
    min_count = log.get(today, {}).get(MODEL_ID, {}).get("minutes", {}).get(minute, 0)

    if day_count >= limit.get("rpd", float("inf")):
        print(f"[Quota] {MODEL_ID} 今日 {day_count}/{limit['rpd']} RPD 已滿，跳過。")
        return False

    rpm = limit.get("rpm")
    if rpm and min_count >= rpm:
        wait_sec = 62 - datetime.now().second
        print(f"[Quota] RPM 上限 {min_count}/{rpm}，等待 {wait_sec}s...")
        time.sleep(wait_sec)

    return True

def _record_call():
    try:
        from api_tracker import record_call
        record_call(MODEL_ID)
    except ImportError:
        pass

# Gemini 失敗追蹤（累計 3 次 或 1 分鐘內 2 次 → 等 10 分鐘後重試一次）
_gemini_errors_recent = []
_gemini_errors_total  = 0

def _call_ai(prompt):
    """呼叫 Gemini；遵守 RPM/RPD 配額，頻繁失敗時等 10 分鐘後重試一次。"""
    global _gemini_errors_recent, _gemini_errors_total
    now = time.time()
    _gemini_errors_recent = [t for t in _gemini_errors_recent if now - t < 60]

    throttled = model is None or _gemini_errors_total >= 3 or len(_gemini_errors_recent) >= 2

    if not throttled:
        if not _wait_for_quota():
            return None
        try:
            resp = model.generate_content(prompt)
            _record_call()
            return resp.text
        except Exception as e:
            _gemini_errors_recent.append(time.time())
            _gemini_errors_total += 1
            print(f"[Warning] Gemini 失敗（累計 {_gemini_errors_total} 次）: {e}")
            throttled = _gemini_errors_total >= 3 or len(_gemini_errors_recent) >= 2

    if throttled:
        print("[Info] Gemini 頻繁失敗，等待 10 分鐘後重試...")
        time.sleep(600)
        if not _wait_for_quota():
            return None
        try:
            resp = model.generate_content(prompt)
            _record_call()
            _gemini_errors_total = 0
            _gemini_errors_recent.clear()
            print("[Info] 重試成功。")
            return resp.text
        except Exception as e:
            print(f"[Error] 重試仍失敗，跳過本次 review: {e}")
            return None

    return None

def get_pr_diff():
    try:
        diff = subprocess.check_output(["git", "diff", f"origin/{BASE_REF}...HEAD"]).decode("utf-8")
        return diff[:30000]
    except Exception as e:
        print(f"[ERROR] Getting diff failed: {e}")
        return ""

def post_comment(comment):
    with open(EVENT_PATH, 'r') as f:
        pr_number = json.load(f)['number']

    url = f"https://api.github.com/repos/{REPO}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "body": f"### 🤖 Antigravity AI Code Review\n\n{comment}\n\n---\n*本評論由 Gemini 2.0 Flash 自動產生*"
    }
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 201:
        print("[SUCCESS] Comment posted to PR.")
    else:
        print(f"[FAILED] Failed to post comment: {res.text}")

PROJECT_CONTEXT = """
技術棧：Python、requests/BeautifulSoup（爬蟲）、Gemini API（資料生成/清洗）、
YAML config 驅動（scraper_config.yaml / transfer_config.yaml）。
資料格式：Alpaca（instruction/input/output）與 ShareGPT（conversations）雙格式輸出。
架構約定：scraper.py → data/raw/，build_dataset.py → data/finetune/，
validate_dataset.py 驗證 token 數與語言比例；pipeline.py 串接全流程。
特殊注意：資料清洗邏輯必須冪等（重複執行不應產生重複筆數）；
大型資料集（.jsonl .parquet .csv）不應進 git。
"""

def main():
    diff = get_pr_diff()
    if not diff or len(diff) < 20:
        print("[INFO] Diff too small, skipping review.")
        return

    focus_area = "資料抓取與清洗流程"

    if build_review_prompt is not None:
        prompt = build_review_prompt(diff, focus_area, PROJECT_CONTEXT.strip())
    else:
        prompt = f"""你是一位資深 Python 工程師，請審核以下 PR diff。
領域：【{focus_area}】
請檢查安全性、錯誤處理、程式碼品質，使用繁體中文條列說明。
<diff>{diff}</diff>"""

    content = _call_ai(prompt)
    if content:
        post_comment(content)
    else:
        print("[ERROR] AI review failed, no comment posted.")

if __name__ == "__main__":
    main()
