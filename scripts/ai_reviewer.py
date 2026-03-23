"""
ai_reviewer.py — Gemini Code Reviewer (Dataset 專屬版)
"""
import os, sys, json, subprocess, requests
import google.generativeai as genai

API_KEY, GITHUB_TOKEN = os.getenv("GEMINI_API_KEY"), os.getenv("GITHUB_TOKEN")
REPO, EVENT_PATH, BASE_REF = os.getenv("GITHUB_REPOSITORY"), os.getenv("GITHUB_EVENT_PATH"), os.getenv("GITHUB_BASE_REF", "main")
if not API_KEY or not GITHUB_TOKEN: sys.exit(0)

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def get_pr_diff():
    try: return subprocess.check_output(["git", "diff", f"origin/{BASE_REF}...HEAD"]).decode("utf-8")
    except: return ""

def post_comment(comment):
    with open(EVENT_PATH, 'r') as f: pr_number = json.load(f)['number']
    requests.post(f"https://api.github.com/repos/{REPO}/issues/{pr_number}/comments",
                  headers={"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"},
                  json={"body": f"### 🤖 Antigravity AI Review (Dataset)\n\n{comment}\n\n---\n*本評論由 Gemini 2.0 Flash 產生*"})

def main():
    diff = get_pr_diff()
    if not diff or len(diff) < 20: return
    prompt = f"你是一位資深的資料科學家與爬蟲專家。請審核以下 Diff 並針對資料抓取效率、正規化邏輯 (ShareGPT 格式)、資料品質監控與反爬蟲策略提供改進建議：\n\n{diff}"
    try:
        response = model.generate_content(prompt)
        if response.text: post_comment(response.text)
    except Exception as e: print(e)

if __name__ == "__main__": main()
