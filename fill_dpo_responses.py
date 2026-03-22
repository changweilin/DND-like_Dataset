"""
fill_dpo_responses.py — 批次填入 DPO chosen 欄位的 GM 回應

對 lora_storyteller_dpo.jsonl 中所有 chosen.gpt == '<NEEDS_GM_RESPONSE>' 的記錄，
使用本地 Ollama 模型以 GM 角色生成回應並原地覆寫。

支援中斷續跑：只處理尚未填入的記錄。

Usage:
    python fill_dpo_responses.py                        # 使用預設模型 qwen2.5:7b
    python fill_dpo_responses.py --model deepseek-r1:14b
    python fill_dpo_responses.py --limit 50             # 測試：只處理 50 筆
    python fill_dpo_responses.py --dry-run              # 不呼叫 Ollama，只顯示統計
    python fill_dpo_responses.py --input PATH           # 指定輸入檔
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys
import time
import urllib.request
import urllib.error

# Windows cp950 終端機相容：強制 stdout 使用 UTF-8
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------

DEFAULT_INPUT = pathlib.Path(
    "data/finetune/sharegpt/cleaned/lora_storyteller_dpo.jsonl"
)
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"

GM_SYSTEM_PROMPT = (
    "你是一位資深的遊戲主持人 (Game Master)。"
    "請以充滿戲劇張力的敘事風格回應玩家的行動，"
    "強調場景氛圍、NPC 情感反應，以及選擇帶來的後果。"
    "描述應簡潔有力，突顯玩家決策的重量，避免百科式陳述。"
)

PLACEHOLDER = "<NEEDS_GM_RESPONSE>"

# ---------------------------------------------------------------------------
# Ollama 呼叫
# ---------------------------------------------------------------------------

def call_ollama(model: str, system: str, prompt: str, timeout: int = 120) -> str:
    """
    呼叫本地 Ollama /api/generate，回傳生成文字。
    失敗時 raise RuntimeError。
    """
    payload = json.dumps({
        "model":  model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "top_p": 0.9,
            "num_predict": 400,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "").strip()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama 連線失敗: {e}") from e


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: pathlib.Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] {path}:{lineno} 解析失敗: {e}")
    return records


def save_jsonl(records: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _get_chosen_gpt(record: dict) -> tuple[int, str]:
    """回傳 chosen 列表中 gpt 項的 (index, value)，找不到則 (-1, '')。"""
    for i, turn in enumerate(record.get("chosen", [])):
        if turn.get("from") == "gpt":
            return i, turn.get("value", "")
    return -1, ""


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="批次填入 DPO chosen.gpt 的 GM 回應（使用本地 Ollama）"
    )
    parser.add_argument("--input",  type=pathlib.Path, default=DEFAULT_INPUT,
                        help=f"DPO JSONL 檔案路徑（預設: {DEFAULT_INPUT}）")
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help=f"Ollama 模型名稱（預設: {DEFAULT_MODEL}）")
    parser.add_argument("--limit",  type=int, default=None,
                        help="只處理前 N 筆待生成記錄（測試用）")
    parser.add_argument("--dry-run", action="store_true",
                        help="不呼叫 Ollama，只顯示統計")
    parser.add_argument("--delay",  type=float, default=0.5,
                        help="每次呼叫後的等待秒數（預設: 0.5）")
    args = parser.parse_args()

    input_path: pathlib.Path = args.input
    if not input_path.is_absolute():
        input_path = pathlib.Path(__file__).parent / input_path

    if not input_path.exists():
        print(f"[ERROR] 找不到輸入檔: {input_path}")
        sys.exit(1)

    records = load_jsonl(input_path)
    total = len(records)

    # 找出需要填入的記錄索引
    pending_indices = [
        i for i, rec in enumerate(records)
        if _get_chosen_gpt(rec)[1] == PLACEHOLDER
    ]
    already_done = total - len(pending_indices)

    print(f"{'=' * 60}")
    print(f"fill_dpo_responses.py")
    print(f"{'=' * 60}")
    print(f"  輸入檔  : {input_path}")
    print(f"  模型    : {args.model}")
    print(f"  總記錄  : {total}")
    print(f"  已完成  : {already_done}")
    print(f"  待生成  : {len(pending_indices)}")
    if args.limit:
        print(f"  本次上限: {args.limit}")
    print(f"{'=' * 60}")

    if not pending_indices:
        print("✓ 所有記錄已完成，無需生成。")
        return

    if args.dry_run:
        print("[dry-run] 不執行生成。")
        return

    # 測試 Ollama 連線
    print("正在測試 Ollama 連線...", end=" ", flush=True)
    try:
        call_ollama(args.model, "test", "hi", timeout=15)
        print("OK")
    except RuntimeError as e:
        print(f"FAILED\n[ERROR] {e}")
        print("請確認 Ollama 已啟動：ollama serve")
        sys.exit(1)

    to_process = pending_indices[:args.limit] if args.limit else pending_indices
    success = 0
    failed  = 0

    for seq, rec_idx in enumerate(to_process, 1):
        rec = records[rec_idx]
        prompt_text: str = rec.get("prompt", "")
        gpt_idx, _ = _get_chosen_gpt(rec)

        print(f"[{seq}/{len(to_process)}] idx={rec_idx}  prompt={prompt_text[:60]!r}...")

        try:
            response = call_ollama(args.model, GM_SYSTEM_PROMPT, prompt_text)
        except RuntimeError as e:
            print(f"  FAIL generate: {e}")
            failed += 1
            continue

        if not response:
            print("  FAIL empty response, skip.")
            failed += 1
            continue

        # 原地更新 chosen[gpt_idx]
        new_chosen = list(rec["chosen"])
        new_chosen[gpt_idx] = {"from": "gpt", "value": response}
        records[rec_idx] = dict(rec)
        records[rec_idx]["chosen"] = new_chosen

        print(f"  OK {len(response)} chars")
        success += 1

        # 每 10 筆存一次（防止中斷遺失）
        if success % 10 == 0:
            save_jsonl(records, input_path)
            print(f"  [checkpoint] saved {success}")

        if args.delay > 0:
            time.sleep(args.delay)

    # 最終儲存
    save_jsonl(records, input_path)

    print(f"\n{'=' * 60}")
    remaining = sum(
        1 for rec in records if _get_chosen_gpt(rec)[1] == PLACEHOLDER
    )
    print(f"完成：成功={success}  失敗={failed}  剩餘待生成={remaining}")
    print(f"輸出：{input_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
