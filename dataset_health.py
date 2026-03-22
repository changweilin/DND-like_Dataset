"""
dataset_health.py — 資料集健康報告產生器（含自動修復）

掃描全部 JSONL 資料集（Alpaca 格式 + ShareGPT 格式），
計算品質指標，與上次快照比較，輸出 Markdown 健康報告。
使用 --fix 可自動修復偵測到的問題。

使用方式：
    python dataset_health.py              # 掃描全部，輸出報告
    python dataset_health.py --fix        # 掃描 + 自動修復
    python dataset_health.py --dry-fix    # 預覽修復步驟（不執行）
    python dataset_health.py --text       # 只輸出純文字（不存檔）
    python dataset_health.py --flag-bad   # 列出高風險來源詳情
    python dataset_health.py --history    # 列出歷史健康快照

修復鏈：
    高重複率 → build_dataset/convert_to_sharegpt --fresh 重建
    空檔     → postprocess_rl（若 source 空 → 往上找 convert → 若無 raw data → scraper）

輸出：
    data/health_reports/YYYY-MM-DD_HHmmss.md   Markdown 報告
    data/health_reports/latest.json            最新 JSON 快照（供比較）
    data/health_reports/history.jsonl          歷史快照索引
"""

import argparse
import hashlib
import io
import json
import pathlib
import re
import subprocess
import sys
import time
from datetime import datetime

# Windows cp950 終端機相容：強制 stdout 使用 UTF-8
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 路徑設定 ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = pathlib.Path(__file__).parent.resolve()
DATA_DIR     = SCRIPT_DIR / "data"
REPORTS_DIR  = DATA_DIR / "health_reports"
LATEST_JSON  = REPORTS_DIR / "latest.json"
HISTORY_FILE = REPORTS_DIR / "history.jsonl"

# 掃描的資料夾（Alpaca + ShareGPT）
SCAN_DIRS = [
    DATA_DIR / "finetune",
    DATA_DIR / "finetune" / "sharegpt",
    DATA_DIR / "finetune" / "sharegpt" / "cleaned",
]

# 品質閾值（超過 → 警告 / 危險）
THRESHOLDS = {
    "bad_rate_warn":   0.03,    # bad samples > 3% → 警告
    "bad_rate_danger": 0.10,    # bad samples > 10% → 危險
    "dup_rate_warn":   0.05,    # exact duplicates > 5% → 警告
    "dup_rate_danger": 0.20,    # exact duplicates > 20% → 危險
    "min_records":     50,      # 少於 50 筆 → 警告
    "min_word_median": 20,      # 中位字數 < 20 → 警告
}

# 逐檔閾值覆寫（結構化/工具呼叫類資料集字數天生較短）
FILE_THRESHOLD_OVERRIDES: dict[str, dict] = {
    "lora_analyst.jsonl":         {"min_word_median": 3},
    "lora_analyst_cleaned.jsonl": {"min_word_median": 3},
}

# ── 修復地圖 ─────────────────────────────────────────────────────────────────────
#
# 格式：
#   "file_name.jsonl": {
#       "high_dup" | "empty": {
#           "label":       顯示訊息,
#           "cmds":        [[腳本, 參數, ...], ...],   # 按順序執行
#           "raw_dirs":    ["data/raw/xxx"],            # 空時先確認 raw data
#           "scrape_cmd":  ["scraper.py", "--category", "xxx"],  # 若 raw data 不存在
#           "requires":    "source_file.jsonl",         # 若 source 也空，先補建
#       }
#   }
REMEDIATION_MAP: dict[str, dict] = {
    # ── Alpaca 格式（build_dataset.py 產生）──────────────────────────────────────
    "literature_dataset.jsonl": {
        "high_dup": {
            "label": "重建 literature 資料集（--fresh 去重）",
            "cmds":  [["build_dataset.py", "--fresh", "--dataset", "literature"]],
        },
        "empty": {
            "label": "從頭建置 literature 資料集",
            "cmds":  [["build_dataset.py", "--fresh", "--dataset", "literature"]],
            "raw_dirs":   ["data/raw/webnovel"],
            "scrape_cmd": ["scraper.py", "--category", "webnovel"],
        },
    },
    "rpg_dataset.jsonl": {
        "high_dup": {
            "label": "重建 rpg 資料集（--fresh 去重）",
            "cmds":  [["build_dataset.py", "--fresh", "--dataset", "rpg"]],
        },
        "empty": {
            "label": "從頭建置 rpg 資料集",
            "cmds":  [["build_dataset.py", "--fresh", "--dataset", "rpg"]],
            "raw_dirs":   ["data/raw/trpg", "data/raw/extra_lore"],
            "scrape_cmd": ["scraper.py", "--category", "trpg"],
        },
    },
    # ── ShareGPT 格式（convert_to_sharegpt.py 產生）──────────────────────────────
    "lora_analyst.jsonl": {
        "high_dup": {
            "label": "重建 analyst ShareGPT 資料集（--fresh 去重）+ 原地去重",
            "cmds":  [["convert_to_sharegpt.py", "--fresh", "--task", "analyst"]],
            "dedup_after": "data/finetune/sharegpt/lora_analyst.jsonl",
        },
        "empty": {
            "label": "建置 analyst ShareGPT 資料集",
            "cmds":  [["convert_to_sharegpt.py", "--fresh", "--task", "analyst"]],
            "raw_dirs":   ["data/raw/trpg", "data/raw/webnovel"],
            "scrape_cmd": ["scraper.py", "--category", "trpg"],
            "dedup_after": "data/finetune/sharegpt/lora_analyst.jsonl",
        },
    },
    "lora_storyteller.jsonl": {
        "high_dup": {
            "label": "重建 storyteller ShareGPT 資料集（--fresh 去重）",
            "cmds":  [["convert_to_sharegpt.py", "--fresh", "--task", "storyteller"]],
        },
        "empty": {
            "label": "建置 storyteller ShareGPT 資料集",
            "cmds":  [["convert_to_sharegpt.py", "--fresh", "--task", "storyteller"]],
            "raw_dirs":   ["data/raw/trpg", "data/raw/webnovel"],
            "scrape_cmd": ["scraper.py", "--category", "trpg"],
        },
    },
    "lora_translator.jsonl": {
        "high_dup": {
            "label": "重建 translator ShareGPT 資料集（--fresh 去重）",
            "cmds":  [["convert_to_sharegpt.py", "--fresh", "--task", "translator"]],
        },
        "empty": {
            "label": "建置 translator ShareGPT 資料集",
            "cmds":  [["convert_to_sharegpt.py", "--fresh", "--task", "translator"]],
            "raw_dirs":   ["data/raw/multilingual_lore"],
            "scrape_cmd": ["scraper.py", "--category", "multilingual_lore"],
        },
    },
    # ── Cleaned（postprocess_rl.py 產生）─────────────────────────────────────────
    "lora_analyst_cleaned.jsonl": {
        "high_dup": {
            "label": "重新後處理 analyst（覆蓋去重）+ 原地去重",
            "cmds":  [["postprocess_rl.py", "--task", "analyst"]],
            "dedup_after": "data/finetune/sharegpt/cleaned/lora_analyst_cleaned.jsonl",
        },
        "empty": {
            "label": "後處理 analyst",
            "cmds":  [["postprocess_rl.py", "--task", "analyst"]],
            "requires": "lora_analyst.jsonl",
            "dedup_after": "data/finetune/sharegpt/cleaned/lora_analyst_cleaned.jsonl",
        },
    },
    "lora_reasoning_cleaned.jsonl": {
        "high_dup": {
            "label": "重新後處理 reasoning（覆蓋去重）",
            "cmds":  [["postprocess_rl.py", "--task", "reasoning"]],
        },
        "empty": {
            "label": "後處理 reasoning",
            "cmds":  [["postprocess_rl.py", "--task", "reasoning"]],
            "requires": "lora_reasoning.jsonl",
        },
    },
    "lora_storyteller_dpo.jsonl": {
        "empty": {
            "label": "後處理 storyteller → 生成 DPO 偏好對",
            "cmds":  [["postprocess_rl.py", "--task", "storyteller"]],
            "requires": "lora_storyteller.jsonl",
        },
    },
    "lora_translator_cleaned.jsonl": {
        "high_dup": {
            "label": "重新後處理 translator（覆蓋去重）",
            "cmds":  [["postprocess_rl.py", "--task", "translator"]],
        },
        "empty": {
            "label": "後處理 translator",
            "cmds":  [["postprocess_rl.py", "--task", "translator"]],
            "requires": "lora_translator.jsonl",
        },
    },
}

# 垃圾內容辨識模式（與 validate_dataset.py 一致）
_BAD_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"just a moment",
    r"enable javascript",
    r"checking your browser",
    r"ddos protection by cloudflare",
    r"ray id:",
    r"<html",
    r"window\.__",
    r"document\.cookie",
    r"function\s*\(",
    r"subscribe (now|today)",
    r"click here to",
    r"\bnewsletter\b.{0,30}\bsign up\b",
    r"all rights reserved",
]]


# ── 工具函數 ───────────────────────────────────────────────────────────────────────

def _is_bad(text: str) -> bool:
    return any(pat.search(text) for pat in _BAD_PATTERNS)


def _md5(text: str) -> str:
    return hashlib.md5(text.strip().encode()).hexdigest()


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
    return sorted_vals[idx]


def _detect_format(record: dict) -> str:
    """偵測單筆記錄的格式。"""
    if "conversations" in record:
        return "sharegpt"
    if "instruction" in record and "output" in record:
        return "alpaca"
    if "prompt" in record and "chosen" in record:
        return "dpo"
    return "unknown"


def _extract_text(record: dict, fmt: str) -> str:
    """從記錄中提取主要文本（用於品質檢查）。"""
    if fmt == "alpaca":
        return record.get("output", "")
    if fmt == "sharegpt":
        convos = record.get("conversations", [])
        parts = [c.get("value", "") for c in convos if c.get("from") == "gpt"]
        return " ".join(parts)
    if fmt == "dpo":
        # DPO: {"prompt": ..., "chosen": [{from, value}], "rejected": [{from, value}]}
        chosen = record.get("chosen", [])
        parts  = [c.get("value", "") for c in chosen if c.get("from") == "gpt"]
        return record.get("prompt", "") + " " + " ".join(parts)
    return ""


def _dpo_needs_generation(record: dict) -> bool:
    """DPO 記錄是否為 placeholder（chosen.gpt = <NEEDS_GM_RESPONSE>）。"""
    for turn in record.get("chosen", []):
        if turn.get("from") == "gpt" and "<NEEDS_GM_RESPONSE>" in turn.get("value", ""):
            return True
    return False


# ── 核心分析 ──────────────────────────────────────────────────────────────────────

def analyze_file(path: pathlib.Path) -> dict:
    """
    分析單個 JSONL 檔案，返回品質指標 dict。
    """
    stats: dict = {
        "file":            str(path.relative_to(SCRIPT_DIR)),
        "name":            path.name,
        "format":          "unknown",
        "total_lines":     0,
        "valid_records":   0,
        "format_errors":   0,
        "bad_samples":     0,
        "exact_dups":      0,
        "word_count_min":  0,
        "word_count_p25":  0,
        "word_count_median": 0,
        "word_count_p75":  0,
        "word_count_max":  0,
        "by_source":       {},
        "by_language":     {},
        "bad_sources":     {},   # {source_id: bad_count} 高風險來源
        "needs_generation": 0,  # DPO placeholder 筆數（<NEEDS_GM_RESPONSE>）
        "error":           None,
    }

    if not path.exists():
        stats["error"] = "file not found"
        return stats

    records: list[dict] = []
    format_errors = 0

    try:
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                stats["total_lines"] += 1
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    format_errors += 1
                    continue
                fmt = _detect_format(rec)
                if fmt == "unknown":
                    format_errors += 1
                    continue
                if stats["format"] == "unknown":
                    stats["format"] = fmt
                records.append(rec)
    except Exception as e:
        stats["error"] = str(e)
        return stats

    stats["format_errors"] = format_errors
    stats["valid_records"] = len(records)

    if not records:
        return stats

    # ── 字數統計 ───────────────────────────────────────────────────────────────
    word_counts: list[int] = []
    fmt = stats["format"]
    seen_hashes: set[str] = set()
    exact_dups = 0
    bad_samples = 0
    by_source: dict[str, int]   = {}
    by_language: dict[str, int] = {}
    bad_by_source: dict[str, int] = {}

    for rec in records:
        text = _extract_text(rec, fmt)
        words = len(text.split())
        word_counts.append(words)

        # 重複檢查
        h = _md5(text)
        if h in seen_hashes:
            exact_dups += 1
        else:
            seen_hashes.add(h)

        # 垃圾內容
        if _is_bad(text):
            bad_samples += 1
            if fmt == "alpaca":
                src = rec.get("metadata", {}).get("source_id", "unknown")
            else:
                src = rec.get("source_id", "unknown")
            bad_by_source[src] = bad_by_source.get(src, 0) + 1

        # 來源 / 語言分布
        if fmt == "alpaca":
            meta = rec.get("metadata", {})
            src  = meta.get("source_id", "unknown")
            lang = meta.get("language", "en")
        else:
            src  = rec.get("source_id", "unknown")
            lang = rec.get("language", "unknown")
        by_source[src]    = by_source.get(src, 0) + 1
        by_language[lang] = by_language.get(lang, 0) + 1

    word_counts.sort()
    stats["bad_samples"]       = bad_samples
    stats["exact_dups"]        = exact_dups
    # DPO placeholder 統計
    if stats["format"] == "dpo":
        stats["needs_generation"] = sum(1 for r in records if _dpo_needs_generation(r))
    stats["by_source"]         = dict(sorted(by_source.items(), key=lambda x: -x[1]))
    stats["by_language"]       = dict(sorted(by_language.items()))
    stats["bad_sources"]       = dict(sorted(bad_by_source.items(), key=lambda x: -x[1]))
    stats["word_count_min"]    = word_counts[0]
    stats["word_count_p25"]    = int(_percentile(word_counts, 25))
    stats["word_count_median"] = int(_percentile(word_counts, 50))
    stats["word_count_p75"]    = int(_percentile(word_counts, 75))
    stats["word_count_max"]    = word_counts[-1]

    return stats


def scan_all() -> list[dict]:
    """掃描所有 JSONL 檔案，返回分析結果列表。"""
    results: list[dict] = []
    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            continue
        for jsonl_path in sorted(scan_dir.glob("*.jsonl")):
            print(f"  分析：{jsonl_path.relative_to(SCRIPT_DIR)}", flush=True)
            stats = analyze_file(jsonl_path)
            results.append(stats)
    return results


# ── 健康評分 ──────────────────────────────────────────────────────────────────────

def _health_status(stats: dict) -> str:
    """返回 '✓ 健康' / '⚠ 警告' / '✗ 危險' / '⏳ 待生成' / '✗ 空檔'。"""
    n = stats.get("valid_records", 0)

    # DPO placeholder：有記錄但全部需要 LLM 生成
    if stats.get("format") == "dpo":
        needs_gen = stats.get("needs_generation", 0)
        if needs_gen > 0 and needs_gen == n:
            return "⏳ 待生成"
        if n == 0:
            return "✗ 空檔"

    if n == 0:
        return "✗ 空檔"

    bad_rate = stats["bad_samples"] / n
    dup_rate = stats["exact_dups"] / n
    median_w = stats.get("word_count_median", 0)

    # 套用逐檔閾值覆寫
    overrides = FILE_THRESHOLD_OVERRIDES.get(stats.get("name", ""), {})
    min_word_median = overrides.get("min_word_median", THRESHOLDS["min_word_median"])
    min_records     = overrides.get("min_records",     THRESHOLDS["min_records"])

    danger = (
        bad_rate > THRESHOLDS["bad_rate_danger"] or
        dup_rate > THRESHOLDS["dup_rate_danger"]
    )
    warn = (
        bad_rate > THRESHOLDS["bad_rate_warn"] or
        dup_rate > THRESHOLDS["dup_rate_warn"] or
        n        < min_records or
        median_w < min_word_median
    )

    if danger:
        return "✗ 危險"
    if warn:
        return "⚠ 警告"
    return "✓ 健康"


def _delta_str(new_val, old_val, lower_is_better: bool = True) -> str:
    """產生趨勢字串，例如 '↓0.5%' 或 '↑2'。"""
    if old_val is None or new_val is None:
        return ""
    diff = new_val - old_val
    if diff == 0:
        return "→"
    arrow = "↓" if diff < 0 else "↑"
    improved = (diff < 0) == lower_is_better
    sign = "✓" if improved else "！"
    if isinstance(diff, float):
        return f"{sign}{arrow}{abs(diff)*100:.1f}%"
    return f"{sign}{arrow}{abs(diff)}"


# ── Markdown 報告產生 ────────────────────────────────────────────────────────────

def generate_markdown(results: list[dict], prev_results: list[dict], flag_bad: bool) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    # 建立 prev 查詢表
    prev_map = {r["name"]: r for r in prev_results} if prev_results else {}

    # ── 標題 ──
    lines.append(f"# 資料集健康報告")
    lines.append(f"")
    lines.append(f"**產生時間**：{ts}")
    lines.append(f"**掃描目錄**：{', '.join(str(d) for d in SCAN_DIRS)}")
    lines.append(f"")

    # ── 總覽 ──
    total_records   = sum(r.get("valid_records", 0) for r in results)
    total_bad       = sum(r.get("bad_samples", 0) for r in results)
    total_dups      = sum(r.get("exact_dups", 0) for r in results)
    healthy_count   = sum(1 for r in results if _health_status(r) == "✓ 健康")
    warn_count      = sum(1 for r in results if _health_status(r) == "⚠ 警告")
    danger_count    = sum(1 for r in results if "✗" in _health_status(r))

    lines.append(f"## 總覽")
    lines.append(f"")
    lines.append(f"| 指標 | 數值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 掃描檔案數 | {len(results)} |")
    lines.append(f"| 總有效記錄 | {total_records:,} |")
    lines.append(f"| 總垃圾樣本 | {total_bad:,} ({total_bad/max(total_records,1)*100:.1f}%) |")
    lines.append(f"| 總重複記錄 | {total_dups:,} ({total_dups/max(total_records,1)*100:.1f}%) |")
    lines.append(f"| 健康 ✓ | {healthy_count} 個檔案 |")
    lines.append(f"| 警告 ⚠ | {warn_count} 個檔案 |")
    lines.append(f"| 危險 ✗ | {danger_count} 個檔案 |")
    lines.append(f"")

    # ── 逐檔詳情 ──
    lines.append(f"## 逐檔詳情")
    lines.append(f"")
    lines.append(f"| 狀態 | 檔案 | 格式 | 記錄數 | 垃圾% | 重複% | 字數中位 | 語言分布 |")
    lines.append(f"|------|------|------|--------|-------|-------|----------|----------|")

    for r in results:
        n        = r.get("valid_records", 0)
        bad_pct  = r["bad_samples"] / n * 100 if n else 0
        dup_pct  = r["exact_dups"]  / n * 100 if n else 0
        status   = _health_status(r)
        prev     = prev_map.get(r["name"])
        prev_n   = prev.get("valid_records") if prev else None

        n_str    = f"{n:,}"
        if prev_n is not None and n != prev_n:
            diff = n - prev_n
            n_str += f" ({'+'if diff>0 else ''}{diff})"

        lang_str = " ".join(f"{k}:{v}" for k, v in list(r["by_language"].items())[:3])
        lines.append(
            f"| {status} | `{r['name']}` | {r['format']} | {n_str} "
            f"| {bad_pct:.1f}% | {dup_pct:.1f}% | {r.get('word_count_median',0)} "
            f"| {lang_str} |"
        )

    lines.append(f"")

    # ── 高風險來源 ──
    if flag_bad:
        lines.append(f"## 高風險來源（垃圾樣本來源分布）")
        lines.append(f"")
        has_any = False
        for r in results:
            bad_srcs = r.get("bad_sources", {})
            if not bad_srcs:
                continue
            has_any = True
            lines.append(f"### `{r['name']}`")
            lines.append(f"| 來源 ID | 垃圾筆數 |")
            lines.append(f"|---------|----------|")
            for src, cnt in sorted(bad_srcs.items(), key=lambda x: -x[1]):
                lines.append(f"| {src} | {cnt} |")
            lines.append(f"")
        if not has_any:
            lines.append(f"_無高風險來源。_")
            lines.append(f"")

    # ── 改善建議 ──
    lines.append(f"## 改善建議")
    lines.append(f"")
    suggestions: list[str] = []

    for r in results:
        n = r.get("valid_records", 0)
        if n == 0:
            suggestions.append(f"- `{r['name']}`：空檔，請確認爬取是否成功。")
            continue
        bad_rate = r["bad_samples"] / n
        dup_rate = r["exact_dups"] / n

        if bad_rate > THRESHOLDS["bad_rate_danger"]:
            top_bad = list(r.get("bad_sources", {}).keys())[:3]
            suggestions.append(
                f"- `{r['name']}`：垃圾率 {bad_rate*100:.1f}% 超過 {THRESHOLDS['bad_rate_danger']*100:.0f}%，"
                f"建議排除來源：{', '.join(top_bad) or '（未知）'}。"
            )
        elif bad_rate > THRESHOLDS["bad_rate_warn"]:
            suggestions.append(f"- `{r['name']}`：垃圾率 {bad_rate*100:.1f}%，建議排查低品質來源。")

        if dup_rate > THRESHOLDS["dup_rate_warn"]:
            suggestions.append(f"- `{r['name']}`：重複率 {dup_rate*100:.1f}%，建議執行去重後重新建置。")

        if n < THRESHOLDS["min_records"]:
            suggestions.append(f"- `{r['name']}`：記錄數 {n} 筆偏少，建議補充更多語料。")

    if suggestions:
        for s in suggestions:
            lines.append(s)
    else:
        lines.append("_所有資料集品質符合標準，無需立即行動。_")

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"*報告由 `dataset_health.py` 自動產生*")

    return "\n".join(lines)


# ── 快照管理 ──────────────────────────────────────────────────────────────────────

def load_latest_snapshot() -> list[dict]:
    """讀取上次的 JSON 快照（用於比較趨勢）。"""
    if not LATEST_JSON.exists():
        return []
    with open(LATEST_JSON, encoding="utf-8") as f:
        return json.load(f)


def save_snapshot(results: list[dict]):
    """儲存本次快照為 latest.json 並附加至 history.jsonl。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(LATEST_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    record = {
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_records": sum(r.get("valid_records", 0) for r in results),
        "total_bad":    sum(r.get("bad_samples", 0) for r in results),
        "files":        [r["name"] for r in results],
    }
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def cmd_history():
    """列出歷史健康快照。"""
    if not HISTORY_FILE.exists():
        print("尚無歷史快照。")
        return
    records: list[dict] = []
    with open(HISTORY_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not records:
        print("尚無歷史快照。")
        return
    print(f"{'#':<5} {'timestamp':<22} {'total_records':<15} {'total_bad':<12} {'files'}")
    print("-" * 80)
    for i, rec in enumerate(records):
        files_str = ", ".join(rec.get("files", []))[:50]
        print(
            f"{i+1:<5} {rec.get('timestamp',''):<22} "
            f"{rec.get('total_records',0):<15,} {rec.get('total_bad',0):<12,} {files_str}"
        )


# ── CLI ─────────────────────────────────────────────────────────────────────────

# ── 自動修復 ───────────────────────────────────────────────────────────────────────

def _raw_data_exists(raw_dirs: list[str]) -> bool:
    """確認指定目錄下是否有原始資料檔案（.txt 或 .json）。"""
    for raw_dir in raw_dirs:
        d = SCRIPT_DIR / raw_dir
        if d.exists():
            if list(d.rglob("*.txt")) or list(d.rglob("*.json")):
                return True
    return False


def _source_has_data(source_name: str) -> bool:
    """確認 source JSONL 是否存在且有記錄。"""
    for scan_dir in SCAN_DIRS:
        path = scan_dir / source_name
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        return True
    return False


def dedup_jsonl_inplace(rel_path: str) -> tuple[int, int]:
    """
    原地去重 JSONL 檔案（依 gpt/output 欄位 hash）。
    返回 (原始筆數, 去重後筆數)。
    """
    path = SCRIPT_DIR / rel_path
    if not path.exists():
        return 0, 0

    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    seen: set[str] = set()
    unique: list[dict] = []
    fmt = _detect_format(records[0]) if records else "unknown"

    for rec in records:
        text = _extract_text(rec, fmt)
        h    = _md5(text)
        if h not in seen:
            seen.add(h)
            unique.append(rec)

    with path.open("w", encoding="utf-8") as f:
        for rec in unique:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(records), len(unique)


def plan_remediation(results: list[dict]) -> list[dict]:
    """
    根據健康報告規劃修復步驟。
    返回 list of plans：
        { file, issue, reason, steps: [{label, cmd?, dedup?}] }
    """
    plans: list[dict] = []

    for stats in results:
        name   = stats["name"]
        n      = stats.get("valid_records", 0)
        status = _health_status(stats)

        if name not in REMEDIATION_MAP:
            continue

        rule = REMEDIATION_MAP[name]

        # DPO 待生成：非錯誤，不修復
        if status == "⏳ 待生成":
            continue

        # 判斷問題類型
        if n == 0:
            issue = "empty"
        else:
            dup_rate = stats["exact_dups"] / n
            if dup_rate > THRESHOLDS["dup_rate_warn"]:
                issue = "high_dup"
            else:
                continue

        if issue not in rule:
            continue

        action   = rule[issue]
        steps: list[dict] = []

        # ── empty 專用：往上追溯依賴 ──
        if issue == "empty":
            requires = action.get("requires")
            if requires and not _source_has_data(requires):
                # source 也沒資料 → 先建 source
                src_rule = REMEDIATION_MAP.get(requires, {}).get("empty", {})
                if src_rule:
                    # 若 source 需要 raw data，先確認
                    src_raw = src_rule.get("raw_dirs", [])
                    if src_raw and not _raw_data_exists(src_raw):
                        scrape = src_rule.get("scrape_cmd")
                        if scrape:
                            steps.append({
                                "label": f"爬取原始資料（{requires} 缺少 raw data）",
                                "cmd":   scrape,
                            })
                    for cmd in src_rule.get("cmds", []):
                        steps.append({"label": f"建立 {requires}", "cmd": cmd})

            # 若自身也需要 raw data，確認
            raw_dirs = action.get("raw_dirs", [])
            if raw_dirs and not _raw_data_exists(raw_dirs):
                scrape = action.get("scrape_cmd")
                if scrape:
                    # 插到最前面（若 source 鏈已有 scrape，避免重複）
                    already = any(s["cmd"] == scrape for s in steps)
                    if not already:
                        steps.insert(0, {
                            "label": f"爬取原始資料（{name} 缺少 raw data）",
                            "cmd":   scrape,
                        })

        # ── 主要修復指令 ──
        for cmd in action["cmds"]:
            steps.append({"label": action["label"], "cmd": cmd})

        # ── 原地去重（若設定）──
        dedup_path = action.get("dedup_after")
        if dedup_path:
            steps.append({
                "label": f"原地去重：{dedup_path}",
                "dedup": dedup_path,
            })

        dup_rate_str = f"{stats['exact_dups']/n*100:.1f}%" if n else "—"
        plans.append({
            "file":   name,
            "issue":  issue,
            "reason": "空檔" if issue == "empty" else f"重複率={dup_rate_str}",
            "steps":  steps,
        })

    return plans


def execute_remediation(plans: list[dict], dry_run: bool = False) -> list[dict]:
    """
    執行修復計畫。返回每個計畫的執行結果列表。
    """
    fix_results: list[dict] = []

    for plan in plans:
        print(f"\n{'─' * 62}")
        print(f"  修復目標：{plan['file']}  （{plan['reason']}）")
        print(f"{'─' * 62}")

        plan_ok = True
        for step in plan["steps"]:
            print(f"\n  [步驟] {step['label']}")

            # ── 原地去重步驟（不是子程序）──
            if "dedup" in step:
                if dry_run:
                    print("  [DRY-RUN] 跳過原地去重")
                    continue
                before, after = dedup_jsonl_inplace(step["dedup"])
                removed = before - after
                print(f"  [OK] 去重完成：{before} → {after} 筆（移除 {removed} 筆重複）")
                continue

            cmd = [sys.executable] + step["cmd"]

            if dry_run:
                print(f"  CMD  : {' '.join(cmd)}")
                print("  [DRY-RUN] 跳過執行")
                continue

            print(f"  CMD  : {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
            if result.returncode != 0:
                print(f"  [FAIL] returncode={result.returncode}")
                plan_ok = False
                break
            print("  [OK]")

        fix_results.append({
            "file":   plan["file"],
            "issue":  plan["issue"],
            "status": "ok" if (plan_ok or dry_run) else "failed",
        })

    return fix_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="資料集健康報告產生器（含自動修復）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--text",     action="store_true", help="只輸出純文字，不存檔")
    parser.add_argument("--flag-bad", action="store_true", help="報告中列出高風險來源詳情")
    parser.add_argument("--history",  action="store_true", help="列出歷史健康快照後退出")
    parser.add_argument("--no-save",  action="store_true", help="不儲存快照（唯讀模式）")
    parser.add_argument("--fix",      action="store_true", help="自動修復偵測到的問題（高重複率 / 空檔）")
    parser.add_argument("--dry-fix",  action="store_true", help="預覽修復步驟，不實際執行")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.history:
        cmd_history()
        return

    print("=" * 60)
    print("  資料集健康報告產生器")
    print("=" * 60)
    print()

    # 掃描全部 JSONL
    print("[掃描] 分析資料集品質...")
    results = scan_all()

    if not results:
        print("[WARN] 未找到任何 JSONL 檔案。")
        print(f"  掃描目錄：{[str(d) for d in SCAN_DIRS]}")
        sys.exit(0)

    # 載入上次快照（用於趨勢比較）
    prev_results = load_latest_snapshot()

    # 產生 Markdown 報告
    print("\n[產生] 建立 Markdown 報告...")
    md_content = generate_markdown(results, prev_results, flag_bad=args.flag_bad)

    if args.text:
        print("\n" + md_content)
        return

    # 儲存報告
    if not args.no_save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts_str      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        report_path = REPORTS_DIR / f"{ts_str}.md"
        report_path.write_text(md_content, encoding="utf-8")
        print(f"\n[儲存] 報告已存至：{report_path}")

        # 更新快照
        save_snapshot(results)
        print(f"[儲存] 快照已更新：{LATEST_JSON}")

    # 在 terminal 也顯示摘要
    print()
    print("─" * 60)
    print("  健康摘要")
    print("─" * 60)
    for r in results:
        n       = r.get("valid_records", 0)
        status  = _health_status(r)
        bad_pct = r["bad_samples"] / n * 100 if n else 0
        dup_pct = r["exact_dups"]  / n * 100 if n else 0
        print(f"  {status:<10} {r['name']:<40} {n:>6,} 筆  垃圾={bad_pct:.1f}%  重複={dup_pct:.1f}%")

    # 整體評分
    total = sum(r.get("valid_records", 0) for r in results)
    danger_files = [r["name"] for r in results if "✗" in _health_status(r)]
    warn_files   = [r["name"] for r in results if "⚠" in _health_status(r)]
    print()
    if danger_files:
        print(f"  ⚠ 危險檔案：{', '.join(danger_files)}")
    if warn_files:
        print(f"  ⚠ 警告檔案：{', '.join(warn_files)}")
    if not danger_files and not warn_files:
        print(f"  ✓ 全部 {len(results)} 個檔案健康，共 {total:,} 筆記錄。")
    print("─" * 60)

    if not args.no_save:
        print(f"\n  使用 `python dataset_health.py --history` 查看歷史趨勢")
        print(f"  使用 `python dataset_health.py --flag-bad` 列出高風險來源")

    # ── 自動修復模式 ─────────────────────────────────────────────────────────────
    if args.fix or args.dry_fix:
        print(f"\n{'=' * 62}")
        mode_label = "[DRY-RUN] 修復計畫預覽" if args.dry_fix else "自動修復"
        print(f"  {mode_label}")
        print(f"{'=' * 62}")

        plans = plan_remediation(results)

        if not plans:
            print("\n  ✓ 無需修復：所有資料集已符合品質標準。")
        else:
            print(f"\n  偵測到 {len(plans)} 個需修復的問題：")
            for i, p in enumerate(plans, 1):
                print(f"\n  [{i}] {p['file']}  →  {p['reason']}")
                for s in p["steps"]:
                    print(f"      • {s['label']}")
                    if "cmd" in s:
                        print(f"        {' '.join([sys.executable] + s['cmd'])}")
                    elif "dedup" in s:
                        print(f"        [原地去重] {s['dedup']}")

            if not args.dry_fix:
                print(f"\n[執行] 開始修復...")
                fix_results = execute_remediation(plans, dry_run=False)

                # ── 修復後重新掃描 ──────────────────────────────────────────────
                print(f"\n{'=' * 62}")
                print(f"  修復完成，重新掃描驗證結果...")
                print(f"{'=' * 62}")
                new_results = scan_all()

                print()
                print("─" * 62)
                print("  修復後健康摘要（前 → 後）")
                print("─" * 62)

                old_map = {r["name"]: r for r in results}
                for r in new_results:
                    name = r["name"]
                    old  = old_map.get(name, {})
                    n_old = old.get("valid_records", 0)
                    n_new = r.get("valid_records", 0)
                    dup_old = old.get("exact_dups", 0) / n_old * 100 if n_old else 0
                    dup_new = r.get("exact_dups", 0)  / n_new * 100 if n_new else 0
                    status_new = _health_status(r)
                    changed = name in {p["file"] for p in plans}
                    marker  = "→" if changed else " "
                    print(
                        f"  {marker} {status_new:<10} {name:<38} "
                        f"{n_old:>6,}→{n_new:<6,} 筆  "
                        f"重複 {dup_old:.1f}%→{dup_new:.1f}%"
                    )

                # 更新快照
                if not args.no_save:
                    save_snapshot(new_results)
                    print(f"\n[儲存] 快照已更新（修復後）：{LATEST_JSON}")

                # 修復摘要
                ok_count   = sum(1 for r in fix_results if r["status"] == "ok")
                fail_count = sum(1 for r in fix_results if r["status"] == "failed")
                print(f"\n  修復結果：成功 {ok_count} 個，失敗 {fail_count} 個")


if __name__ == "__main__":
    main()
