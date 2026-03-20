"""
postprocess_rl.py — RL 訓練數據後處理腳本

依 dataset_format_report.md 建議，對 ShareGPT 數據集進行清洗、過濾、格式轉換，
為 GRPO/DPO 訓練做準備。

輸出位置：data/finetune/sharegpt/cleaned/

Usage:
    python postprocess_rl.py                        # 處理全部
    python postprocess_rl.py --task analyst          # 只處理 analyst
    python postprocess_rl.py --task translator       # 只處理 translator
    python postprocess_rl.py --task storyteller      # DPO 偏好對準備
    python postprocess_rl.py --task reasoning        # JSON key 標準化
    python postprocess_rl.py --stats                 # 僅報告，不寫入
    python postprocess_rl.py --report report.md      # 輸出 markdown 報告
    python postprocess_rl.py --max-tokens 900        # 自訂 translator token 上限
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import re
import sys
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("postprocess_rl")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_DIR = pathlib.Path("data/finetune/sharegpt")
OUTPUT_DIR = pathlib.Path("data/finetune/sharegpt/cleaned")

INPUT_FILES = {
    "analyst":     INPUT_DIR / "lora_analyst.jsonl",
    "translator":  INPUT_DIR / "lora_translator.jsonl",
    "storyteller": INPUT_DIR / "lora_storyteller.jsonl",
    "reasoning":   INPUT_DIR / "lora_reasoning.jsonl",
}

OUTPUT_FILES = {
    "analyst":     OUTPUT_DIR / "lora_analyst_cleaned.jsonl",
    "translator":  OUTPUT_DIR / "lora_translator_cleaned.jsonl",
    "storyteller": OUTPUT_DIR / "lora_storyteller_dpo.jsonl",
    "reasoning":   OUTPUT_DIR / "lora_reasoning_cleaned.jsonl",
}

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_TIKTOKEN_AVAILABLE: Optional[bool] = None
_TIKTOKEN_ENC = None


def _init_tiktoken() -> bool:
    global _TIKTOKEN_AVAILABLE, _TIKTOKEN_ENC
    if _TIKTOKEN_AVAILABLE is not None:
        return _TIKTOKEN_AVAILABLE
    try:
        import tiktoken
        _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        _TIKTOKEN_AVAILABLE = True
    except ImportError:
        log.warning("tiktoken not installed — using character-based fallback (len/3.5)")
        _TIKTOKEN_AVAILABLE = False
    return _TIKTOKEN_AVAILABLE


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base, or len/3.5 fallback."""
    if _init_tiktoken() and _TIKTOKEN_ENC is not None:
        return len(_TIKTOKEN_ENC.encode(text))
    return int(len(text) / 3.5)


def _record_text(record: dict) -> str:
    """Concatenate all conversation turn values into one string."""
    parts: list[str] = []
    for turn in record.get("conversations", []):
        val = turn.get("value", "")
        if val:
            parts.append(val)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        log.error(f"File not found: {path}")
        return []
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning(f"{path}:{lineno}: JSON parse error — {exc}")
    return records


def save_jsonl(records: list[dict], path: pathlib.Path, dry_run: bool = False) -> None:
    if dry_run:
        log.info(f"[dry-run] Would write {len(records)} records to {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(records)} records → {path}")


# ---------------------------------------------------------------------------
# Step 1: Analyst — NER entity cleaning
# ---------------------------------------------------------------------------

# Blacklisted entity strings (case-insensitive exact or substring match)
ENTITY_BLACKLIST: set[str] = {
    # Generic nouns
    "age", "era", "campaign", "chapter", "arc", "season", "city", "town",
    "world", "land", "region", "area", "place", "point", "empire", "kingdom",
    "realm", "guild", "order", "court", "council", "temple", "cult",
    "north", "south", "east", "west", "central",
    # Show / meta names
    "critical role", "crit recap", "exandria unlimited",
    # Single letters & trivial
    "a", "an", "the",
}

# Adjective suffixes that indicate a demonym/adjective form, not a proper name
_ADJECTIVE_SUFFIX_RE = re.compile(r"(an|ean|ian)$", re.IGNORECASE)

# Numeric or "Campaign N" patterns
_CAMPAIGN_NUM_RE = re.compile(r"^campaign\s+\w+$", re.IGNORECASE)
_PURE_NUMBER_RE = re.compile(r"^\d+$")


def _is_valid_entity(entity: str) -> bool:
    """Return True if the entity string should be kept."""
    e = entity.strip()

    # Remove entities containing newlines
    if "\n" in e:
        return False

    # Length check
    if len(e) <= 2:
        return False

    e_lower = e.lower()

    # Exact blacklist match
    if e_lower in ENTITY_BLACKLIST:
        return False

    # Blacklist substring match (e.g. "Critical Role Campaign" contains "campaign")
    for bl in ENTITY_BLACKLIST:
        if " " in bl and bl in e_lower:
            return False

    # Pure numeric
    if _PURE_NUMBER_RE.match(e):
        return False

    # "Campaign + anything" pattern
    if _CAMPAIGN_NUM_RE.match(e):
        return False

    # Adjective/demonym form: single word ending in -an/-ean/-ian
    # (e.g. "Exandrian", "Voxilnian") — these are adjectives, not names
    parts = e.split()
    if len(parts) == 1 and _ADJECTIVE_SUFFIX_RE.search(e) and len(e) > 5:
        return False

    return True


def clean_analyst_ner_record(record: dict) -> dict | None:
    """
    Clean a single NER record's gpt output.
    Returns cleaned record, or None if the record should be discarded.
    Only NER records are cleaned; sentiment records pass through unchanged.
    """
    conversations = record.get("conversations", [])

    # Detect record type by system prompt
    system_val = ""
    gpt_idx = -1
    for i, turn in enumerate(conversations):
        if turn.get("from") == "system":
            system_val = turn.get("value", "")
        if turn.get("from") == "gpt":
            gpt_idx = i

    # Sentiment records: pass through
    if "情緒判讀" in system_val or "情緒基調" in system_val:
        return record

    # NER records: clean entity lists
    if gpt_idx == -1:
        return None

    gpt_val = conversations[gpt_idx].get("value", "")
    try:
        entities = json.loads(gpt_val)
    except (json.JSONDecodeError, ValueError):
        # Can't parse — discard
        return None

    if not isinstance(entities, dict):
        return None

    # Filter each entity list
    cleaned: dict[str, list[str]] = {}
    for key, lst in entities.items():
        if not isinstance(lst, list):
            cleaned[key] = lst
            continue
        cleaned[key] = [e for e in lst if isinstance(e, str) and _is_valid_entity(e)]

    # Discard if both 角色 and 組織 are empty after filtering
    roles = cleaned.get("角色", [])
    orgs = cleaned.get("組織", [])
    if not roles and not orgs:
        return None

    # Rebuild record with cleaned gpt output
    new_record = dict(record)
    new_conversations = list(conversations)
    new_conversations[gpt_idx] = dict(conversations[gpt_idx])
    new_conversations[gpt_idx]["value"] = json.dumps(cleaned, ensure_ascii=False)
    new_record["conversations"] = new_conversations
    return new_record


def postprocess_analyst(records: list[dict]) -> tuple[list[dict], dict]:
    """
    Clean analyst records.
    Returns (cleaned_records, stats).
    """
    cleaned: list[dict] = []
    stats = {
        "input": len(records),
        "sentiment_passed": 0,
        "ner_kept": 0,
        "ner_discarded": 0,
        "entities_removed": 0,
    }

    for rec in records:
        conversations = rec.get("conversations", [])
        system_val = next(
            (t.get("value", "") for t in conversations if t.get("from") == "system"), ""
        )
        is_sentiment = "情緒判讀" in system_val or "情緒基調" in system_val

        result = clean_analyst_ner_record(rec)
        if result is None:
            stats["ner_discarded"] += 1
            continue

        if is_sentiment:
            stats["sentiment_passed"] += 1
        else:
            stats["ner_kept"] += 1
            # Count removed entities
            orig_gpt = next(
                (t.get("value", "") for t in conversations if t.get("from") == "gpt"), ""
            )
            new_gpt = next(
                (t.get("value", "") for t in result.get("conversations", [])
                 if t.get("from") == "gpt"), ""
            )
            try:
                orig_ents = json.loads(orig_gpt)
                new_ents = json.loads(new_gpt)
                orig_count = sum(len(v) for v in orig_ents.values() if isinstance(v, list))
                new_count = sum(len(v) for v in new_ents.values() if isinstance(v, list))
                stats["entities_removed"] += orig_count - new_count
            except Exception:
                pass

        cleaned.append(result)

    stats["output"] = len(cleaned)
    return cleaned, stats


# ---------------------------------------------------------------------------
# Step 2: Translator — long record filtering
# ---------------------------------------------------------------------------

def postprocess_translator(
    records: list[dict], max_tokens: int = 900
) -> tuple[list[dict], dict]:
    """
    Filter translator records exceeding max_tokens.
    Returns (filtered_records, stats).
    """
    kept: list[dict] = []
    discarded = 0
    token_counts: list[int] = []

    for rec in records:
        text = _record_text(rec)
        n = count_tokens(text)
        token_counts.append(n)
        if n <= max_tokens:
            kept.append(rec)
        else:
            discarded += 1

    stats = {
        "input": len(records),
        "output": len(kept),
        "discarded": discarded,
        "max_tokens_limit": max_tokens,
        "token_method": "tiktoken" if _TIKTOKEN_AVAILABLE else "char/3.5",
    }
    if token_counts:
        token_counts.sort()
        n = len(token_counts)
        stats["p50_tokens"] = token_counts[n // 2]
        stats["p95_tokens"] = token_counts[int(n * 0.95)]
        stats["max_tokens_seen"] = token_counts[-1]

    return kept, stats


# ---------------------------------------------------------------------------
# Step 3: Storyteller — DPO preference pair preparation
# ---------------------------------------------------------------------------

_GM_SYSTEM_PROMPT = (
    "你是一位資深的遊戲主持人 (Game Master)。"
    "請以充滿戲劇張力的敘事風格回應玩家的行動，"
    "強調場景氛圍、NPC 情感反應，以及選擇帶來的後果。"
    "描述應簡潔有力，突顯玩家決策的重量，避免百科式陳述。"
)


def convert_storyteller_to_dpo(records: list[dict]) -> tuple[list[dict], dict]:
    """
    Convert storyteller ShareGPT records to DPO preference pair format.
    chosen.gpt is a placeholder <NEEDS_GM_RESPONSE> for later LLM generation.
    Returns (dpo_records, stats).
    """
    dpo_records: list[dict] = []
    skipped = 0

    for rec in records:
        conversations = rec.get("conversations", [])

        # Extract turns
        system_turn = next(
            (t for t in conversations if t.get("from") == "system"), None
        )
        human_turn = next(
            (t for t in conversations if t.get("from") == "human"), None
        )
        gpt_turn = next(
            (t for t in conversations if t.get("from") == "gpt"), None
        )

        if not human_turn or not gpt_turn:
            skipped += 1
            continue

        orig_system_val = system_turn.get("value", "") if system_turn else ""
        human_val = human_turn.get("value", "")
        orig_gpt_val = gpt_turn.get("value", "")

        dpo_rec = {
            "prompt": human_val,
            "chosen": [
                {"from": "system", "value": _GM_SYSTEM_PROMPT},
                {"from": "gpt",    "value": "<NEEDS_GM_RESPONSE>"},
            ],
            "rejected": [
                {"from": "system", "value": orig_system_val},
                {"from": "gpt",    "value": orig_gpt_val},
            ],
        }
        dpo_records.append(dpo_rec)

    stats = {
        "input": len(records),
        "output": len(dpo_records),
        "skipped": skipped,
        "note": "chosen.gpt = <NEEDS_GM_RESPONSE>; requires LLM generation + human review",
    }
    return dpo_records, stats


# ---------------------------------------------------------------------------
# Step 4: Reasoning — JSON key standardization
# ---------------------------------------------------------------------------

REASONING_KEY_CANONICAL: dict[str, str] = {
    "好感度變化":  "好感度增量",
    "NPC特殊回應": "NPC回應",
    "NPC態度":    "NPC回應",
}


def _standardize_keys_in_obj(obj: object) -> object:
    """Recursively rename keys in a parsed JSON object."""
    if isinstance(obj, dict):
        result: dict = {}
        for k, v in obj.items():
            new_k = REASONING_KEY_CANONICAL.get(k, k)
            result[new_k] = _standardize_keys_in_obj(v)
        return result
    if isinstance(obj, list):
        return [_standardize_keys_in_obj(item) for item in obj]
    return obj


def standardize_reasoning_keys(record: dict) -> dict:
    """
    Standardize JSON keys in the gpt output of a reasoning record.
    Returns the (possibly mutated) record.
    """
    conversations = record.get("conversations", [])
    for i, turn in enumerate(conversations):
        if turn.get("from") != "gpt":
            continue
        val = turn.get("value", "")
        try:
            parsed = json.loads(val)
        except (json.JSONDecodeError, ValueError):
            continue  # Not JSON — leave as-is

        new_parsed = _standardize_keys_in_obj(parsed)
        if new_parsed != parsed:
            new_conversations = list(conversations)
            new_conversations[i] = dict(turn)
            new_conversations[i]["value"] = json.dumps(new_parsed, ensure_ascii=False)
            record = dict(record)
            record["conversations"] = new_conversations
        break  # Only process first gpt turn

    return record


def postprocess_reasoning(records: list[dict]) -> tuple[list[dict], dict]:
    """
    Standardize JSON keys in reasoning records.
    Returns (standardized_records, stats).
    """
    result: list[dict] = []
    keys_renamed = 0

    for rec in records:
        new_rec = standardize_reasoning_keys(rec)
        # Detect if any change was made by comparing gpt values
        orig_gpt = next(
            (t.get("value", "") for t in rec.get("conversations", [])
             if t.get("from") == "gpt"), ""
        )
        new_gpt = next(
            (t.get("value", "") for t in new_rec.get("conversations", [])
             if t.get("from") == "gpt"), ""
        )
        if orig_gpt != new_gpt:
            keys_renamed += 1
        result.append(new_rec)

    stats = {
        "input": len(records),
        "output": len(result),
        "records_with_key_changes": keys_renamed,
        "canonical_map": REASONING_KEY_CANONICAL,
    }
    return result, stats


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt_stats_block(task: str, stats: dict) -> str:
    lines = [f"### {task}"]
    for k, v in stats.items():
        if isinstance(v, dict):
            lines.append(f"- **{k}**:")
            for kk, vv in v.items():
                lines.append(f"  - `{kk}` → `{vv}`")
        else:
            lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


def generate_report(all_stats: dict[str, dict], report_path: pathlib.Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections = [
        f"# RL 數據後處理報告",
        f"",
        f"> 生成時間：{now}",
        f"> 腳本：postprocess_rl.py",
        f"",
        "---",
        "",
        "## 處理統計",
        "",
    ]
    for task, stats in all_stats.items():
        sections.append(_fmt_stats_block(task, stats))
        sections.append("")

    content = "\n".join(sections)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    log.info(f"Report written → {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL 訓練數據後處理：清洗、過濾、格式轉換"
    )
    parser.add_argument(
        "--task",
        choices=["analyst", "translator", "storyteller", "reasoning"],
        help="只處理指定任務（預設：全部）",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="僅報告統計，不寫入任何檔案",
    )
    parser.add_argument(
        "--report", type=pathlib.Path, metavar="FILE",
        help="輸出 markdown 處理報告到指定路徑",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=900, metavar="N",
        help="Translator 記錄的 token 上限（預設：900）",
    )
    args = parser.parse_args()

    dry_run: bool = args.stats
    tasks = [args.task] if args.task else ["analyst", "translator", "storyteller", "reasoning"]

    all_stats: dict[str, dict] = {}

    for task in tasks:
        input_path = INPUT_FILES[task]
        output_path = OUTPUT_FILES[task]

        log.info(f"{'='*60}")
        log.info(f"Task: {task} | Input: {input_path}")

        records = load_jsonl(input_path)
        if not records:
            log.warning(f"No records loaded for {task} — skipping.")
            all_stats[task] = {"input": 0, "output": 0, "error": "file empty or missing"}
            continue

        if task == "analyst":
            result, stats = postprocess_analyst(records)
        elif task == "translator":
            result, stats = postprocess_translator(records, max_tokens=args.max_tokens)
        elif task == "storyteller":
            result, stats = convert_storyteller_to_dpo(records)
        elif task == "reasoning":
            result, stats = postprocess_reasoning(records)
        else:
            continue

        all_stats[task] = stats
        _log_stats(task, stats)

        if not dry_run:
            save_jsonl(result, output_path)

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY")
    for task, stats in all_stats.items():
        inp = stats.get("input", "?")
        out = stats.get("output", "?")
        log.info(f"  {task:<14} {inp:>5} → {out:>5}")
    log.info("=" * 60)

    # Optional markdown report
    if args.report:
        report_path = args.report
        generate_report(all_stats, report_path)
        if not dry_run:
            pass  # already written inside generate_report
    elif not dry_run:
        # Default report in output dir
        default_report = OUTPUT_DIR / "postprocess_report.md"
        generate_report(all_stats, default_report)


def _log_stats(task: str, stats: dict) -> None:
    log.info(f"  Stats for [{task}]:")
    for k, v in stats.items():
        if isinstance(v, dict):
            log.info(f"    {k}:")
            for kk, vv in v.items():
                log.info(f"      {kk}: {vv}")
        else:
            log.info(f"    {k}: {v}")


if __name__ == "__main__":
    main()
