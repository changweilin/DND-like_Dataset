"""
validate_dataset.py — Quality report for fine-tuning JSONL datasets.

Checks format correctness, content quality, source distribution, and flags
any chunks that look like scraper garbage (Cloudflare pages, nav debris, etc.).

Usage:
    python validate_dataset.py                     # report on both datasets
    python validate_dataset.py --dataset rpg       # only rpg_dataset.jsonl
    python validate_dataset.py --dataset literature
    python validate_dataset.py --sample 5          # print 5 random output samples
    python validate_dataset.py --format-check      # format validation only, no stats
    python validate_dataset.py --output-dir PATH   # custom JSONL directory
"""

import argparse
import json
import logging
import pathlib
import random
import re
import sys

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("validate_dataset")

# ---------------------------------------------------------------------------
# Bad-content patterns (scraper garbage indicators)
# ---------------------------------------------------------------------------

_BAD_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"just a moment",           # Cloudflare challenge page
    r"enable javascript",
    r"checking your browser",
    r"ddos protection by cloudflare",
    r"ray id:",                 # Cloudflare footer
    r"<html",                   # raw HTML leaked into output
    r"window\.__",              # JS variable leak
    r"document\.cookie",
    r"function\s*\(",           # JS code fragment
    r"subscribe (now|today)",
    r"click here to",
    r"\bnewsletter\b.{0,30}\bsign up\b",
    r"all rights reserved",
]]

REQUIRED_FIELDS = {"instruction", "output", "metadata"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def _count_bad(output: str) -> int:
    return sum(1 for pat in _BAD_PATTERNS if pat.search(output))


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def validate_file(
    path: pathlib.Path,
    sample_n: int = 0,
    format_only: bool = False,
) -> dict:
    """
    Validate one JSONL file. Returns a stats dict.
    Prints the report to stdout.
    """
    if not path.exists():
        log.warning(f"File not found: {path}")
        return {"file": str(path), "error": "not found"}

    records = []
    format_errors = 0
    missing_fields: list[int] = []

    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                format_errors += 1
                log.debug(f"  JSON error line {lineno}: {exc}")
                continue
            missing = REQUIRED_FIELDS - set(rec.keys())
            if missing:
                missing_fields.append(lineno)
                format_errors += 1
                continue
            if not rec.get("output", "").strip():
                missing_fields.append(lineno)
                format_errors += 1
                continue
            records.append(rec)

    total = len(records) + format_errors

    if format_only:
        _print_format_report(path, total, format_errors, missing_fields)
        return {"file": str(path), "total": total, "format_errors": format_errors}

    if not records:
        print(f"\n=== {path.name} ===")
        print(f"  Records      : 0 (format_errors={format_errors})")
        return {"file": str(path), "total": 0, "format_errors": format_errors}

    # ----- Word count distribution -----
    word_counts = sorted(len(r["output"].split()) for r in records)

    # ----- Source distribution -----
    by_source: dict[str, int] = {}
    by_language: dict[str, int] = {}
    for r in records:
        meta = r.get("metadata", {})
        sid = meta.get("source_id", "unknown")
        lang = meta.get("language", "en")
        by_source[sid] = by_source.get(sid, 0) + 1
        by_language[lang] = by_language.get(lang, 0) + 1

    # ----- Exact duplicates -----
    seen_hashes: set[str] = set()
    exact_dups = 0
    import hashlib
    for r in records:
        h = hashlib.md5(r["output"].strip().encode()).hexdigest()
        if h in seen_hashes:
            exact_dups += 1
        else:
            seen_hashes.add(h)

    # ----- Bad samples -----
    bad_samples = sum(1 for r in records if _count_bad(r["output"]) > 0)

    # ----- Print report -----
    print(f"\n{'='*55}")
    print(f"  {path.name}")
    print(f"{'='*55}")
    print(f"  Records       : {len(records):,}")
    print(f"  Format errors : {format_errors}")
    print(f"  Exact dups    : {exact_dups} ({exact_dups/len(records)*100:.1f}%)")
    print(f"  Bad samples   : {bad_samples} ({bad_samples/len(records)*100:.1f}%)")
    print()
    print(f"  Output words  : "
          f"min={word_counts[0]}  "
          f"p25={int(_percentile(word_counts, 25))}  "
          f"median={int(_percentile(word_counts, 50))}  "
          f"p75={int(_percentile(word_counts, 75))}  "
          f"max={word_counts[-1]}")
    print()
    source_str = "  ".join(f"{k}({v})" for k, v in sorted(by_source.items(), key=lambda x: -x[1]))
    print(f"  Sources       : {source_str}")
    lang_str = "  ".join(f"{k}({v})" for k, v in sorted(by_language.items()))
    print(f"  Languages     : {lang_str}")

    if sample_n > 0:
        _print_samples(records, sample_n)

    return {
        "file": str(path),
        "total": total,
        "valid_records": len(records),
        "format_errors": format_errors,
        "exact_dups": exact_dups,
        "bad_samples": bad_samples,
        "word_count_median": int(_percentile(word_counts, 50)),
        "by_source": by_source,
        "by_language": by_language,
    }


def _print_format_report(
    path: pathlib.Path,
    total: int,
    format_errors: int,
    bad_lines: list[int],
) -> None:
    print(f"\n=== {path.name} (format check) ===")
    print(f"  Total lines   : {total}")
    print(f"  Format errors : {format_errors}")
    if bad_lines:
        print(f"  Bad lines     : {bad_lines[:20]}" + (" ..." if len(bad_lines) > 20 else ""))
    else:
        print("  Format        : OK")


def _print_samples(records: list[dict], n: int) -> None:
    sample = random.sample(records, min(n, len(records)))
    print(f"\n  --- {len(sample)} random sample(s) ---")
    for i, r in enumerate(sample, 1):
        meta = r.get("metadata", {})
        sid = meta.get("source_id", "?")
        lang = meta.get("language", "?")
        words = len(r["output"].split())
        print(f"\n  [{i}] source={sid} lang={lang} words={words}")
        print(f"  instruction: {r['instruction'][:120]}")
        # Print first 200 chars of output
        out_preview = r["output"][:200].replace("\n", " ")
        print(f"  output[:200]: {out_preview}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASET_FILES = {
    "rpg": "rpg_dataset.jsonl",
    "literature": "literature_dataset.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate fine-tuning JSONL datasets and print quality reports."
    )
    parser.add_argument(
        "--dataset", choices=["rpg", "literature"],
        help="Which dataset to validate (default: both).",
    )
    parser.add_argument(
        "--sample", type=int, default=0, metavar="N",
        help="Print N random output samples per dataset.",
    )
    parser.add_argument(
        "--format-check", action="store_true",
        help="Only check JSON format; skip content analysis.",
    )
    parser.add_argument(
        "--output-dir", default="data/finetune",
        help="Directory containing JSONL files (default: data/finetune).",
    )
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    datasets = [args.dataset] if args.dataset else ["rpg", "literature"]

    all_ok = True
    for ds in datasets:
        fname = DATASET_FILES[ds]
        result = validate_file(
            output_dir / fname,
            sample_n=args.sample,
            format_only=args.format_check,
        )
        if result.get("format_errors", 0) > 0 or result.get("bad_samples", 0) > 0:
            all_ok = False

    print()
    if all_ok:
        log.info("All datasets passed validation.")
    else:
        log.warning("Some datasets have issues — review the report above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
