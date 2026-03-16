"""
export_hf.py — Export fine-tuning JSONL datasets to HuggingFace-compatible format.

Reads data/finetune/*.jsonl, shuffles with a fixed seed, splits into train/validation,
and writes the split files to data/hf_export/.

Optionally uploads to the HuggingFace Hub if the `datasets` package is installed
and a repo ID is provided.

Usage:
    python export_hf.py                          # export both datasets (90/10 split)
    python export_hf.py --dataset rpg            # only rpg_dataset.jsonl
    python export_hf.py --dataset literature
    python export_hf.py --split 0.95             # 95% train / 5% validation
    python export_hf.py --upload YOUR/REPO       # upload to HuggingFace Hub
    python export_hf.py --output-dir PATH        # custom export directory
    python export_hf.py --input-dir PATH         # custom input directory
"""

import argparse
import json
import logging
import pathlib
import random
import sys

# Optional HuggingFace integration (not required for local export)
try:
    from datasets import Dataset as _HFDataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("export_hf")

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

SHUFFLE_SEED = 42

DATASET_FILES = {
    "rpg": "rpg_dataset.jsonl",
    "literature": "literature_dataset.jsonl",
}


def load_jsonl(path: pathlib.Path) -> list[dict]:
    """Load all valid JSON lines from a JSONL file."""
    records = []
    if not path.exists():
        log.warning(f"File not found: {path}")
        return records
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning(f"  Skipping line {lineno} in {path.name}: {exc}")
    return records


def split_dataset(
    records: list[dict],
    train_ratio: float,
    seed: int = SHUFFLE_SEED,
) -> tuple[list[dict], list[dict]]:
    """Shuffle with a fixed seed and split into train/validation."""
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * train_ratio))
    return shuffled[:cut], shuffled[cut:]


def write_jsonl(records: list[dict], path: pathlib.Path) -> None:
    """Write records to a JSONL file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info(f"  Wrote {len(records):,} records -> {path}")


def export_dataset(
    name: str,
    input_path: pathlib.Path,
    export_dir: pathlib.Path,
    train_ratio: float,
) -> dict:
    """
    Load one JSONL, split, write to export_dir/<name>/train.jsonl and
    validation.jsonl. Returns summary dict.
    """
    records = load_jsonl(input_path)
    if not records:
        log.warning(f"  {name}: no records found, skipping.")
        return {"name": name, "total": 0}

    train, val = split_dataset(records, train_ratio)
    ds_dir = export_dir / name
    write_jsonl(train, ds_dir / "train.jsonl")
    write_jsonl(val, ds_dir / "validation.jsonl")

    log.info(
        f"  {name}: {len(records):,} total → "
        f"train={len(train):,} ({len(train)/len(records)*100:.0f}%) | "
        f"val={len(val):,} ({len(val)/len(records)*100:.0f}%)"
    )
    return {
        "name": name,
        "total": len(records),
        "train": len(train),
        "validation": len(val),
        "train_path": str(ds_dir / "train.jsonl"),
        "validation_path": str(ds_dir / "validation.jsonl"),
    }


# ---------------------------------------------------------------------------
# HuggingFace Hub upload (optional)
# ---------------------------------------------------------------------------

def upload_to_hub(export_dir: pathlib.Path, repo_id: str, datasets: list[str]) -> None:
    """Upload exported splits to HuggingFace Hub."""
    if not _HF_AVAILABLE:
        log.error(
            "HuggingFace `datasets` library is not installed. "
            "Run: pip install datasets huggingface_hub"
        )
        sys.exit(1)

    for ds_name in datasets:
        train_path = export_dir / ds_name / "train.jsonl"
        val_path = export_dir / ds_name / "validation.jsonl"
        if not train_path.exists():
            log.warning(f"  {ds_name}: no exported file at {train_path}, skipping upload.")
            continue

        train_records = load_jsonl(train_path)
        val_records = load_jsonl(val_path)

        hf_ds = _HFDataset.from_list(train_records + val_records)
        full_repo = f"{repo_id}-{ds_name}" if len(datasets) > 1 else repo_id
        log.info(f"  Uploading {ds_name} ({len(hf_ds)} records) to {full_repo} …")
        hf_ds.push_to_hub(full_repo)
        log.info(f"  Upload complete: https://huggingface.co/datasets/{full_repo}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export JSONL datasets to HuggingFace-compatible train/validation splits."
    )
    parser.add_argument(
        "--dataset", choices=["rpg", "literature"],
        help="Which dataset to export (default: both).",
    )
    parser.add_argument(
        "--split", type=float, default=0.9, metavar="RATIO",
        help="Fraction of data used for training split (default: 0.9).",
    )
    parser.add_argument(
        "--upload", metavar="HF_REPO_ID",
        help="Upload exported datasets to this HuggingFace Hub repo ID.",
    )
    parser.add_argument(
        "--input-dir", default="data/finetune",
        help="Directory containing source JSONL files (default: data/finetune).",
    )
    parser.add_argument(
        "--output-dir", default="data/hf_export",
        help="Directory for exported train/validation splits (default: data/hf_export).",
    )
    args = parser.parse_args()

    if not (0.5 <= args.split <= 0.99):
        log.error("--split must be between 0.5 and 0.99")
        sys.exit(1)

    input_dir = pathlib.Path(args.input_dir)
    export_dir = pathlib.Path(args.output_dir)
    datasets_to_export = [args.dataset] if args.dataset else ["rpg", "literature"]

    log.info(
        f"Exporting {datasets_to_export} | "
        f"train/val split: {args.split:.0%}/{1-args.split:.0%} | "
        f"output: {export_dir}"
    )

    summaries = []
    for ds_name in datasets_to_export:
        fname = DATASET_FILES[ds_name]
        summary = export_dataset(ds_name, input_dir / fname, export_dir, args.split)
        summaries.append(summary)

    # Print summary
    print()
    print(f"{'='*55}")
    print("  EXPORT SUMMARY")
    print(f"{'='*55}")
    for s in summaries:
        if s["total"] == 0:
            print(f"  {s['name']:<15} (no data)")
        else:
            print(
                f"  {s['name']:<15} total={s['total']:,}  "
                f"train={s['train']:,}  val={s['validation']:,}"
            )
    print(f"{'='*55}")
    print(f"  Output dir: {export_dir.resolve()}")
    print()

    if args.upload:
        log.info(f"Uploading to HuggingFace Hub: {args.upload}")
        upload_to_hub(export_dir, args.upload, datasets_to_export)

    log.info("Done.")


if __name__ == "__main__":
    main()
