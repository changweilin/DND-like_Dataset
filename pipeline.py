"""
pipeline.py — End-to-end LLM dataset pipeline orchestrator.

Runs the full workflow in sequence:
  1. scraper.py      — fetch / update raw text files
  2. build_dataset.py — build Alpaca-format JSONL from raw text
  3. validate_dataset.py — print quality report
  4. postprocess_rl.py — (optional) RL post-processing: clean/filter/convert
  5. export_hf.py    — (optional) export train/val splits

Usage:
    python pipeline.py                    # full pipeline
    python pipeline.py --skip-scrape      # skip scraping, only build+validate
    python pipeline.py --skip-build       # only scrape (no build/validate)
    python pipeline.py --export           # also run export_hf.py at the end
    python pipeline.py --postprocess      # run postprocess_rl.py after validate
    python pipeline.py --category trpg   # limit scrape+build to trpg category
    python pipeline.py --fresh            # pass --fresh to build_dataset.py
    python pipeline.py --dry-run          # dry-run all steps (no writes)
    python pipeline.py --fail-fast        # abort on first failed step
    python pipeline.py --config FILE      # use alternate config (default: scraper_config.yaml)
"""

import argparse
import logging
import pathlib
import subprocess
import sys

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

CONFIG_PATH = pathlib.Path("scraper_config.yaml")


def run_step(cmd: list[str], step_name: str, fail_fast: bool) -> int:
    """Run a subprocess step. Returns exit code. Logs outcome."""
    log.info(f"{'='*60}")
    log.info(f"STEP: {step_name}")
    log.info(f"CMD : {' '.join(cmd)}")
    log.info(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        log.info(f"STEP {step_name}: OK")
    else:
        log.warning(f"STEP {step_name}: exit code {result.returncode}")
        if fail_fast:
            log.error("--fail-fast: aborting pipeline.")
            sys.exit(result.returncode)
    return result.returncode


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _scraper_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "scraper.py", "--config", str(args.config)]
    if args.update_check:
        cmd.append("--update-check")
    if args.interleave:
        cmd.append("--interleave")
    if args.category:
        cmd += ["--category", args.category]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.no_robots:
        cmd.append("--no-robots")
    return cmd


def _build_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "build_dataset.py"]
    if args.category:
        cmd += ["--category", args.category]
    if args.fresh:
        cmd.append("--fresh")
    if args.dry_run:
        # build_dataset.py has no --dry-run; simulate with --stats (no writes)
        cmd.append("--stats")
    return cmd


def _validate_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "validate_dataset.py"]
    # dataset filter follows category
    if args.category == "webnovel":
        cmd += ["--dataset", "literature"]
    elif args.category:
        cmd += ["--dataset", "rpg"]
    return cmd


def _postprocess_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "postprocess_rl.py"]
    if args.dry_run:
        cmd.append("--stats")
    return cmd


def _export_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "export_hf.py"]
    if args.category == "webnovel":
        cmd += ["--dataset", "literature"]
    elif args.category:
        cmd += ["--dataset", "rpg"]
    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: scrape → build → validate → (export)"
    )
    parser.add_argument(
        "--config", type=pathlib.Path, default=CONFIG_PATH, metavar="FILE",
        help=f"Config file for scraper.py (default: {CONFIG_PATH}).",
    )
    parser.add_argument(
        "--skip-scrape", action="store_true",
        help="Skip scraper.py — start from build_dataset.py.",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Only run scraper.py; skip build/validate/export.",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Run export_hf.py as the final step.",
    )
    parser.add_argument(
        "--category",
        help="Limit scrape and build to one category (e.g. trpg, webnovel, extra_lore).",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Pass --fresh to build_dataset.py (overwrite JSONL instead of append).",
    )
    parser.add_argument(
        "--update-check", action="store_true", default=True,
        help="Pass --update-check to scraper.py (default: on).",
    )
    parser.add_argument(
        "--interleave", action="store_true", default=True,
        help="Pass --interleave to scraper.py (default: on).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Dry-run mode: scraper writes nothing; build prints stats only.",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Abort pipeline on first step that exits non-zero.",
    )
    parser.add_argument(
        "--no-robots", action="store_true",
        help="Skip robots.txt checks in scraper.py.",
    )
    parser.add_argument(
        "--postprocess", action="store_true",
        help="Run postprocess_rl.py after validate (RL data cleaning/conversion).",
    )
    args = parser.parse_args()

    results: dict[str, int] = {}

    # Step 1: Scrape
    if not args.skip_scrape:
        results["scrape"] = run_step(_scraper_cmd(args), "scrape", args.fail_fast)

    if args.skip_build:
        log.info("--skip-build: stopping after scrape.")
        _print_summary(results)
        return

    # Step 2: Build dataset
    results["build"] = run_step(_build_cmd(args), "build", args.fail_fast)

    # Step 3: Validate
    results["validate"] = run_step(_validate_cmd(args), "validate", args.fail_fast)

    # Step 4 (optional): RL post-processing
    if args.postprocess:
        results["postprocess"] = run_step(
            _postprocess_cmd(args), "postprocess", args.fail_fast
        )

    # Step 5 (optional): Export
    if args.export:
        results["export"] = run_step(_export_cmd(args), "export", args.fail_fast)

    _print_summary(results)


def _print_summary(results: dict[str, int]) -> None:
    log.info("=" * 60)
    log.info("PIPELINE SUMMARY")
    ok = all(v == 0 for v in results.values())
    for step, code in results.items():
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        log.info(f"  {step:<12} {status}")
    log.info("=" * 60)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
