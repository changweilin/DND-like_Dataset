"""
scheduler.py — Periodic scraper runner for DND-like_Dataset.

Reads scraper_config.yaml for timing and run-mode settings, then executes
scraper.py on the configured interval (default: once per 24 hours).

Usage:
    python scheduler.py                   # follow config interval (default 24h)
    python scheduler.py --interval 6      # override interval to every 6 hours
    python scheduler.py --run-now         # run immediately, then follow schedule
    python scheduler.py --once            # run once and exit (no loop)
    python scheduler.py --dry-run         # pass --dry-run to scraper; no files written
    python scheduler.py --config alt.yaml # use a different config file
"""

import argparse
import datetime
import logging
import pathlib
import subprocess
import sys
import time

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scheduler")

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

CONFIG_PATH = pathlib.Path("scraper_config.yaml")
DEFAULT_INTERVAL_HOURS = 24.0


def load_schedule_config(config_path: pathlib.Path) -> dict:
    """Return the [schedule] section of scraper_config.yaml, or empty dict."""
    if not _YAML_AVAILABLE or not config_path.exists():
        return {}
    try:
        with open(config_path, encoding="utf-8") as fh:
            raw = _yaml.safe_load(fh) or {}
        return raw.get("schedule", {})
    except Exception as exc:
        log.warning(f"Could not read {config_path}: {exc}; using defaults.")
        return {}


def build_scraper_cmd(
    sched_cfg: dict,
    config_path: pathlib.Path,
    dry_run: bool,
) -> list[str]:
    """Build the subprocess command list for scraper.py."""
    cmd = [sys.executable, "scraper.py", "--config", str(config_path)]
    if sched_cfg.get("update_check", True):
        cmd.append("--update-check")
    if sched_cfg.get("interleave", True):
        cmd.append("--interleave")
    if not sched_cfg.get("respect_robots", True):
        cmd.append("--no-robots")
    if dry_run:
        cmd.append("--dry-run")
    return cmd

# ---------------------------------------------------------------------------
# Run logic
# ---------------------------------------------------------------------------

def run_scraper(cmd: list[str], dry_run: bool) -> int:
    """Execute scraper.py as a subprocess. Returns the exit code."""
    log.info(f"Running: {' '.join(cmd)}")
    if dry_run:
        log.info("(scheduler --dry-run: not executing)")
        return 0
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.warning(f"scraper.py exited with code {result.returncode}")
    return result.returncode


def run_build(dry_run: bool) -> int:
    """Execute build_dataset.py after a scrape run. Returns the exit code."""
    if dry_run:
        log.info("(scheduler --dry-run: skipping build_dataset.py)")
        return 0
    cmd = [sys.executable, "build_dataset.py"]
    log.info(f"auto_build: Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.warning(f"build_dataset.py exited with code {result.returncode}")
    return result.returncode


def _fmt_next(dt: datetime.datetime) -> str:
    delta = dt - datetime.datetime.now()
    h, rem = divmod(int(delta.total_seconds()), 3600)
    m = rem // 60
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S')} (in {h}h {m:02d}m)"

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Periodic scheduler for scraper.py — runs on a configurable interval."
    )
    parser.add_argument(
        "--config", type=pathlib.Path, default=CONFIG_PATH, metavar="FILE",
        help=f"Config file passed to scraper.py and used for schedule settings "
             f"(default: {CONFIG_PATH}).",
    )
    parser.add_argument(
        "--interval", type=float, metavar="HOURS",
        help=(
            f"Override the schedule.interval_hours from config. "
            f"Default: {DEFAULT_INTERVAL_HOURS}h (or whatever config says)."
        ),
    )
    parser.add_argument(
        "--run-now", action="store_true",
        help="Run scraper immediately before entering the schedule loop.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run scraper once and exit (no recurring loop).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Pass --dry-run to scraper.py; no HTTP requests or files written.",
    )
    args = parser.parse_args()

    sched_cfg = load_schedule_config(args.config)
    interval_hours = args.interval or sched_cfg.get("interval_hours", DEFAULT_INTERVAL_HOURS)
    interval_s = interval_hours * 3600
    cmd = build_scraper_cmd(sched_cfg, args.config, dry_run=args.dry_run)

    auto_build = sched_cfg.get("auto_build", False)
    log.info(
        f"Scheduler started | interval={interval_hours:.1f}h | "
        f"update_check={sched_cfg.get('update_check', True)} | "
        f"interleave={sched_cfg.get('interleave', True)} | "
        f"auto_build={auto_build}"
    )
    log.info(f"Scraper command: {' '.join(cmd)}")

    # ---- Run immediately if requested ----
    if args.run_now or args.once:
        run_scraper(cmd, dry_run=args.dry_run)
        if auto_build:
            run_build(dry_run=args.dry_run)
        if args.once:
            log.info("--once: exiting after single run.")
            return

    # ---- Recurring loop ----
    while True:
        next_run_dt = datetime.datetime.now() + datetime.timedelta(seconds=interval_s)
        log.info(f"Next run: {_fmt_next(next_run_dt)}")

        # Sleep in small slices so Ctrl-C interrupts promptly
        deadline = time.monotonic() + interval_s
        while time.monotonic() < deadline:
            time.sleep(min(60, deadline - time.monotonic()))

        log.info("Scheduled run starting…")
        run_scraper(cmd, dry_run=args.dry_run)
        if auto_build:
            run_build(dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Scheduler stopped by user (KeyboardInterrupt).")
        sys.exit(0)
