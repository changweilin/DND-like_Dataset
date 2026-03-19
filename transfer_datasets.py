"""
transfer_datasets.py — 自動搬移已轉換完畢資料集

功能：
  - 掃描 transfer_config.yaml 中設定的來源目錄
  - 自動識別每個 .jsonl 檔的資料集類型（ShareGPT / Alpaca 格式、lora 名稱）
  - 比對 transfer_state.json，判斷哪些檔案已搬移、哪些待搬移
  - 依照 config 設定的目標路徑執行複製或移動
  - 可選驗證 MD5 校驗碼確保資料完整性
  - 支援單次執行與定期迴圈模式

Usage:
    python transfer_datasets.py                # 執行一次，依 config 決定 copy/move
    python transfer_datasets.py --loop         # 定期執行（間隔由 config.interval_hours）
    python transfer_datasets.py --dry-run      # 模擬執行，不實際搬移
    python transfer_datasets.py --status       # 只顯示目前狀態，不搬移
    python transfer_datasets.py --reset FILE   # 重置指定檔案的搬移記錄（允許重新搬移）
    python transfer_datasets.py --config PATH  # 使用指定的設定檔
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import pathlib
import shutil
import sys
import time

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("transfer_datasets")

# ---------------------------------------------------------------------------
# DATASET TYPE DETECTION
# ---------------------------------------------------------------------------

# 依照檔名關鍵字對應類型（優先匹配）
_FILENAME_TYPE_MAP: dict[str, str] = {
    "lora_storyteller":  "lora_storyteller",
    "lora_analyst":      "lora_analyst",
    "lora_translator":   "lora_translator",
    "rpg_dataset":       "rpg_dataset",
    "literature_dataset":"literature_dataset",
}

# 用於格式識別的欄位特徵
_FORMAT_SIGNATURES = {
    "sharegpt": lambda rec: "conversations" in rec and isinstance(rec["conversations"], list),
    "alpaca":   lambda rec: "instruction" in rec and "output" in rec,
}


def detect_dataset_type(jsonl_path: pathlib.Path) -> tuple[str, str]:
    """
    Return (type_key, format_name) for a JSONL file.

    type_key   — matches a key in transfer_config.yaml [targets]
                 e.g. "lora_storyteller", "rpg_dataset", "unknown"
    format_name — "sharegpt", "alpaca", or "unknown"
    """
    stem = jsonl_path.stem.lower()

    # 1. Filename-based type detection
    type_key = "unknown"
    for pattern, mapped in _FILENAME_TYPE_MAP.items():
        if pattern in stem:
            type_key = mapped
            break

    # 2. Content-based format detection (read first non-empty line)
    format_name = "unknown"
    try:
        with jsonl_path.open(encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    rec = json.loads(raw_line)
                except json.JSONDecodeError:
                    break
                for fmt, check in _FORMAT_SIGNATURES.items():
                    if check(rec):
                        format_name = fmt
                        break
                break
    except OSError:
        pass

    return type_key, format_name


# ---------------------------------------------------------------------------
# MD5 CHECKSUM
# ---------------------------------------------------------------------------

def md5_file(path: pathlib.Path, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# STATE MANAGEMENT
# ---------------------------------------------------------------------------

class TransferState:
    """
    Persistent state file (JSON) tracking transferred files.

    Schema per entry:
    {
      "abs_source": str,          # absolute source path at time of transfer
      "type_key": str,            # dataset type key
      "format": str,              # sharegpt | alpaca | unknown
      "size_bytes": int,
      "md5": str,
      "dest_path": str,           # where it was copied/moved to
      "transferred_at": str,      # ISO-8601 timestamp
      "mode": str,                # copy | move
      "verified": bool            # checksum verified at destination
    }
    """

    def __init__(self, state_path: pathlib.Path) -> None:
        self.path = state_path
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
                log.debug("Loaded transfer state: %d entries", len(self._data))
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Could not read state file %s: %s — starting fresh", self.path, e)
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _key(self, source: pathlib.Path) -> str:
        return str(source.resolve())

    def is_transferred(self, source: pathlib.Path, current_md5: str) -> bool:
        """Return True if this file has already been transferred with the same MD5."""
        entry = self._data.get(self._key(source))
        if not entry:
            return False
        return entry.get("md5") == current_md5

    def record(
        self,
        source: pathlib.Path,
        type_key: str,
        format_name: str,
        md5: str,
        dest_path: pathlib.Path,
        mode: str,
        verified: bool,
    ) -> None:
        self._data[self._key(source)] = {
            "abs_source":      str(source.resolve()),
            "type_key":        type_key,
            "format":          format_name,
            "size_bytes":      source.stat().st_size if source.exists() else 0,
            "md5":             md5,
            "dest_path":       str(dest_path.resolve()),
            "transferred_at":  datetime.datetime.now().isoformat(timespec="seconds"),
            "mode":            mode,
            "verified":        verified,
        }

    def reset(self, source: pathlib.Path) -> bool:
        key = self._key(source)
        if key in self._data:
            del self._data[key]
            return True
        return False

    def all_entries(self) -> list[dict]:
        return list(self._data.values())

    def get_entry(self, source: pathlib.Path) -> dict | None:
        return self._data.get(self._key(source))


# ---------------------------------------------------------------------------
# CONFIG LOADER
# ---------------------------------------------------------------------------

def load_config(config_path: pathlib.Path) -> dict:
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        sys.exit(1)

    if not _YAML_OK:
        log.error("PyYAML is not installed. Run: pip install pyyaml")
        sys.exit(1)

    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


# ---------------------------------------------------------------------------
# SOURCE SCANNING
# ---------------------------------------------------------------------------

def scan_sources(source_dirs: list[str], project_root: pathlib.Path) -> list[pathlib.Path]:
    """
    Return all .jsonl files found in the configured source directories.
    Top-level directories are scanned non-recursively; subdirectories that
    appear in the list are scanned one level deep.
    """
    found: list[pathlib.Path] = []
    for src_str in source_dirs:
        src_path = pathlib.Path(src_str)
        if not src_path.is_absolute():
            src_path = project_root / src_path
        if not src_path.exists():
            log.warning("Source directory not found, skipping: %s", src_path)
            continue
        # Only scan immediate .jsonl files (non-recursive) to avoid picking up
        # sub-directories that have their own source entry
        for jf in sorted(src_path.glob("*.jsonl")):
            if jf not in found:
                found.append(jf)
    return found


# ---------------------------------------------------------------------------
# TRANSFER ENGINE
# ---------------------------------------------------------------------------

def transfer_file(
    src: pathlib.Path,
    dest_dir: pathlib.Path,
    mode: str,
    overwrite: bool,
    verify: bool,
    src_md5: str,
    dry_run: bool,
) -> tuple[bool, bool, pathlib.Path]:
    """
    Copy or move src to dest_dir/src.name.

    Returns (success, verified, dest_path).
    """
    dest_path = dest_dir / src.name

    if dest_path.exists() and not overwrite:
        log.warning("  SKIP (already exists, overwrite=false): %s", dest_path)
        return False, False, dest_path

    if dry_run:
        log.info("  [DRY-RUN] Would %s -> %s", mode.upper(), dest_path)
        return True, True, dest_path

    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        if mode == "move":
            shutil.move(str(src), str(dest_path))
        else:  # copy
            shutil.copy2(str(src), str(dest_path))
    except OSError as e:
        log.error("  FAILED (%s): %s", mode, e)
        return False, False, dest_path

    # Checksum verification
    verified = False
    if verify and dest_path.exists():
        dest_md5 = md5_file(dest_path)
        if dest_md5 == src_md5:
            verified = True
        else:
            log.error(
                "  CHECKSUM MISMATCH after %s! src=%s dest=%s",
                mode, src_md5, dest_md5,
            )
            return False, False, dest_path

    return True, verified, dest_path


# ---------------------------------------------------------------------------
# STATUS REPORT
# ---------------------------------------------------------------------------

def print_status(
    all_files: list[pathlib.Path],
    state: TransferState,
    targets: dict[str, str],
) -> None:
    """Print a formatted status table to stdout."""
    RESET  = "\033[0m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    RED    = "\033[31m"
    CYAN   = "\033[36m"
    BOLD   = "\033[1m"

    print(f"\n{BOLD}{'=' * 72}{RESET}")
    print(f"{BOLD}  資料集搬移狀態報告  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{BOLD}{'=' * 72}{RESET}")

    pending_count = 0
    done_count = 0
    skip_count = 0

    for fpath in sorted(all_files):
        type_key, format_name = detect_dataset_type(fpath)
        size_mb = fpath.stat().st_size / 1_048_576
        target = targets.get(type_key, "")

        entry = state.get_entry(fpath)
        current_md5 = md5_file(fpath)
        already = state.is_transferred(fpath, current_md5)

        if already:
            status_str = f"{GREEN}✓ 已搬移{RESET}"
            dest = entry.get("dest_path", "?") if entry else "?"
            done_count += 1
        elif not target:
            status_str = f"{YELLOW}— 跳過 (無目標){RESET}"
            dest = "（config 未填寫目標路徑）"
            skip_count += 1
        else:
            status_str = f"{RED}● 待搬移{RESET}"
            dest = target
            pending_count += 1

        print(f"\n  {BOLD}{fpath.name}{RESET}")
        print(f"    類型:   {CYAN}{type_key}{RESET}  [{format_name}]")
        print(f"    大小:   {size_mb:.2f} MB")
        print(f"    狀態:   {status_str}")
        print(f"    目標:   {dest}")
        if entry and already:
            print(f"    時間:   {entry.get('transferred_at','?')}  mode={entry.get('mode','?')}  verified={entry.get('verified','?')}")

    print(f"\n{BOLD}{'─' * 72}{RESET}")
    print(
        f"  合計: {GREEN}{done_count} 已完成{RESET} | "
        f"{RED}{pending_count} 待搬移{RESET} | "
        f"{YELLOW}{skip_count} 跳過{RESET}"
    )
    print(f"{BOLD}{'=' * 72}{RESET}\n")


# ---------------------------------------------------------------------------
# MAIN TRANSFER LOOP
# ---------------------------------------------------------------------------

def run_once(
    cfg: dict,
    project_root: pathlib.Path,
    state: TransferState,
    dry_run: bool,
) -> dict[str, int]:
    """Scan sources, transfer pending files. Returns counts."""
    targets: dict[str, str] = cfg.get("targets", {})
    source_dirs: list[str] = cfg.get("sources", [])
    t_cfg: dict = cfg.get("transfer", {})
    mode: str       = t_cfg.get("mode", "copy")
    overwrite: bool = t_cfg.get("overwrite", False)
    verify: bool    = t_cfg.get("verify_checksum", True)

    all_files = scan_sources(source_dirs, project_root)
    log.info("Found %d JSONL file(s) in source directories.", len(all_files))

    counts = {"transferred": 0, "skipped_done": 0, "skipped_no_target": 0, "failed": 0}

    for fpath in all_files:
        type_key, format_name = detect_dataset_type(fpath)
        size_mb = fpath.stat().st_size / 1_048_576

        log.info(
            "Processing: %s  [type=%s, format=%s, %.2f MB]",
            fpath.name, type_key, format_name, size_mb,
        )

        # Compute MD5 (used for dedup + verification)
        src_md5 = md5_file(fpath)

        # Already transferred?
        if state.is_transferred(fpath, src_md5):
            log.info("  SKIP — already transferred (MD5 match).")
            counts["skipped_done"] += 1
            continue

        # Has a target?
        target_str = targets.get(type_key, "")
        if not target_str:
            log.info("  SKIP — no target configured for type '%s'.", type_key)
            counts["skipped_no_target"] += 1
            continue

        dest_dir = pathlib.Path(target_str)
        if not dest_dir.is_absolute():
            dest_dir = project_root / dest_dir

        success, verified, dest_path = transfer_file(
            src=fpath,
            dest_dir=dest_dir,
            mode=mode,
            overwrite=overwrite,
            verify=verify,
            src_md5=src_md5,
            dry_run=dry_run,
        )

        if success:
            log.info(
                "  %s -> %s  [verified=%s]",
                mode.upper(), dest_path, verified,
            )
            if not dry_run:
                state.record(fpath, type_key, format_name, src_md5, dest_path, mode, verified)
                state.save()
            counts["transferred"] += 1
        else:
            counts["failed"] += 1

    return counts


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="自動搬移已轉換完畢的 LoRA 訓練資料集。"
    )
    parser.add_argument(
        "--config", default="transfer_config.yaml",
        help="設定檔路徑 (預設: transfer_config.yaml)",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="定期迴圈執行（間隔由 config.transfer.interval_hours 決定）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="模擬執行，顯示將執行的操作但不實際搬移檔案",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="只顯示目前狀態報告，不搬移",
    )
    parser.add_argument(
        "--reset", metavar="FILE",
        help="重置指定檔案的搬移記錄（路徑或檔名），允許重新搬移",
    )
    args = parser.parse_args()

    project_root = pathlib.Path(__file__).parent.resolve()
    config_path  = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    cfg = load_config(config_path)

    state_path_str = cfg.get("state_file", "data/transfer_state.json")
    state_path = pathlib.Path(state_path_str)
    if not state_path.is_absolute():
        state_path = project_root / state_path

    state = TransferState(state_path)

    # ── --reset ────────────────────────────────────────────────────────────
    if args.reset:
        target = pathlib.Path(args.reset)
        if not target.is_absolute():
            target = project_root / target
        if state.reset(target):
            state.save()
            log.info("Reset transfer record for: %s", target)
        else:
            log.warning("No record found for: %s", target)
        return

    # ── --status ───────────────────────────────────────────────────────────
    if args.status:
        source_dirs = cfg.get("sources", [])
        all_files = scan_sources(source_dirs, project_root)
        targets = cfg.get("targets", {})
        print_status(all_files, state, targets)
        return

    # ── single run or loop ─────────────────────────────────────────────────
    if args.dry_run:
        log.info("*** DRY-RUN 模式 — 不會實際搬移任何檔案 ***")

    if args.loop:
        interval_hours: float = cfg.get("transfer", {}).get("interval_hours", 6)
        interval_sec = interval_hours * 3600
        log.info("Loop mode: 每 %.1f 小時執行一次 (Ctrl-C 停止)", interval_hours)
        while True:
            log.info("--- 開始本輪搬移 ---")
            counts = run_once(cfg, project_root, state, dry_run=args.dry_run)
            log.info(
                "本輪完成: transferred=%d, already_done=%d, no_target=%d, failed=%d",
                counts["transferred"], counts["skipped_done"],
                counts["skipped_no_target"], counts["failed"],
            )
            next_run = datetime.datetime.now() + datetime.timedelta(seconds=interval_sec)
            log.info("下次執行時間: %s", next_run.strftime("%Y-%m-%d %H:%M:%S"))
            try:
                time.sleep(interval_sec)
            except KeyboardInterrupt:
                log.info("收到中斷信號，結束迴圈。")
                break
    else:
        counts = run_once(cfg, project_root, state, dry_run=args.dry_run)
        log.info(
            "完成: transferred=%d, already_done=%d, no_target=%d, failed=%d",
            counts["transferred"], counts["skipped_done"],
            counts["skipped_no_target"], counts["failed"],
        )


if __name__ == "__main__":
    main()
