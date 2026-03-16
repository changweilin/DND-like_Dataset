"""
Fine-tuning dataset builder for DND-like_Dataset project.

Reads raw scraped text from data/raw/, applies quality filtering and chunking,
then outputs Alpaca-format JSONL files for two training objectives:
  1. rpg_dataset.jsonl     — GM narration, world lore, scenario descriptions
  2. literature_dataset.jsonl — creative writing in web fiction styles

Usage:
    python build_dataset.py                      # build both datasets
    python build_dataset.py --dataset rpg
    python build_dataset.py --dataset literature
    python build_dataset.py --category trpg
    python build_dataset.py --category webnovel
    python build_dataset.py --fresh              # overwrite instead of append
    python build_dataset.py --stats              # show stats only, no write
    python build_dataset.py --min-words 100      # override quality threshold
"""

import argparse
import hashlib
import json
import logging
import pathlib
import random
import re
import sys
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("build_dataset")

# ---------------------------------------------------------------------------
# INSTRUCTION TEMPLATE POOLS
# ---------------------------------------------------------------------------

RPG_INSTRUCTION_TEMPLATES = [
    "You are a Game Master for a {system} campaign. Describe the following location or faction to the players in vivid, atmospheric prose: {topic}",
    "As a narrator for {system}, write a lore entry about: {topic}",
    "Create a GM handout for players about the following {system} worldbuilding element: {topic}",
    "Write a world almanac entry in the style of {system} about: {topic}",
    "You are writing flavor text for a {system} sourcebook. The subject is: {topic}",
    "A scholar in the world of {system} writes in their journal about: {topic}",
    "Describe the history and significance of the following {system} concept for new players: {topic}",
    "Write an in-world document from the universe of {system} about: {topic}",
    "As the narrator of a {system} adventure, set the scene for: {topic}",
    "Provide a detailed lore description suitable for a {system} campaign sourcebook about: {topic}",
]

LITERATURE_INSTRUCTION_TEMPLATES = [
    "Write a passage in the style of {genre} web fiction. Continue or respond to the following creative writing prompt: {derived_prompt}",
    "You are writing a {genre} story. Continue the narrative: {derived_prompt}",
    "In the {genre} genre, write a scene that develops from: {derived_prompt}",
    "As a {genre} author, write the next passage given this setup: {derived_prompt}",
    "Complete this {genre} story excerpt with vivid detail: {derived_prompt}",
    "Write the next chapter segment for a {genre} story that begins: {derived_prompt}",
    "Continue this {genre} narrative in an engaging style: {derived_prompt}",
    "Expand upon the following {genre} story beat into a full prose passage: {derived_prompt}",
]

# Mapping from source tags to human-readable genre label
_GENRE_MAP = {
    "litrpg": "LitRPG",
    "progression": "Progression Fantasy",
    "isekai": "Isekai fantasy",
    "fantasy": "epic fantasy",
    "fanfic": "fan fiction",
    "original": "original fantasy",
    "ja": "Japanese light novel",
    "en": "English web novel",
    "horror": "dark horror",
    "scifi": "science fiction",
    "cyberpunk": "cyberpunk",
}

# Mapping from source display_name to system name
_SYSTEM_NAME_MAP = {
    "Pathfinder": "Pathfinder",
    "Warhammer Fantasy": "Warhammer Fantasy Roleplay",
    "Warhammer 40,000": "Warhammer 40,000",
    "Shadowrun": "Shadowrun",
    "World of Darkness": "World of Darkness",
    "Call of Cthulhu": "Call of Cthulhu",
    "Iron Kingdoms": "Iron Kingdoms",
    "Blades in the Dark": "Blades in the Dark",
    "Legend of the Five Rings": "Legend of the Five Rings",
    "Deadlands": "Deadlands",
    "Mutant: Year Zero": "Mutant: Year Zero",
    "Gloomhaven": "Gloomhaven",
    "Lexicanum (Warhammer)": "Warhammer",
}

# ---------------------------------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------------------------------

def split_into_chunks(
    text: str,
    min_words: int = 80,
    max_words: int = 500,
) -> list[str]:
    """
    Split text into training-ready chunks.
    Strategy: split on paragraph breaks first, then sentence boundaries for
    paragraphs that exceed max_words. Merge short consecutive paragraphs.
    Discard chunks below min_words.
    """
    # Normalise line endings and collapse 3+ blank lines
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    raw_paragraphs = text.split("\n\n")
    merged: list[str] = []
    current = ""

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        para_words = len(para.split())

        if para_words > max_words:
            # Flush current buffer
            if current:
                merged.append(current)
                current = ""
            # Split this paragraph on sentence boundaries
            sentences = re.split(r"(?<=[.!?。！？])\s+", para)
            sent_buf = ""
            for sent in sentences:
                candidate = (sent_buf + " " + sent).strip() if sent_buf else sent
                if len(candidate.split()) <= max_words:
                    sent_buf = candidate
                else:
                    if sent_buf:
                        merged.append(sent_buf)
                    sent_buf = sent
            if sent_buf:
                merged.append(sent_buf)
        else:
            candidate = (current + "\n\n" + para).strip() if current else para
            if len(candidate.split()) <= max_words:
                current = candidate
            else:
                if current:
                    merged.append(current)
                current = para

    if current:
        merged.append(current)

    return [c for c in merged if len(c.split()) >= min_words]


# ---------------------------------------------------------------------------
# BOILERPLATE / AD PATTERN FILTER
# ---------------------------------------------------------------------------

# Phrases that reliably indicate non-narrative boilerplate surviving extraction.
# Each pattern is matched case-insensitively against the full chunk text.
_BOILERPLATE_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bcookies?\b.{0,40}\baccept\b",          # "accept cookies"
    r"\bprivacy policy\b",
    r"\bterms of (service|use)\b",
    r"\ball rights reserved\b",
    r"\bcopyright \d{4}\b",
    r"\bsubscribe (now|today|to)\b",
    r"\bnewsletter\b",
    r"\bclick here\b",
    r"\badvertisement\b",
    r"\bsponsored (by|content)\b",
    r"\bskip (to|the) (main |navigation|content)\b",
    r"\blog in\b.{0,20}\bsign up\b",
    r"\bsign (in|up) (to|for)\b",
    r"\bshare (this|on)\b",
    r"\bdisqus\b",
    r"\bjavascript (is |must be )(enabled|required)\b",
]]

# If this many patterns match, the chunk is flagged as boilerplate.
_BOILERPLATE_THRESHOLD = 2

# ---------------------------------------------------------------------------
# LANGUAGE DETECTION HEURISTIC (no external library)
# ---------------------------------------------------------------------------

def _detect_language_mismatch(text: str, expected_language: str) -> bool:
    """
    Return True if the text appears to be in the WRONG language.
    Uses Unicode block ratios — no external dependency.
    """
    if not text:
        return False

    # Count letter characters by Unicode category
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return False
    total = len(letters)

    ascii_letters = sum(1 for c in letters if ord(c) < 128)
    cjk_chars = sum(1 for c in letters if "\u4E00" <= c <= "\u9FFF" or "\u3400" <= c <= "\u4DBF")
    kana_chars = sum(1 for c in letters if "\u3040" <= c <= "\u30FF")
    cyrillic_chars = sum(1 for c in letters if "\u0400" <= c <= "\u04FF")

    if expected_language == "en":
        # English text should be overwhelmingly ASCII letters
        if ascii_letters / total < 0.70:
            return True
    elif expected_language == "ja":
        # Japanese text should have a meaningful CJK + kana presence
        if (cjk_chars + kana_chars) / total < 0.30:
            return True

    return False

# ---------------------------------------------------------------------------
# QUALITY FILTER
# ---------------------------------------------------------------------------

def is_quality_chunk(text: str, language: str = "en") -> bool:
    """Return True if the chunk is suitable for fine-tuning training data."""
    words = text.split()
    if len(words) < 80:
        return False

    # Filter chunks that are mostly single-word lines (navigation debris)
    lines = [l for l in text.splitlines() if l.strip()]
    if lines:
        single_word_ratio = sum(1 for l in lines if len(l.split()) == 1) / len(lines)
        if single_word_ratio > 0.30:
            return False

    # Must contain actual prose characters
    if not re.search(r"[a-zA-Z\u3040-\u30FF\u4E00-\u9FFF]", text):
        return False

    # English: average word length sanity check
    if language == "en":
        avg_word_len = sum(len(w) for w in words) / len(words)
        if not (3.0 <= avg_word_len <= 12.0):
            return False

    # Japanese: at least 10% hiragana/katakana
    if language == "ja":
        jp_chars = sum(1 for c in text if "\u3040" <= c <= "\u30FF")
        if jp_chars / len(text) < 0.10:
            return False

    # Language mismatch detection (heuristic, no external library)
    if _detect_language_mismatch(text, language):
        return False

    # Boilerplate / ad text detection
    matches = sum(1 for p in _BOILERPLATE_PATTERNS if p.search(text))
    if matches >= _BOILERPLATE_THRESHOLD:
        return False

    return True


def deduplicate_chunks(chunks: list[str], threshold: float = 0.85) -> list[str]:
    """Remove near-duplicate chunks using Jaccard similarity on word sets."""
    seen: list[set] = []
    result: list[str] = []
    for chunk in chunks:
        words = set(chunk.lower().split())
        duplicate = False
        for prev_words in seen:
            if not words or not prev_words:
                continue
            intersection = len(words & prev_words)
            union = len(words | prev_words)
            if union > 0 and intersection / union >= threshold:
                duplicate = True
                break
        if not duplicate:
            result.append(chunk)
            seen.append(words)
    return result

# ---------------------------------------------------------------------------
# PROMPT DERIVATION
# ---------------------------------------------------------------------------

def extract_topic_from_chunk(chunk: str) -> str:
    """Extract a short topic label from a chunk for RPG instruction template."""
    lines = chunk.strip().splitlines()
    first_line = lines[0].strip() if lines else ""

    # If first line looks like a heading (short, title-cased or all caps)
    if first_line and len(first_line.split()) <= 8:
        if first_line.istitle() or first_line.isupper() or first_line[0].isupper():
            return first_line[:80]

    # Take first sentence
    match = re.match(r"^(.+?[.!?])\s", chunk)
    if match:
        sentence = match.group(1)
        # Extract first ~8 words after removing common starters
        words = sentence.split()
        words = [w for w in words if w.lower() not in {"the", "a", "an", "in", "on", "at", "during", "within"}]
        return " ".join(words[:8])

    return " ".join(chunk.split()[:8])


def derive_prompt_from_chunk(
    chunk: str,
    prev_chunk: Optional[str],
) -> str:
    """Derive a creative writing prompt from context for literature instruction."""
    if prev_chunk:
        sentences = re.split(r"(?<=[.!?])\s+", prev_chunk.strip())
        last = sentences[-1].strip() if sentences else ""
        if last and len(last.split()) >= 5:
            return last[:200]

    # Use first sentence of current chunk as prompt
    match = re.match(r"^(.+?[.!?])\s", chunk)
    if match:
        return match.group(1)[:200]

    return " ".join(chunk.split()[:20])


def _get_genre_label(tags: list[str]) -> str:
    for tag in tags:
        if tag in _GENRE_MAP:
            return _GENRE_MAP[tag]
    return "fantasy"


def _get_system_name(display_name: str) -> str:
    return _SYSTEM_NAME_MAP.get(display_name, display_name)


# ---------------------------------------------------------------------------
# RECORD BUILDERS
# ---------------------------------------------------------------------------

def build_rpg_record(
    chunk: str,
    chunk_index: int,
    metadata: dict,
    rng: random.Random,
) -> dict:
    """Build one Alpaca-format record for the RPG dataset."""
    system = _get_system_name(metadata.get("display_name", metadata.get("source_id", "TRPG")))
    topic = extract_topic_from_chunk(chunk)
    template = rng.choice(RPG_INSTRUCTION_TEMPLATES)
    instruction = template.format(system=system, topic=topic)

    return {
        "instruction": instruction,
        "input": "",
        "output": chunk,
        "metadata": {
            "source_id": metadata.get("source_id", ""),
            "source_url": metadata.get("url", ""),
            "category": metadata.get("category", "trpg"),
            "display_name": metadata.get("display_name", ""),
            "tags": metadata.get("tags", []),
            "language": metadata.get("language", "en"),
            "chunk_index": chunk_index,
            "word_count": len(chunk.split()),
        },
    }


def build_literature_record(
    chunk: str,
    chunk_index: int,
    metadata: dict,
    prev_chunk: Optional[str],
    rng: random.Random,
) -> dict:
    """Build one Alpaca-format record for the literature dataset."""
    genre = _get_genre_label(metadata.get("tags", []))
    derived_prompt = derive_prompt_from_chunk(chunk, prev_chunk)
    template = rng.choice(LITERATURE_INSTRUCTION_TEMPLATES)
    instruction = template.format(genre=genre, derived_prompt=derived_prompt)

    return {
        "instruction": instruction,
        "input": "",
        "output": chunk,
        "metadata": {
            "source_id": metadata.get("source_id", ""),
            "source_url": metadata.get("url", ""),
            "category": metadata.get("category", "webnovel"),
            "display_name": metadata.get("display_name", ""),
            "tags": metadata.get("tags", []),
            "language": metadata.get("language", "en"),
            "chunk_index": chunk_index,
            "word_count": len(chunk.split()),
        },
    }

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_raw_sources(
    raw_dir: pathlib.Path,
    category: Optional[str] = None,
) -> list[dict]:
    """
    Walk data/raw/, return list of dicts: {"text": str, "metadata": dict}.
    Filters by category if provided.
    """
    sources = []
    if not raw_dir.exists():
        log.warning(f"Raw data directory not found: {raw_dir}")
        return sources

    # Structure: raw_dir / category / source_id / fname.txt + fname.json
    for cat_dir in sorted(raw_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat = cat_dir.name
        if category and cat != category:
            continue
        for source_dir in sorted(cat_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            for txt_file in sorted(source_dir.glob("*.txt")):
                json_file = txt_file.with_suffix(".json")
                if not json_file.exists():
                    log.debug(f"No metadata for {txt_file}, skipping")
                    continue
                try:
                    text = txt_file.read_text(encoding="utf-8")
                    metadata = json.loads(json_file.read_text(encoding="utf-8"))
                    if text.strip():
                        sources.append({"text": text, "metadata": metadata})
                except Exception as e:
                    log.warning(f"Failed to read {txt_file}: {e}")

    log.info(f"Loaded {len(sources)} source files from {raw_dir}")
    return sources


def write_jsonl(
    records: list[dict],
    output_path: pathlib.Path,
    fresh: bool = False,
) -> int:
    """Write records to JSONL. Appends unless fresh=True. Returns count written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if fresh else "a"
    written = 0
    with output_path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def compute_stats(records: list[dict]) -> dict:
    if not records:
        return {"total_records": 0}

    word_counts = [r["metadata"].get("word_count", 0) for r in records]
    by_source: dict[str, int] = {}
    by_language: dict[str, int] = {}
    for r in records:
        sid = r["metadata"].get("source_id", "unknown")
        by_source[sid] = by_source.get(sid, 0) + 1
        lang = r["metadata"].get("language", "en")
        by_language[lang] = by_language.get(lang, 0) + 1

    return {
        "total_records": len(records),
        "total_words": sum(word_counts),
        "avg_output_words": round(sum(word_counts) / len(word_counts), 1),
        "min_output_words": min(word_counts),
        "max_output_words": max(word_counts),
        "by_source": by_source,
        "by_language": by_language,
    }

# ---------------------------------------------------------------------------
# MAIN BUILD LOGIC
# ---------------------------------------------------------------------------

def build_records_from_source(
    source: dict,
    dataset_type: str,
    min_words: int,
    global_seen: Optional[set] = None,
) -> list[dict]:
    """Process one raw source file into a list of JSONL records.

    global_seen: shared set of MD5 hashes across all sources for cross-source
    exact deduplication. Pass the same set instance when processing multiple
    sources to avoid identical chunks appearing more than once in the JSONL.
    Within-source near-duplicate removal (Jaccard) still runs first.
    """
    text = source["text"]
    metadata = source["metadata"]
    language = metadata.get("language", "en")

    # Use a deterministic RNG per source for reproducible template selection
    seed_str = metadata.get("source_id", "") + metadata.get("url", "")
    rng = random.Random(hashlib.md5(seed_str.encode()).hexdigest())

    chunks = split_into_chunks(text, min_words=min_words, max_words=500)
    chunks = [c for c in chunks if is_quality_chunk(c, language)]
    # Within-source near-duplicate removal (Jaccard similarity)
    chunks = deduplicate_chunks(chunks)

    # Cross-source exact deduplication via MD5 hash
    if global_seen is not None:
        deduped = []
        for chunk in chunks:
            h = hashlib.md5(chunk.strip().encode()).hexdigest()
            if h not in global_seen:
                global_seen.add(h)
                deduped.append(chunk)
        chunks = deduped

    records = []
    prev_chunk: Optional[str] = None
    for i, chunk in enumerate(chunks):
        if dataset_type == "rpg":
            record = build_rpg_record(chunk, i, metadata, rng)
        else:
            record = build_literature_record(chunk, i, metadata, prev_chunk, rng)
        records.append(record)
        prev_chunk = chunk

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fine-tuning JSONL datasets from scraped raw text."
    )
    parser.add_argument(
        "--dataset", choices=["rpg", "literature"],
        help="Which dataset to build (default: both)."
    )
    parser.add_argument(
        "--category",
        help="Only process sources from this category."
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Overwrite output files instead of appending."
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print statistics only, do not write files."
    )
    parser.add_argument(
        "--min-words", type=int, default=80,
        help="Minimum words per chunk (default: 80)."
    )
    parser.add_argument(
        "--raw-dir", default="data/raw",
        help="Path to raw data directory (default: data/raw)."
    )
    parser.add_argument(
        "--output-dir", default="data/finetune",
        help="Output directory for JSONL files (default: data/finetune)."
    )
    args = parser.parse_args()

    raw_dir = pathlib.Path(args.raw_dir)
    output_dir = pathlib.Path(args.output_dir)

    # Determine which datasets to build and which source categories to use
    datasets_to_build = (
        [args.dataset] if args.dataset
        else ["rpg", "literature"]
    )

    # Category routing:
    # trpg sources -> rpg dataset
    # webnovel sources -> literature dataset
    # If --category is specified, only load those sources (and only build matching dataset)
    category_filter = args.category

    # Load sources
    trpg_sources = []
    webnovel_sources = []

    if category_filter:
        if category_filter == "webnovel":
            webnovel_sources = load_raw_sources(raw_dir, category="webnovel")
        else:
            # Any other category (trpg, extra_lore, etc.) maps to the RPG dataset
            trpg_sources = load_raw_sources(raw_dir, category=category_filter)
    else:
        # Load everything
        if raw_dir.exists():
            for cat_path in raw_dir.iterdir():
                if cat_path.is_dir():
                    if cat_path.name == "webnovel":
                        webnovel_sources.extend(load_raw_sources(raw_dir, category="webnovel"))
                    else:
                        trpg_sources.extend(load_raw_sources(raw_dir, category=cat_path.name))

    rpg_records: list[dict] = []
    literature_records: list[dict] = []

    # Shared set for cross-source exact deduplication (MD5 hashes)
    rpg_global_seen: set = set()
    lit_global_seen: set = set()

    if "rpg" in datasets_to_build:
        log.info(f"Building RPG dataset from {len(trpg_sources)} TRPG sources...")
        for source in trpg_sources:
            records = build_records_from_source(source, "rpg", args.min_words, rpg_global_seen)
            rpg_records.extend(records)
            log.info(
                f"  {source['metadata'].get('source_id','')} "
                f"{source['metadata'].get('url','')} "
                f"-> {len(records)} records"
            )

    if "literature" in datasets_to_build:
        log.info(f"Building literature dataset from {len(webnovel_sources)} webnovel sources...")
        for source in webnovel_sources:
            records = build_records_from_source(source, "literature", args.min_words, lit_global_seen)
            literature_records.extend(records)
            log.info(
                f"  {source['metadata'].get('source_id','')} "
                f"{source['metadata'].get('url','')} "
                f"-> {len(records)} records"
            )

    # Stats
    if rpg_records:
        stats = compute_stats(rpg_records)
        log.info(f"RPG dataset stats: {json.dumps(stats, ensure_ascii=False)}")
    if literature_records:
        stats = compute_stats(literature_records)
        log.info(f"Literature dataset stats: {json.dumps(stats, ensure_ascii=False)}")

    if args.stats:
        log.info("--stats mode: no files written.")
        return

    # Write outputs
    if rpg_records:
        out_path = output_dir / "rpg_dataset.jsonl"
        n = write_jsonl(rpg_records, out_path, fresh=args.fresh)
        log.info(f"Wrote {n} records -> {out_path}")
    else:
        log.info("No RPG records to write.")

    if literature_records:
        out_path = output_dir / "literature_dataset.jsonl"
        n = write_jsonl(literature_records, out_path, fresh=args.fresh)
        log.info(f"Wrote {n} records -> {out_path}")
    else:
        log.info("No literature records to write.")

    log.info("Done.")


if __name__ == "__main__":
    main()
