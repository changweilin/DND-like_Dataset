"""
convert_to_sharegpt.py — Convert raw scraped text to ShareGPT (JSONL) format
for multi-LoRA fine-tuning as described in lora_training_guide.md.

Outputs three JSONL files under data/finetune/sharegpt/:

  lora_storyteller.jsonl  — Task A: 關鍵字故事接龍生成
  lora_analyst.jsonl      — Task D+E merged: NER 人名/組織抓取 + 情緒判讀
  lora_translator.jsonl   — Task B: 翻譯 (multilingual lore sources)

ShareGPT record format:
  {"conversations": [
    {"from": "system", "value": "..."},
    {"from": "human", "value": "..."},
    {"from": "gpt",   "value": "..."}
  ]}

Usage:
    python convert_to_sharegpt.py                 # convert all tasks
    python convert_to_sharegpt.py --task storyteller
    python convert_to_sharegpt.py --task analyst
    python convert_to_sharegpt.py --task translator
    python convert_to_sharegpt.py --fresh         # overwrite instead of append
    python convert_to_sharegpt.py --stats         # show stats only, no write
    python convert_to_sharegpt.py --min-words 80
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pathlib
import random
import re
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("convert_to_sharegpt")

# ---------------------------------------------------------------------------
# CATEGORY ROUTING
# ---------------------------------------------------------------------------

# Categories that map to the "storyteller" LoRA
STORYTELLER_CATEGORIES = {"webnovel", "extra_lore", "fan_creation", "trpg", "trpg_extra"}
# Categories that supply translation examples
TRANSLATOR_CATEGORIES = {"multilingual_lore"}
# ALL categories contribute to the analyst LoRA
ANALYST_CATEGORIES = STORYTELLER_CATEGORIES | TRANSLATOR_CATEGORIES

# Genre labels derived from tags (same mapping as build_dataset.py)
_GENRE_MAP: dict[str, str] = {
    "litrpg":      "LitRPG",
    "progression": "Progression Fantasy",
    "isekai":      "Isekai 異世界",
    "fantasy":     "奇幻 (Fantasy)",
    "fanfic":      "Fan Fiction",
    "original":    "Original Fantasy",
    "ja":          "日式輕小說 (Light Novel)",
    "en":          "English Web Fiction",
    "horror":      "Dark Horror",
    "scifi":       "Science Fiction",
    "cyberpunk":   "Cyberpunk",
}

_LANGUAGE_NAME: dict[str, str] = {
    "zh": "繁體中文 (Traditional Chinese)",
    "ja": "日語 (Japanese)",
    "es": "西班牙語 (Spanish)",
    "fr": "法語 (French)",
    "en": "英語 (English)",
}

# ---------------------------------------------------------------------------
# SYSTEM PROMPTS
# ---------------------------------------------------------------------------

_STORYTELLER_SYSTEM_EN = (
    "You are a creative narrative writer specializing in {genre} fiction. "
    "Given a story excerpt or opening line, continue the narrative with "
    "vivid prose that matches the existing tone and style."
)

_STORYTELLER_SYSTEM_JA = (
    "あなたは{genre}の熟練した物語作家です。"
    "与えられた文章や書き出しに続けて、既存のトーンやスタイルに合わせた"
    "生き生きとした散文で物語を続けてください。"
)

_NER_SYSTEM = (
    "你是一個遊戲後台的實體抓取器 (NER)。"
    "請分析以下文本，精準提取所有角色名稱與組織名稱，並以 JSON 格式輸出。"
    "輸出格式：{\"角色\": [\"...\"], \"組織\": [\"...\"]}"
)

_SENTIMENT_SYSTEM = (
    "你是一個情緒判讀分析器。"
    "請閱讀以下文本段落，判斷其主要情緒基調，並從下列標籤中選擇最符合者輸出："
    "「正面 (Positive)」、「負面/緊張 (Negative/Tense)」、「中性/描述 (Neutral)」、「動作/衝突 (Action)」。"
    "只需輸出標籤名稱，不需額外說明。"
)

_TRANSLATOR_SYSTEM = (
    "你是一個精通多語言的遊戲世界觀翻譯專家。"
    "請將以下{src_lang}的奇幻/遊戲世界觀文本翻譯成{tgt_lang}，"
    "保留專有名詞的原始風格與語氣。"
)

# ---------------------------------------------------------------------------
# TEXT CHUNKING  (same algorithm as build_dataset.py)
# ---------------------------------------------------------------------------

def split_into_chunks(text: str, min_words: int = 80, max_words: int = 500) -> list[str]:
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
            if current:
                merged.append(current)
                current = ""
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
# QUALITY FILTER
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\bcookies?\b.{0,40}\baccept\b",
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


def _detect_lang_mismatch(text: str, expected: str) -> bool:
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return False
    total = len(letters)
    ascii_l = sum(1 for c in letters if ord(c) < 128)
    cjk = sum(1 for c in letters if "\u4E00" <= c <= "\u9FFF" or "\u3400" <= c <= "\u4DBF")
    kana = sum(1 for c in letters if "\u3040" <= c <= "\u30FF")
    if expected == "en" and ascii_l / total < 0.70:
        return True
    if expected == "ja" and (cjk + kana) / total < 0.30:
        return True
    return False


def is_quality_chunk(text: str, language: str = "en") -> bool:
    words = text.split()
    if len(words) < 80:
        return False
    lines = [l for l in text.splitlines() if l.strip()]
    if lines and sum(1 for l in lines if len(l.split()) == 1) / len(lines) > 0.30:
        return False
    if not re.search(r"[a-zA-Z\u3040-\u30FF\u4E00-\u9FFF]", text):
        return False
    if language == "en":
        avg = sum(len(w) for w in words) / len(words)
        if not (3.0 <= avg <= 12.0):
            return False
    if language == "ja":
        jp = sum(1 for c in text if "\u3040" <= c <= "\u30FF")
        if jp / len(text) < 0.10:
            return False
    if _detect_lang_mismatch(text, language):
        return False
    if sum(1 for p in _BOILERPLATE_PATTERNS if p.search(text)) >= 2:
        return False
    return True


def deduplicate_chunks(chunks: list[str], threshold: float = 0.85) -> list[str]:
    seen: list[set] = []
    result: list[str] = []
    for chunk in chunks:
        words = set(chunk.lower().split())
        dup = any(
            (len(words & p) / len(words | p) >= threshold)
            for p in seen if words and p
        )
        if not dup:
            result.append(chunk)
            seen.append(words)
    return result


# ---------------------------------------------------------------------------
# NER HEURISTICS
# ---------------------------------------------------------------------------

# Patterns to skip common false-positive capitalized words
_SKIP_WORDS = {
    "The", "A", "An", "In", "On", "At", "But", "And", "Or", "Nor", "So",
    "For", "Yet", "With", "By", "From", "To", "Of", "As", "Up", "Out",
    "After", "Before", "During", "When", "While", "Although", "However",
    "Chapter", "Book", "Part", "Volume", "Section", "He", "She", "It",
    "They", "We", "You", "I", "His", "Her", "Their", "Its", "Our",
    "This", "That", "These", "Those", "There", "Here", "Now", "Then",
    "What", "Which", "Who", "Whom", "Where", "How", "Why",
}

_ORG_SUFFIXES = re.compile(
    r"\b(guild|order|clan|company|empire|kingdom|republic|federation|"
    r"legion|brotherhood|sisterhood|council|court|house|alliance|"
    r"association|academy|institute|college|corps|squad|regiment|"
    r"brigade|army|navy|fleet|bureau|ministry|department|church|"
    r"temple|cult|sect|faction|tribe|band|party|group)\b",
    re.IGNORECASE,
)


def extract_entities_heuristic(text: str) -> dict:
    """
    Heuristically extract character names and organization names from text.
    Returns {"角色": [...], "組織": [...]}.
    """
    # --- Character names: 1-3 consecutive Title-Case words not in skip list ---
    char_pattern = re.compile(
        r"\b([A-Z][a-z]{1,20})(?:\s+[A-Z][a-z]{1,20}){0,2}\b"
    )
    characters: set[str] = set()
    for m in char_pattern.finditer(text):
        name = m.group(0)
        parts = name.split()
        # Skip if all parts are common words
        if all(p in _SKIP_WORDS for p in parts):
            continue
        # Skip single-word entries that are very common words
        if len(parts) == 1 and parts[0] in _SKIP_WORDS:
            continue
        # Require at least one part NOT in skip list
        if any(p not in _SKIP_WORDS for p in parts):
            characters.add(name)

    # --- Organization names: "the XYZ <suffix>" pattern ---
    orgs: set[str] = set()
    org_pattern = re.compile(
        r"the\s+([A-Z][a-zA-Z\s]{2,40}?)\s*(?:" + _ORG_SUFFIXES.pattern + r")",
        re.IGNORECASE,
    )
    for m in org_pattern.finditer(text):
        full = m.group(0)
        # Capitalise first letter of each word for consistency
        orgs.add(" ".join(w.capitalize() for w in full.split()))

    # Also grab explicit "X Order", "X Empire" style without "the"
    bare_org = re.compile(
        r"\b([A-Z][a-z]{2,20}(?:\s+[A-Z][a-z]{2,20}){0,2})\s+"
        r"(Guild|Order|Clan|Empire|Kingdom|Legion|Brotherhood|Council|"
        r"Alliance|Academy|Corps|Bureau|Church|Temple|Cult|Faction|Tribe)\b"
    )
    for m in bare_org.finditer(text):
        orgs.add(m.group(0))

    # Remove characters that are substrings of organisations
    filtered_chars = [
        c for c in sorted(characters)
        if not any(c in org for org in orgs)
    ]

    return {
        "角色": filtered_chars[:15],  # cap to avoid bloat
        "組織": sorted(orgs)[:10],
    }


# ---------------------------------------------------------------------------
# SENTIMENT HEURISTICS
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = {
    "joy", "triumph", "victory", "hope", "laugh", "smile", "happy",
    "celebrate", "cheer", "brave", "courage", "glory", "honour",
    "peace", "love", "warm", "light", "bright", "radiant",
}
_NEGATIVE_WORDS = {
    "fear", "dread", "despair", "shadow", "dark", "death", "blood",
    "pain", "agony", "grief", "sorrow", "weep", "cry", "horror",
    "terror", "hate", "anger", "rage", "corrupt", "evil", "doom",
}
_ACTION_WORDS = {
    "battle", "fight", "attack", "strike", "charge", "clash", "rush",
    "dash", "leap", "dodge", "slash", "thrust", "parry", "ambush",
    "siege", "combat", "war", "duel", "chase", "flee", "explode",
}


def classify_sentiment(text: str) -> str:
    words = re.findall(r"\b[a-z]+\b", text.lower())
    word_set = set(words)
    pos = len(word_set & _POSITIVE_WORDS)
    neg = len(word_set & _NEGATIVE_WORDS)
    act = len(word_set & _ACTION_WORDS)

    if act > pos and act > neg:
        return "動作/衝突 (Action)"
    if neg > pos:
        return "負面/緊張 (Negative/Tense)"
    if pos > neg:
        return "正面 (Positive)"
    return "中性/描述 (Neutral)"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_raw_sources(
    raw_dir: pathlib.Path,
    category_filter: Optional[str] = None,
) -> list[dict]:
    """Walk data/raw/, return list of {text, metadata} dicts."""
    sources = []
    if not raw_dir.exists():
        log.warning("Raw dir not found: %s", raw_dir)
        return sources

    for cat_dir in sorted(raw_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        if category_filter and cat_dir.name != category_filter:
            continue
        for source_dir in sorted(cat_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            for txt_file in sorted(source_dir.glob("*.txt")):
                json_file = txt_file.with_suffix(".json")
                if not json_file.exists():
                    continue
                try:
                    text = txt_file.read_text(encoding="utf-8")
                    meta = json.loads(json_file.read_text(encoding="utf-8"))
                    if text.strip():
                        sources.append({"text": text, "metadata": meta})
                except Exception as e:
                    log.warning("Failed to read %s: %s", txt_file, e)

    log.info("Loaded %d source files from %s (filter=%s)", len(sources), raw_dir, category_filter)
    return sources


def write_jsonl(records: list[dict], path: pathlib.Path, fresh: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if fresh else "a"
    written = 0
    with path.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    return written


def sharegpt(system: str, human: str, gpt: str) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": system},
            {"from": "human",  "value": human},
            {"from": "gpt",    "value": gpt},
        ]
    }


# ---------------------------------------------------------------------------
# TASK A — STORYTELLER (Story Continuation)
# ---------------------------------------------------------------------------

def _genre_label(tags: list[str]) -> str:
    for t in tags:
        if t in _GENRE_MAP:
            return _GENRE_MAP[t]
    return "奇幻 (Fantasy)"


_CONTINUATION_PROMPTS_EN = [
    "Continue the story from where it left off:",
    "The next passage begins here:",
    "Continue this narrative:",
    "Write the continuation of the following excerpt:",
    "What happens next in the story?",
]

_CONTINUATION_PROMPTS_JA = [
    "次の文章を続けてください：",
    "物語の続きを書いてください：",
    "この場面の続きを記してください：",
]


def build_storyteller_records(
    sources: list[dict],
    min_words: int,
    global_seen: set,
    rng: random.Random,
) -> list[dict]:
    """Produce story-continuation ShareGPT records from narrative sources."""
    records: list[dict] = []

    for src in sources:
        meta = src["metadata"]
        lang = meta.get("language", "en")
        tags = meta.get("tags", [])
        genre = _genre_label(tags)

        chunks = split_into_chunks(src["text"], min_words=min_words)
        chunks = [c for c in chunks if is_quality_chunk(c, lang)]
        chunks = deduplicate_chunks(chunks)

        # Cross-source exact deduplication
        deduped = []
        for c in chunks:
            h = hashlib.md5(c.strip().encode()).hexdigest()
            if h not in global_seen:
                global_seen.add(h)
                deduped.append(c)
        chunks = deduped

        # Build continuation pairs: chunk[i] → chunk[i+1]
        for i in range(len(chunks) - 1):
            prompt_chunk = chunks[i]
            continuation = chunks[i + 1]

            if lang == "ja":
                system = _STORYTELLER_SYSTEM_JA.format(genre=genre)
                connector = rng.choice(_CONTINUATION_PROMPTS_JA)
            else:
                system = _STORYTELLER_SYSTEM_EN.format(genre=genre)
                connector = rng.choice(_CONTINUATION_PROMPTS_EN)

            # Human turn: last 1-2 sentences of the context chunk as the prompt
            sents = re.split(r"(?<=[.!?。！？])\s+", prompt_chunk.strip())
            prompt_text = " ".join(sents[-2:]).strip()
            if len(prompt_text.split()) < 10:
                prompt_text = prompt_chunk[-300:].strip()

            human_turn = f"{connector}\n\n{prompt_text}"
            records.append(sharegpt(system, human_turn, continuation))

    return records


# ---------------------------------------------------------------------------
# TASK D+E — ANALYST (NER + Sentiment, merged)
# ---------------------------------------------------------------------------

def build_analyst_records(
    sources: list[dict],
    min_words: int,
    global_seen: set,
    rng: random.Random,
) -> list[dict]:
    """Produce NER and Sentiment ShareGPT records from all narrative sources."""
    records: list[dict] = []

    for src in sources:
        meta = src["metadata"]
        lang = meta.get("language", "en")

        # Analyst training is most effective on English text (NER patterns are EN)
        # For non-English sources, generate sentiment-only records
        chunks = split_into_chunks(src["text"], min_words=min_words)
        chunks = [c for c in chunks if is_quality_chunk(c, lang)]
        chunks = deduplicate_chunks(chunks)

        deduped = []
        for c in chunks:
            h = hashlib.md5(("analyst:" + c.strip()).encode()).hexdigest()
            if h not in global_seen:
                global_seen.add(h)
                deduped.append(c)
        chunks = deduped

        for chunk in chunks:
            # Alternate between NER and Sentiment, weighted 60/40
            roll = rng.random()
            if roll < 0.60 and lang == "en":
                # NER record
                entities = extract_entities_heuristic(chunk)
                # Only keep if we found at least one entity
                if not entities["角色"] and not entities["組織"]:
                    # Fallback: emit a sentiment record instead
                    label = classify_sentiment(chunk)
                    records.append(sharegpt(_SENTIMENT_SYSTEM, chunk, label))
                    continue
                gpt_answer = json.dumps(entities, ensure_ascii=False)
                records.append(sharegpt(_NER_SYSTEM, chunk, gpt_answer))
            else:
                # Sentiment record
                label = classify_sentiment(chunk)
                records.append(sharegpt(_SENTIMENT_SYSTEM, chunk, label))

    return records


# ---------------------------------------------------------------------------
# TASK B — TRANSLATOR
# ---------------------------------------------------------------------------

_TRANSLATE_TO_EN_HINT = [
    "將以下文本翻譯成流暢的英語，保留奇幻風格與專有名詞：",
    "請把以下段落翻譯成英文，維持原有的世界觀語氣：",
]

_TRANSLATE_FROM_EN_HINT = [
    "請將以下英語文本翻譯成{lang}，保留奇幻風格：",
    "把以下英語段落翻譯成{lang}，維持世界觀風格：",
]


def build_translator_records(
    sources: list[dict],
    min_words: int,
    global_seen: set,
    rng: random.Random,
) -> list[dict]:
    """
    Produce translation ShareGPT records from multilingual sources.

    Strategy: for each non-English chunk, create a record that asks the model
    to translate it INTO English (we use the chunk itself as the target output,
    since these are lore texts — not true bilingual pairs).

    Additionally, for each English chunk found in the same source batch,
    create records asking to translate INTO the target language present in
    the dataset.
    """
    records: list[dict] = []

    # Separate English sources from non-English sources
    en_sources = [s for s in sources if s["metadata"].get("language", "en") == "en"]
    foreign_sources = [s for s in sources if s["metadata"].get("language", "en") != "en"]

    # --- Foreign → English records ---
    for src in foreign_sources:
        meta = src["metadata"]
        lang = meta.get("language", "?")
        src_lang_name = _LANGUAGE_NAME.get(lang, lang)
        tgt_lang_name = _LANGUAGE_NAME["en"]

        chunks = split_into_chunks(src["text"], min_words=min_words)
        chunks = [c for c in chunks if is_quality_chunk(c, lang)]
        chunks = deduplicate_chunks(chunks)

        for chunk in chunks:
            h = hashlib.md5(("trans:" + chunk.strip()).encode()).hexdigest()
            if h in global_seen:
                continue
            global_seen.add(h)

            system = _TRANSLATOR_SYSTEM.format(
                src_lang=src_lang_name,
                tgt_lang=tgt_lang_name,
            )
            hint = rng.choice(_TRANSLATE_TO_EN_HINT)
            human_turn = f"{hint}\n\n{chunk}"

            # NOTE: Ground truth would require a human translator.
            # We emit a placeholder marker so these records can be filtered
            # or post-processed. During actual training, replace <TRANSLATE>
            # with a real translation.
            records.append(sharegpt(system, human_turn, f"<TRANSLATE>\n{chunk}"))

    # --- English → Foreign records (using foreign source lang as target) ---
    # Collect available non-English languages from foreign sources
    available_langs = list({s["metadata"].get("language") for s in foreign_sources} - {"en", None})

    if en_sources and available_langs:
        for src in en_sources:
            meta = src["metadata"]
            chunks = split_into_chunks(src["text"], min_words=min_words)
            chunks = [c for c in chunks if is_quality_chunk(c, "en")]
            chunks = deduplicate_chunks(chunks)

            for chunk in chunks:
                h = hashlib.md5(("trans_en:" + chunk.strip()).encode()).hexdigest()
                if h in global_seen:
                    continue
                global_seen.add(h)

                tgt_lang = rng.choice(available_langs)
                tgt_lang_name = _LANGUAGE_NAME.get(tgt_lang, tgt_lang)
                system = _TRANSLATOR_SYSTEM.format(
                    src_lang=_LANGUAGE_NAME["en"],
                    tgt_lang=tgt_lang_name,
                )
                hint = rng.choice(_TRANSLATE_FROM_EN_HINT).format(lang=tgt_lang_name)
                human_turn = f"{hint}\n\n{chunk}"
                records.append(sharegpt(system, human_turn, f"<TRANSLATE>\n{chunk}"))

    return records


# ---------------------------------------------------------------------------
# STATS
# ---------------------------------------------------------------------------

def print_stats(name: str, records: list[dict]) -> None:
    if not records:
        log.info("[%s] No records generated.", name)
        return

    # Token estimate: ~1.3 tokens per word on average
    word_counts = []
    for rec in records:
        total_words = sum(
            len(turn["value"].split())
            for turn in rec["conversations"]
        )
        word_counts.append(total_words)

    total = len(records)
    avg_w = sum(word_counts) / total
    log.info(
        "[%s] %d records | avg %.0f words/record | est. %.0f tokens/record | "
        "total est. %.1fM tokens",
        name, total, avg_w, avg_w * 1.3, total * avg_w * 1.3 / 1_000_000,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw scraped text to ShareGPT JSONL for LoRA training."
    )
    parser.add_argument(
        "--task", choices=["storyteller", "analyst", "translator"],
        help="Which LoRA task to build (default: all).",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Overwrite output files instead of appending.",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print statistics only, do not write files.",
    )
    parser.add_argument(
        "--min-words", type=int, default=80,
        help="Minimum words per chunk (default: 80).",
    )
    parser.add_argument(
        "--raw-dir", default="data/raw",
        help="Path to raw data directory (default: data/raw).",
    )
    parser.add_argument(
        "--output-dir", default="data/finetune/sharegpt",
        help="Output directory for JSONL files (default: data/finetune/sharegpt).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    raw_dir = pathlib.Path(args.raw_dir)
    output_dir = pathlib.Path(args.output_dir)
    rng = random.Random(args.seed)

    tasks = [args.task] if args.task else ["storyteller", "analyst", "translator"]

    # ------------------------------------------------------------------ load
    log.info("Loading raw sources from %s ...", raw_dir)
    all_sources: list[dict] = []
    for cat_dir in sorted(raw_dir.iterdir()):
        if cat_dir.is_dir():
            all_sources.extend(load_raw_sources(raw_dir, category_filter=cat_dir.name))

    story_sources = [
        s for s in all_sources
        if s["metadata"].get("category", "") in STORYTELLER_CATEGORIES
    ]
    analyst_sources = [
        s for s in all_sources
        if s["metadata"].get("category", "") in ANALYST_CATEGORIES
    ]
    translator_sources = [
        s for s in all_sources
        if s["metadata"].get("category", "") in TRANSLATOR_CATEGORIES
    ]
    # Add a small sample of English lore sources for EN→foreign translation records
    en_lore_for_trans = [
        s for s in story_sources
        if s["metadata"].get("language", "en") == "en"
    ]
    translator_sources_full = translator_sources + en_lore_for_trans

    log.info(
        "Sources: storyteller=%d, analyst=%d, translator=%d",
        len(story_sources), len(analyst_sources), len(translator_sources_full),
    )

    # ----------------------------------------------------------------- build
    results: dict[str, list[dict]] = {}
    global_seen_story: set = set()
    global_seen_analyst: set = set()
    global_seen_trans: set = set()

    if "storyteller" in tasks:
        log.info("Building lora_storyteller records ...")
        records = build_storyteller_records(
            story_sources, args.min_words, global_seen_story, rng
        )
        results["storyteller"] = records
        print_stats("lora_storyteller", records)

    if "analyst" in tasks:
        log.info("Building lora_analyst records ...")
        records = build_analyst_records(
            analyst_sources, args.min_words, global_seen_analyst, rng
        )
        results["analyst"] = records
        print_stats("lora_analyst", records)

    if "translator" in tasks:
        log.info("Building lora_translator records ...")
        records = build_translator_records(
            translator_sources_full, args.min_words, global_seen_trans, rng
        )
        results["translator"] = records
        print_stats("lora_translator", records)

    if args.stats:
        log.info("--stats mode: no files written.")
        return

    # ----------------------------------------------------------------- write
    output_map = {
        "storyteller": output_dir / "lora_storyteller.jsonl",
        "analyst":     output_dir / "lora_analyst.jsonl",
        "translator":  output_dir / "lora_translator.jsonl",
    }

    for task_name, records in results.items():
        if not records:
            log.warning("[%s] No records to write, skipping.", task_name)
            continue
        out_path = output_map[task_name]
        n = write_jsonl(records, out_path, fresh=args.fresh)
        log.info("Wrote %d records -> %s", n, out_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
