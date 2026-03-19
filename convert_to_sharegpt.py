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

# Words that are never valid as (part of) a character name.
# Keep this list generous — false positives hurt training quality more than
# false negatives (fewer but accurate entities > many noisy ones).
_SKIP_WORDS = {
    # Articles / prepositions / conjunctions
    "The", "A", "An", "In", "On", "At", "But", "And", "Or", "Nor", "So",
    "For", "Yet", "With", "By", "From", "To", "Of", "As", "Up", "Out",
    "After", "Before", "During", "When", "While", "Although", "However",
    "Into", "Onto", "Upon", "About", "Against", "Between", "Through",
    "Over", "Under", "Around", "Within", "Without", "Beyond", "Across",
    # Pronouns
    "He", "She", "It", "They", "We", "You", "I",
    "His", "Her", "Their", "Its", "Our", "My", "Your",
    "Him", "Them", "Us", "Me",
    "Himself", "Herself", "Themselves", "Itself", "Yourself",
    # Demonstratives / relative
    "This", "That", "These", "Those", "There", "Here",
    "Now", "Then", "What", "Which", "Who", "Whom", "Where", "How", "Why",
    # Structure words
    "Also", "Both", "Each", "Every", "All", "Some", "Any", "No",
    "More", "Most", "Much", "Many", "Few", "Little", "Other", "Another",
    "Same", "Such", "Even", "Only", "Just", "Still", "Already",
    "Very", "Quite", "Rather", "Almost", "Often", "Never", "Always",
    # Publication / game structure terms  (cause "Chapter Four", "Campaign Three" etc.)
    "Chapter", "Book", "Part", "Volume", "Section", "Episode",
    "Campaign", "Season", "Series", "Story", "Saga", "Arc",
    "Act", "Scene", "Page", "Line",
    # Time / measurement
    "Age", "Era", "Year", "Years", "Day", "Days", "Night", "Nights",
    "Time", "Times", "Hour", "Hours", "Week", "Month", "Century",
    # Generic world nouns (cause "Dark Forest", "Ancient Temple" etc.)
    "World", "Land", "Lands", "City", "Town", "Village",
    "Kingdom", "Empire", "Realm", "Region", "Area", "Place", "Point",
    "North", "South", "East", "West", "Central",
    "New", "Old", "Great", "Grand", "Ancient", "Dark", "Light",
    "High", "Low", "Long", "Short", "Far", "Near", "Deep", "True",
    "First", "Last", "Next", "Previous",
    # Common adjectives that are capitalised at sentence start
    "Many", "Most", "Both", "Several", "Various", "Certain", "Entire",
}

# Cardinal and ordinal number words — names containing these are almost never
# character names (e.g. "Campaign Four", "Age One", "Season Two").
_NUMBER_WORDS = {
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Twenty", "Thirty",
    "Hundred", "Thousand",
    "First", "Second", "Third", "Fourth", "Fifth", "Sixth",
    "Seventh", "Eighth", "Ninth", "Tenth",
    "Once", "Twice",
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

    Quality improvements over naive capitalisation matching:
    - Expands _SKIP_WORDS to cover publication terms, number words, time words
    - Filters names that contain any cardinal/ordinal number word
    - Requires single-word names to appear ≥ 2 times in the chunk (reduces
      sentence-initial false positives)
    - Requires single-word names to be ≥ 4 characters
    - Multi-word names require every non-skip word to be ≥ 3 characters
    """
    # Pre-count all Title-Case tokens for frequency filtering
    all_caps_tokens = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    token_freq: dict[str, int] = {}
    for t in all_caps_tokens:
        token_freq[t] = token_freq.get(t, 0) + 1

    # --- Character names: 1-3 consecutive Title-Case words ---
    char_pattern = re.compile(
        r"\b([A-Z][a-z]{1,20})(?:\s+[A-Z][a-z]{1,20}){0,2}\b"
    )
    characters: set[str] = set()
    for m in char_pattern.finditer(text):
        name = m.group(0)
        parts = name.split()

        # Reject if any part is a skip word or number word
        if any(p in _SKIP_WORDS for p in parts):
            continue
        if any(p in _NUMBER_WORDS for p in parts):
            continue

        # Single-word name: must be ≥ 4 chars AND appear ≥ 2 times
        if len(parts) == 1:
            if len(parts[0]) < 4:
                continue
            if token_freq.get(parts[0], 0) < 2:
                continue

        # Multi-word name: every non-skip word must be ≥ 3 chars
        if len(parts) > 1:
            if any(len(p) < 3 for p in parts):
                continue

        characters.add(name)

    # --- Organization names: "the XYZ <suffix>" pattern ---
    orgs: set[str] = set()
    org_pattern = re.compile(
        r"the\s+([A-Z][a-zA-Z\s]{2,40}?)\s*(?:" + _ORG_SUFFIXES.pattern + r")",
        re.IGNORECASE,
    )
    for m in org_pattern.finditer(text):
        full = m.group(0)
        orgs.add(" ".join(w.capitalize() for w in full.split()))

    # Explicit "X Order / X Empire" without "the"
    bare_org = re.compile(
        r"\b([A-Z][a-z]{2,20}(?:\s+[A-Z][a-z]{2,20}){0,2})\s+"
        r"(Guild|Order|Clan|Empire|Kingdom|Legion|Brotherhood|Council|"
        r"Alliance|Academy|Corps|Bureau|Church|Temple|Cult|Faction|Tribe)\b"
    )
    for m in bare_org.finditer(text):
        name = m.group(0)
        parts = name.split()[:-1]  # exclude the suffix word itself for number check
        if not any(p in _NUMBER_WORDS for p in parts):
            orgs.add(name)

    # Remove character names that are substrings of organisation names
    filtered_chars = [
        c for c in sorted(characters)
        if not any(c in org for org in orgs)
    ]

    return {
        "角色": filtered_chars[:10],
        "組織": sorted(orgs)[:8],
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

        # max_words=400 for output chunks: ~520 tokens output + ~80 system/human
        # overhead ≈ 600 tokens total, safely within max_seq_len=1024.
        chunks = split_into_chunks(src["text"], min_words=min_words, max_words=400)
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
        # For non-English sources, generate sentiment-only records.
        # max_words=500: GPT output is a small JSON/label (<80 tokens), so the
        # full 500-word input chunk still keeps total sequence < 1024 tokens.
        chunks = split_into_chunks(src["text"], min_words=min_words, max_words=500)
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

        # max_words=200: the source chunk appears in BOTH human and gpt turns,
        # so total tokens ≈ 2 × (200 × 1.3) + overhead ≈ 580 tokens — within 1024.
        chunks = split_into_chunks(src["text"], min_words=min_words, max_words=200)
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
            chunks = split_into_chunks(src["text"], min_words=min_words, max_words=200)
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
# TASK C — REASONING (語意推理)
# ---------------------------------------------------------------------------

_REASONING_SYSTEM = (
    "你是遊戲後台的語意推理引擎（Semantic Reasoning Engine）。"
    "根據玩家行動與當前遊戲狀態，透過逐步邏輯推導得出結論——"
    "包括任務觸發、NPC好感度變化、隱藏事件解鎖或陣營關係變動。"
    "請嚴格按照以下格式輸出：\n"
    "【推理步驟】\n（逐條列出推理過程與計算）\n\n【結論】\n（JSON格式）"
)

# Fallback entity pools when lore extraction yields too few names
_DEFAULT_NPCS = [
    "艾莉絲", "雷納德", "米拉", "塔利恩", "希薇亞", "卡索夫", "蓮恩", "奧格里姆",
    "艾琳", "費南多", "盧西亞", "古倫", "薩菲娜", "德拉克", "伊芙",
    "Aria", "Roland", "Mira", "Thane", "Lyria", "Kael", "Elara",
    "Gorin", "Sylvia", "Draven", "Nira", "Aldric", "Seraphina",
]
_DEFAULT_FACTIONS = [
    "黎明議會", "暗夜刺客公會", "鐵拳傭兵團", "星月商會", "守護者聯盟",
    "古老秩序", "自由市民聯盟", "龍骨騎士團", "Silver Hand", "Iron Circle",
    "Dawn's Vanguard", "The Crimson Order", "Ember Syndicate",
]
_DEFAULT_QUESTS = [
    "星光之誓", "暗影追蹤", "遺失的傳說", "黎明之前", "命運的抉擇",
    "失落的神殿", "古老的誓言", "碎裂的鏡", "深淵邊緣", "最後的守護者",
    "The Fading Ember", "Whispers of the Abyss", "The Sunken Relic",
]
_DEFAULT_LOCATIONS = [
    "星月酒館", "黑鐵鍛造廠", "古老圖書館", "市場廣場", "王宮花園",
    "地下遺跡", "迷霧森林", "海港碼頭", "修道院", "競技場",
    "The Rusted Gate Inn", "Ironhaven Market", "The Obsidian Vault",
]
_DEFAULT_ITEMS = [
    "月光石", "暗影精華", "古代捲軸", "元素晶核", "魔法藥草",
    "神聖符印", "鍛造材料", "稀有礦石", "遠古鑰匙", "星辰碎片",
    "Moonshard Crystal", "Void Essence", "Ancient Sigil",
]


# ---------------------------------------------------------------------------
# Reasoning scenario templates — each is a callable:
#   f(npc, faction, quest, location, item, rng) -> (action, state, steps, conclusion)
# ---------------------------------------------------------------------------

def _tmpl_npc_gift(npc, faction, quest, location, item, rng):
    n = rng.randint(2, 5)
    gift_v = rng.randint(5, 10)
    talk_v = rng.randint(8, 15)
    base = rng.randint(20, 55)
    gain = n * gift_v + talk_v
    new_aff = base + gain
    thr = rng.randint(55, 75)
    triggered = quest if new_aff >= thr else "無"
    delta_str = f"差 {thr - new_aff} 點" if new_aff < thr else "條件達成"
    action = f"玩家向NPC「{npc}」連續贈送 {n} 件{item}，並在互動中選擇傾聽其心聲。"
    state = json.dumps({"「"+npc+"」好感度": base, "任務「"+quest+"」狀態": "未觸發", "地點": location}, ensure_ascii=False)
    steps = (
        f"1. 行動拆解：\n"
        f"   - 贈禮 {n} 次（每次 +{gift_v} 好感）= +{n * gift_v}\n"
        f"   - 傾聽互動（+{talk_v} 好感）\n"
        f"   - 本輪好感增益合計：+{gain}\n"
        f"2. 新好感度：{base} + {gain} = {new_aff}\n"
        f"3. 任務「{quest}」觸發門檻：≥ {thr}\n"
        f"   當前好感 {new_aff} {'≥' if new_aff >= thr else '<'} {thr} → {delta_str}\n"
        f"4. 地點「{location}」滿足互動場景條件。"
    )
    conclusion = json.dumps({
        "好感度增量": gain, "新好感度": new_aff,
        "任務觸發": triggered, "NPC特殊回應": "感激" if new_aff >= thr else "友好",
        "解鎖特殊對話": new_aff >= thr,
    }, ensure_ascii=False)
    return action, state, steps, conclusion


def _tmpl_npc_reject(npc, faction, quest, location, item, rng):
    base = rng.randint(40, 70)
    pen1 = rng.randint(10, 18)
    pen2 = rng.randint(5, 12)
    new_aff = max(0, base - pen1 - pen2)
    thr = rng.randint(30, 45)
    fail_quest = new_aff < thr
    action = f"玩家拒絕了NPC「{npc}」的請求，並在隨後的對話選項中選擇嘲諷。"
    state = json.dumps({"「"+npc+"」好感度": base, "任務「"+quest+"」狀態": "進行中", "地點": location}, ensure_ascii=False)
    steps = (
        f"1. 行動分類：\n"
        f"   - 拒絕請求（-{pen1} 好感，屬於負面社交互動）\n"
        f"   - 嘲諷選項（-{pen2} 好感，屬於敵意言語）\n"
        f"   - 本輪好感懲罰：-{pen1 + pen2}\n"
        f"2. 新好感度：{base} - {pen1 + pen2} = {new_aff}\n"
        f"3. 任務「{quest}」中止條件：好感度 < {thr}\n"
        f"   當前 {new_aff} {'<' if fail_quest else '≥'} {thr} "
        f"→ 任務{'即將失敗，NPC將在下次互動中撤回委託' if fail_quest else '仍可繼續，但需盡快修復關係'}\n"
        f"4. 若好感度降至 0，NPC轉為敵對狀態。"
    )
    conclusion = json.dumps({
        "好感度變化": -(pen1 + pen2), "新好感度": new_aff,
        "任務狀態": "瀕臨失敗" if fail_quest else "警告",
        "NPC態度": "憤怒" if new_aff < thr else "冷淡",
        "需要補救": fail_quest,
    }, ensure_ascii=False)
    return action, state, steps, conclusion


def _tmpl_npc_help(npc, faction, quest, location, item, rng):
    base = rng.randint(30, 60)
    combat_v = rng.randint(10, 20)
    reward_v = rng.randint(5, 10)
    new_aff = base + combat_v + reward_v
    skill_req = rng.randint(3, 6)
    player_skill = rng.randint(4, 8)
    action = f"玩家協助NPC「{npc}」擊退了{faction}的伏擊，並將戰利品中的{item}分給了她。"
    state = json.dumps({"「"+npc+"」好感度": base, "玩家戰鬥技能等級": player_skill, "任務「"+quest+"」狀態": "未觸發"}, ensure_ascii=False)
    steps = (
        f"1. 行動評估：\n"
        f"   - 協助戰鬥（+{combat_v} 好感，高風險護衛行為）\n"
        f"   - 分享戰利品{item}（+{reward_v} 好感，展現信任）\n"
        f"2. 新好感度：{base} + {combat_v + reward_v} = {new_aff}\n"
        f"3. 任務「{quest}」前置條件：戰鬥技能 ≥ {skill_req} ∧ 好感度 ≥ {base + combat_v}\n"
        f"   玩家技能 {player_skill} ≥ {skill_req} ✓，好感 {new_aff} ≥ {base + combat_v} ✓\n"
        f"4. 所有前置條件滿足，任務解鎖。"
    )
    conclusion = json.dumps({
        "好感度增量": combat_v + reward_v, "新好感度": new_aff,
        "任務觸發": quest, "獲得稱號": f"{npc}的守護者",
        "特殊獎勵": f"「{npc}」提供獨家情報",
    }, ensure_ascii=False)
    return action, state, steps, conclusion


def _tmpl_quest_collect(npc, faction, quest, location, item, rng):
    total = rng.randint(5, 10)
    have = rng.randint(2, total - 1)
    need = total - have
    gold = rng.randint(50, 200)
    exp = rng.randint(100, 500)
    action = f"玩家向NPC「{npc}」提交目前收集的 {have} 件{item}，詢問任務進度。"
    state = json.dumps({"任務「"+quest+"」進度": f"{have}/{total}", "玩家金幣": gold, "地點": location}, ensure_ascii=False)
    steps = (
        f"1. 收集進度核查：已有 {have} 件，目標 {total} 件，尚缺 {need} 件。\n"
        f"2. 完成率：{have}/{total} = {have/total*100:.0f}%\n"
        f"3. 任務觸發條件：收集量 = {total}（未達成）\n"
        f"   → 任務繼續進行，提示玩家還需收集 {need} 件{item}。\n"
        f"4. 預計完成獎勵（達成後）：金幣 +{gold}，經驗值 +{exp}，解鎖後續任務。"
    )
    conclusion = json.dumps({
        "任務狀態": "進行中", "當前進度": f"{have}/{total}",
        "剩餘數量": need, "完成率": f"{have/total*100:.0f}%",
        "預計獎勵": {"金幣": gold, "經驗值": exp},
    }, ensure_ascii=False)
    return action, state, steps, conclusion


def _tmpl_quest_explore(npc, faction, quest, location, item, rng):
    visited = rng.randint(2, 4)
    total_locs = rng.randint(visited + 1, visited + 3)
    trigger_loc = location
    exp = rng.randint(200, 800)
    have_clue = rng.choice([True, False])
    action = f"玩家首次進入「{location}」，觸發探索感知。已探索地點 {visited}/{total_locs}。"
    state = json.dumps({"探索任務「"+quest+"」": f"{visited}/{total_locs}", "已持有線索": have_clue}, ensure_ascii=False)
    trigger = visited + 1 >= total_locs or have_clue
    steps = (
        f"1. 地點識別：「{location}」為任務「{quest}」的關鍵地點。\n"
        f"2. 探索進度更新：{visited} → {visited + 1}（共 {total_locs} 個地點）\n"
        f"3. 觸發條件組合（OR邏輯）：\n"
        f"   - 探索所有地點（{visited+1}/{total_locs}）：{'✓' if visited+1 >= total_locs else '✗ 未達成'}\n"
        f"   - 持有關鍵線索：{'✓' if have_clue else '✗ 未持有'}\n"
        f"4. OR 條件至少一項成立 → 任務{'觸發' if trigger else '繼續等待'}。"
    )
    conclusion = json.dumps({
        "探索進度": f"{visited+1}/{total_locs}", "地點「"+location+"」": "已標記",
        "任務觸發": quest if trigger else "待完成",
        "獲得": f"經驗值 +{exp}" if trigger else "探索記錄更新",
    }, ensure_ascii=False)
    return action, state, steps, conclusion


def _tmpl_quest_defeat(npc, faction, quest, location, item, rng):
    kills = rng.randint(1, 3)
    need_kills = rng.randint(kills, kills + 2)
    player_atk = rng.randint(40, 80)
    enemy_hp = rng.randint(60, 120)
    exp = rng.randint(300, 1000)
    action = f"玩家在「{location}」擊敗了 {kills} 名{faction}成員，完成本回合戰鬥。"
    state = json.dumps({"任務「"+quest+"」擊殺進度": f"{kills}/{need_kills}", "玩家攻擊力": player_atk, "剩餘敵人HP": enemy_hp}, ensure_ascii=False)
    done = kills >= need_kills
    steps = (
        f"1. 擊殺判定：玩家攻擊力 {player_atk} vs 敵人HP {enemy_hp}\n"
        f"   {player_atk} {'>' if player_atk > enemy_hp else '≤'} {enemy_hp} "
        f"→ {'一擊必殺' if player_atk > enemy_hp else '需多輪攻擊'}\n"
        f"2. 擊殺進度更新：{kills}/{need_kills}\n"
        f"3. 任務完成條件：擊殺數 ≥ {need_kills}\n"
        f"   當前 {kills} {'≥' if done else '<'} {need_kills} → {'任務完成' if done else f'尚需擊敗 {need_kills - kills} 名'}\n"
        f"4. 完成後觸發：經驗值 +{exp}，解鎖與{npc}的後續劇情對話。"
    )
    conclusion = json.dumps({
        "擊殺進度": f"{kills}/{need_kills}", "任務狀態": "已完成" if done else "進行中",
        "任務觸發": quest if done else "待完成",
        "獎勵": {"經驗值": exp, "後續劇情": f"與{npc}的對話"} if done else {},
    }, ensure_ascii=False)
    return action, state, steps, conclusion


def _tmpl_hidden_time_place(npc, faction, quest, location, item, rng):
    hour = rng.randint(0, 3)   # 0=midnight, 1=dawn, 2=dusk, 3=noon
    hour_name = ["午夜", "黎明", "黃昏", "正午"][hour]
    req_hour = ["午夜", "黎明"][hour % 2]
    time_match = hour_name == req_hour
    faction_rep = rng.randint(20, 80)
    rep_req = rng.randint(40, 60)
    rep_ok = faction_rep >= rep_req
    both = time_match and rep_ok
    action = f"玩家於{hour_name}時分抵達「{location}」，當前與{faction}聲望值為 {faction_rep}。"
    state = json.dumps({"時間": hour_name, "地點": location, f"{faction}聲望": faction_rep, "隱藏任務「"+quest+"」": "未觸發"}, ensure_ascii=False)
    steps = (
        f"1. 時間條件：隱藏任務「{quest}」要求{req_hour}時分抵達「{location}」\n"
        f"   當前 {hour_name} {'=' if time_match else '≠'} {req_hour} → {'✓ 滿足' if time_match else '✗ 不滿足'}\n"
        f"2. 聲望條件：需{faction}聲望 ≥ {rep_req}\n"
        f"   當前 {faction_rep} {'≥' if rep_ok else '<'} {rep_req} → {'✓ 滿足' if rep_ok else '✗ 不滿足'}\n"
        f"3. 觸發邏輯（AND）：時間 {'✓' if time_match else '✗'} AND 聲望 {'✓' if rep_ok else '✗'} "
        f"→ {'兩項均滿足，觸發隱藏任務' if both else '條件未完全滿足，任務待機'}"
    )
    conclusion = json.dumps({
        "時間條件": "滿足" if time_match else "不滿足",
        "聲望條件": "滿足" if rep_ok else "不滿足",
        "隱藏任務觸發": quest if both else "無",
        "觸發結果": f"出現神秘NPC，開啟「{quest}」劇情" if both else "無特殊事件",
    }, ensure_ascii=False)
    return action, state, steps, conclusion


def _tmpl_hidden_item_faction(npc, faction, quest, location, item, rng):
    has_item = rng.choice([True, False])
    faction_rep = rng.randint(10, 90)
    rep_req = rng.randint(30, 60)
    rep_ok = faction_rep >= rep_req
    triggered = has_item and rep_ok
    reward_gold = rng.randint(100, 500)
    action = f"玩家持有{item}進入「{location}」，當前{faction}聲望為 {faction_rep}。"
    state = json.dumps({f"持有{item}": has_item, f"{faction}聲望": faction_rep, "隱藏劇情「"+quest+"」": "待機"}, ensure_ascii=False)
    steps = (
        f"1. 道具條件：隱藏劇情「{quest}」需玩家攜帶{item}\n"
        f"   玩家{'✓ 持有' if has_item else '✗ 未持有'}{item}\n"
        f"2. 陣營條件：需{faction}聲望 ≥ {rep_req}\n"
        f"   當前 {faction_rep} {'≥' if rep_ok else '<'} {rep_req} → {'✓ 滿足' if rep_ok else '✗ 不滿足'}\n"
        f"3. 觸發（AND）：{'兩項條件均滿足' if triggered else '條件不足'}\n"
        f"   → {'解鎖隱藏劇情，出現秘密商人，可以 ' + str(reward_gold) + ' 金幣換取稀有裝備' if triggered else '無事件觸發'}"
    )
    conclusion = json.dumps({
        "道具條件": "滿足" if has_item else "不滿足",
        "聲望條件": "滿足" if rep_ok else "不滿足",
        "隱藏劇情觸發": quest if triggered else "無",
        "秘密商人": "出現" if triggered else "未出現",
        "獎勵機會": f"可交易，稀有裝備售價 {reward_gold} 金幣" if triggered else "無",
    }, ensure_ascii=False)
    return action, state, steps, conclusion


def _tmpl_faction_assist(npc, faction, quest, location, item, rng):
    help_rep = rng.randint(15, 30)
    rival_pen = rng.randint(10, 20)
    base_a = rng.randint(30, 60)
    base_b = rng.randint(30, 60)
    rival = _DEFAULT_FACTIONS[(_DEFAULT_FACTIONS.index(faction) + 1) % len(_DEFAULT_FACTIONS)] if faction in _DEFAULT_FACTIONS else "敵對陣營"
    new_a = base_a + help_rep
    new_b = max(0, base_b - rival_pen)
    action = f"玩家協助{faction}完成物資護送任務，成功抵達「{location}」並擊退了攔截者。"
    state = json.dumps({f"{faction}聲望": base_a, f"{rival}聲望": base_b, "任務「"+quest+"」": "進行中"}, ensure_ascii=False)
    steps = (
        f"1. 護送任務完成，判定為對{faction}的直接援助行為。\n"
        f"2. {faction}聲望變化：+{help_rep}（協助護送）\n"
        f"   {base_a} + {help_rep} = {new_a}\n"
        f"3. {rival}聲望懲罰（敵對陣營感知）：-{rival_pen}\n"
        f"   {base_b} - {rival_pen} = {new_b}\n"
        f"4. 陣營聲望影響：{faction}聲望 {new_a} ≥ {base_a + 10} → "
        f"{'解鎖進階委託' if new_a >= 60 else '聲望仍在建立中'}"
    )
    conclusion = json.dumps({
        f"{faction}聲望變化": help_rep, f"{faction}新聲望": new_a,
        f"{rival}聲望變化": -rival_pen, f"{rival}新聲望": new_b,
        "任務狀態": "完成", "解鎖進階委託": new_a >= 60,
    }, ensure_ascii=False)
    return action, state, steps, conclusion


def _tmpl_faction_betray(npc, faction, quest, location, item, rng):
    base_rep = rng.randint(50, 80)
    betray_pen = rng.randint(30, 50)
    new_rep = max(0, base_rep - betray_pen)
    chain_events = new_rep < 20
    action = f"玩家在談判中將{faction}的秘密計劃透露給對立陣營，並接受了對方的報酬。"
    state = json.dumps({f"{faction}聲望": base_rep, "玩家行動類型": "背叛", "地點": location}, ensure_ascii=False)
    steps = (
        f"1. 行動分類：「背叛陣營機密」屬於最高級敵意行為。\n"
        f"2. 聲望懲罰計算：{base_rep} - {betray_pen}（背叛懲罰） = {new_rep}\n"
        f"3. 連鎖後果判定：\n"
        f"   聲望 {new_rep} {'< 20' if chain_events else '≥ 20'} → "
        f"{'觸發追殺令：' + faction + '派遣刺客追殺玩家' if chain_events else '關係惡化但尚未觸發追殺'}\n"
        f"4. 所有{faction}相關任務狀態強制設為「失敗」。\n"
        f"5. NPC「{npc}」（若屬{faction}成員）轉為敵對，拒絕一切互動。"
    )
    conclusion = json.dumps({
        f"{faction}聲望變化": -betray_pen, f"{faction}新聲望": new_rep,
        "追殺令": "已觸發" if chain_events else "未觸發",
        "任務失敗": True, f"NPC{npc}狀態": "敵對",
        "永久後果": f"無法再接受{faction}委託",
    }, ensure_ascii=False)
    return action, state, steps, conclusion


def _tmpl_item_craft(npc, faction, quest, location, item, rng):
    item2 = _DEFAULT_ITEMS[(_DEFAULT_ITEMS.index(item) + 2) % len(_DEFAULT_ITEMS)] if item in _DEFAULT_ITEMS else "神秘材料"
    qty1 = rng.randint(2, 5)
    qty2 = rng.randint(1, 3)
    skill_req = rng.randint(3, 7)
    player_skill = rng.randint(4, 8)
    success_rate = min(95, 60 + (player_skill - skill_req) * 10)
    result_item = f"強化版{item}"
    action = f"玩家在「{location}」的工作台嘗試合成，材料：{item}×{qty1}、{item2}×{qty2}。"
    state = json.dumps({"玩家合成技能": player_skill, f"{item}庫存": qty1, f"{item2}庫存": qty2, "地點": location}, ensure_ascii=False)
    steps = (
        f"1. 配方核查：{item}×{qty1} + {item2}×{qty2} → {result_item}（配方已知）\n"
        f"2. 材料充足性：{item} {qty1}件 ✓，{item2} {qty2}件 ✓\n"
        f"3. 技能需求：合成技能 ≥ {skill_req}\n"
        f"   玩家技能 {player_skill} ≥ {skill_req} → ✓ 滿足\n"
        f"4. 成功率計算：基礎 60% + 技能加成 ({player_skill} - {skill_req}) × 10% = {success_rate}%\n"
        f"5. 消耗材料：{item}×{qty1}、{item2}×{qty2}（無論成功與否均消耗）"
    )
    conclusion = json.dumps({
        "合成結果": result_item, "成功率": f"{success_rate}%",
        "材料消耗": {item: qty1, item2: qty2},
        "技能條件": "滿足", "注意": "材料於嘗試時扣除，失敗不退還",
    }, ensure_ascii=False)
    return action, state, steps, conclusion


_REASONING_SCENARIO_FUNCS = [
    _tmpl_npc_gift,
    _tmpl_npc_reject,
    _tmpl_npc_help,
    _tmpl_quest_collect,
    _tmpl_quest_explore,
    _tmpl_quest_defeat,
    _tmpl_hidden_time_place,
    _tmpl_hidden_item_faction,
    _tmpl_faction_assist,
    _tmpl_faction_betray,
    _tmpl_item_craft,
]


def _extract_lore_entities(sources: list[dict]) -> tuple[list[str], list[str]]:
    """
    Extract character names and faction names from lore sources to use as
    realistic entity variables in reasoning templates.
    Returns (npc_names, faction_names), both de-duplicated lists.
    """
    npcs: set[str] = set()
    factions: set[str] = set()
    for src in sources:
        text = src["text"]
        lang = src["metadata"].get("language", "en")
        if lang != "en":
            continue
        # Lightweight extraction — only multi-word names for NPCs (higher precision)
        for m in re.finditer(r"\b([A-Z][a-z]{2,15})\s+([A-Z][a-z]{2,15})\b", text):
            p1, p2 = m.group(1), m.group(2)
            if p1 not in _SKIP_WORDS and p2 not in _SKIP_WORDS:
                if p1 not in _NUMBER_WORDS and p2 not in _NUMBER_WORDS:
                    npcs.add(f"{p1} {p2}")
        # Faction suffixes
        for m in re.finditer(
            r"\b([A-Z][a-z]{2,15}(?:\s+[A-Z][a-z]{2,15})?)\s+"
            r"(Order|Guild|Legion|Brotherhood|Council|Alliance|Empire|Kingdom)\b",
            text,
        ):
            factions.add(m.group(0))
    return list(npcs)[:40], list(factions)[:20]


def build_reasoning_records(
    sources: list[dict],
    rng: random.Random,
) -> list[dict]:
    """
    Generate synthetic semantic-reasoning ShareGPT records (Task C).

    Uses 11 scenario templates × entity name combinations extracted from lore
    sources, supplemented by built-in fantasy name lists to ensure ≥ 1,000
    records even with limited source diversity.
    """
    records: list[dict] = []
    global_seen: set[str] = set()

    # Collect entity names from lore
    lore_npcs, lore_factions = _extract_lore_entities(sources)
    npcs      = (lore_npcs or []) + _DEFAULT_NPCS
    factions  = (lore_factions or []) + _DEFAULT_FACTIONS
    quests    = _DEFAULT_QUESTS
    locations = _DEFAULT_LOCATIONS
    items     = _DEFAULT_ITEMS

    # Shuffle to vary combinations
    rng.shuffle(npcs)
    rng.shuffle(factions)

    # Generate records by cycling through templates × entities.
    # 11 templates × 5 repeats × 25 iterations = 1,375 potential records
    # (after dedup, typically 1,000–1,200 unique records).
    template_cycle = list(_REASONING_SCENARIO_FUNCS) * 5   # 55 entries
    iters_per_tmpl = 25
    entity_pool = [
        (rng.choice(npcs), rng.choice(factions), rng.choice(quests),
         rng.choice(locations), rng.choice(items))
        for _ in range(max(500, len(template_cycle) * iters_per_tmpl))
    ]

    for i, tmpl_fn in enumerate(template_cycle):
        for j in range(iters_per_tmpl):
            idx = (i * 6 + j) % len(entity_pool)
            npc, fac, quest, loc, item = entity_pool[idx]
            # Re-seed per (template, entity) for reproducible output
            local_rng = random.Random(rng.randint(0, 2**31))
            try:
                action, state, steps, conclusion = tmpl_fn(npc, fac, quest, loc, item, local_rng)
            except Exception:
                continue

            human_turn = f"【玩家行動】\n{action}\n\n【當前遊戲狀態】\n{state}"
            gpt_turn   = f"【推理步驟】\n{steps}\n\n【結論】\n{conclusion}"

            # Dedup by (template index, npc, quest)
            key = hashlib.md5(f"{i}|{npc}|{quest}|{j}".encode()).hexdigest()
            if key in global_seen:
                continue
            global_seen.add(key)

            records.append(sharegpt(_REASONING_SYSTEM, human_turn, gpt_turn))

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
        "--task", choices=["storyteller", "analyst", "translator", "reasoning"],
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

    tasks = [args.task] if args.task else ["storyteller", "analyst", "translator", "reasoning"]

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

    # All sources used for entity extraction in the reasoning task
    reasoning_sources = all_sources

    log.info(
        "Sources: storyteller=%d, analyst=%d, translator=%d, reasoning(lore)=%d",
        len(story_sources), len(analyst_sources), len(translator_sources_full),
        len(reasoning_sources),
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

    if "reasoning" in tasks:
        log.info("Building lora_reasoning records ...")
        records = build_reasoning_records(reasoning_sources, rng)
        results["reasoning"] = records
        print_stats("lora_reasoning", records)

    if args.stats:
        log.info("--stats mode: no files written.")
        return

    # ----------------------------------------------------------------- write
    output_map = {
        "storyteller": output_dir / "lora_storyteller.jsonl",
        "analyst":     output_dir / "lora_analyst.jsonl",
        "translator":  output_dir / "lora_translator.jsonl",
        "reasoning":   output_dir / "lora_reasoning.jsonl",
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
