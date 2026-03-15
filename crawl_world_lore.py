"""
Crawl public TRPG wiki pages and store chunked text in the world_reference
ChromaDB collection.  Run once per world before starting a campaign so that
save_load.py can auto-generate world_context and seed world-specific rules.

Usage:
    python tools/crawl_world_lore.py                  # crawl all worlds
    python tools/crawl_world_lore.py --worlds pathfinder wh40k
    python tools/crawl_world_lore.py --worlds pathfinder --force
"""

import sys
import os
import time
import hashlib
import argparse
import subprocess

# Project root on PYTHONPATH so absolute imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
from ai.rag_system import RAGSystem

# ---------------------------------------------------------------------------
# Target URLs per world_setting id.
# dnd5e is excluded — its rules come from the SRD JSON seeder (seed_srd.py)
# and the built-in world_lore in config.py.
# ---------------------------------------------------------------------------
CRAWL_TARGETS = {
    "pathfinder": [
        "https://pathfinderwiki.com/wiki/Golarion",
        "https://pathfinderwiki.com/wiki/Pathfinder_Society",
    ],
    "warhammer_fantasy": [
        "https://whfb.lexicanum.com/wiki/The_Empire",
        "https://whfb.lexicanum.com/wiki/Chaos_(Warhammer)",
    ],
    "wh40k": [
        "https://wh40k.lexicanum.com/wiki/Imperium_of_Man",
        "https://wh40k.lexicanum.com/wiki/Chaos",
    ],
    "shadowrun": [
        "https://shadowrun.fandom.com/wiki/Sixth_World",
        "https://shadowrun.fandom.com/wiki/Seattle_metroplex",
    ],
    "world_of_darkness": [
        "https://whitewolf.fandom.com/wiki/World_of_Darkness",
        "https://whitewolf.fandom.com/wiki/Vampire:_The_Masquerade",
    ],
    "call_of_cthulhu": [
        "https://lovecraft.fandom.com/wiki/Cthulhu_Mythos",
        "https://lovecraft.fandom.com/wiki/Arkham",
    ],
    "iron_kingdoms": [
        "https://ironkingdoms.fandom.com/wiki/Iron_Kingdoms",
        "https://ironkingdoms.fandom.com/wiki/Western_Immoren",
    ],
    "blades_in_the_dark": [
        "https://bladesinthedark.com/doskvol",
        "https://bladesinthedark.com/streets-doskvol",
    ],
    "hearts_of_wulin": [
        "https://www.gauntlet-rpg.com/hearts-of-wulin.html",
    ],
    "l5r": [
        "https://l5r.fandom.com/wiki/Rokugan",
        "https://l5r.fandom.com/wiki/Great_Clans",
    ],
    "deadlands": [
        "https://peginc.com/savage-settings/deadlands/",
    ],
    "mutant_year_zero": [
        "https://mutant.fandom.com/wiki/Mutant:_Year_Zero",
    ],
    "gloomhaven": [
        "https://gloomhaven.fandom.com/wiki/Gloomhaven",
    ],
}

# HTTP headers that reduce bot-blocking on wiki sites
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

_CHUNK_SIZE = 400   # target characters per RAG chunk


def _fetch_with_curl(url):
    # Fallback using system curl.exe which often bypasses TLS fingerprinting blocks
    try:
        cmd = [
            "curl.exe", "-s", "-L",
            "-H", f"User-Agent: {_HEADERS['User-Agent']}",
            "-H", f"Accept: {_HEADERS['Accept']}",
            "-H", f"Accept-Language: {_HEADERS['Accept-Language']}",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except Exception as e:
        print(f"  [DEBUG] curl fallback failed: {e}")
    return None


def _fetch_page(url):
    # Returns the raw HTML string or None on any network / HTTP error.
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        if resp.status_code == 403:
            # Try curl fallback on 403
            html = _fetch_with_curl(url)
            if html:
                return html
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        # Final attempt if any other error occurred
        html = _fetch_with_curl(url)
        if html:
            return html
        print(f"  [WARN] Could not fetch {url}: {e}")
        return None


def _extract_text(html):
    # Strip boilerplate tags; return a single whitespace-collapsed string
    # built from <h2>, <h3>, and <p> elements in document order.
    soup = BeautifulSoup(html, "html.parser")
    # Remove navigation and decorative elements before extracting prose
    for tag in soup.find_all(["nav", "footer", "header", "script", "style", "aside"]):
        tag.decompose()

    parts = []
    for el in soup.find_all(["h2", "h3", "p"]):
        text = el.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)
    return " ".join(parts)


def _split_chunks(text, chunk_size=_CHUNK_SIZE):
    # Split on sentence boundaries ('. ', '! ', '? ') so chunks never end
    # mid-sentence.  Any sentence longer than chunk_size stands alone.
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If the sentence itself exceeds chunk_size, store it as-is
            current = sentence
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def crawl_world(world_id, urls, rag, force=False):
    # Skip already-seeded worlds unless --force was passed
    if not force and rag.world_reference_seeded(world_id):
        print(f"[SKIP] {world_id} — already seeded (use --force to re-crawl)")
        return

    total_chunks = 0
    for url in urls:
        print(f"  Fetching {url} …")
        html = _fetch_page(url)
        if not html:
            continue

        text = _extract_text(html)
        if not text.strip():
            print(f"  [WARN] No usable text extracted from {url}")
            time.sleep(1)
            continue

        chunks = _split_chunks(text)
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]

        for i, chunk in enumerate(chunks):
            ref_id = f"ref_{world_id}_{url_hash}_{i}"
            rag.add_world_reference(chunk, ref_id, world_id=world_id, url=url)

        print(f"  Stored {len(chunks)} chunks from {url}")
        total_chunks += len(chunks)
        # Polite crawl delay between requests
        time.sleep(1)

    print(f"[DONE] {world_id} — {total_chunks} total chunks stored")


def main():
    parser = argparse.ArgumentParser(
        description="Crawl TRPG wiki pages into the world_reference RAG collection."
    )
    parser.add_argument(
        "--worlds", nargs="*",
        help="Space-separated list of world_setting ids to crawl. "
             "Omit to crawl all worlds in CRAWL_TARGETS.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-crawl even if world_reference already has entries for this world.",
    )
    args = parser.parse_args()

    target_ids = args.worlds if args.worlds else list(CRAWL_TARGETS.keys())
    unknown = [w for w in target_ids if w not in CRAWL_TARGETS]
    if unknown:
        print(f"[WARN] Unknown world ids (no URLs defined): {unknown}")
        target_ids = [w for w in target_ids if w in CRAWL_TARGETS]

    if not target_ids:
        print("Nothing to crawl.")
        return

    rag = RAGSystem()
    for world_id in target_ids:
        print(f"\n=== Crawling: {world_id} ===")
        crawl_world(world_id, CRAWL_TARGETS[world_id], rag, force=args.force)

    print("\nAll done.")


if __name__ == "__main__":
    main()
