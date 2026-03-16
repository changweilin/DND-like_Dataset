"""
Automated web scraper for DND-like_Dataset project.

Crawls websites mentioned in context_ref.md and the global narrative ecosystem
research report. Saves raw text to data/raw/ with metadata JSON sidecar files.
Logs every URL with HTTP status, word count, and char count.

Usage:
    python scraper.py                          # scrape everything
    python scraper.py --category trpg          # only TRPG targets
    python scraper.py --category webnovel      # only web novel targets
    python scraper.py --sources pathfinder wh40k
    python scraper.py --force                  # re-scrape already-done URLs
    python scraper.py --no-robots             # skip robots.txt checks
    python scraper.py --dry-run               # list targets without fetching
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import pathlib
import random
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.robotparser
from typing import Callable, Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# CRAWL TARGETS
# ---------------------------------------------------------------------------

CRAWL_TARGETS: dict[str, dict] = {
    "trpg": {
        "pathfinder": {
            "urls": [
                "https://pathfinderwiki.com/wiki/Golarion",
                "https://pathfinderwiki.com/wiki/Pathfinder_Society",
            ],
            "extractor": "mediawiki",
            "language": "en",
            "display_name": "Pathfinder",
            "tags": ["fantasy", "pathfinder"],
        },
        "warhammer_fantasy": {
            "urls": [
                "https://whfb.lexicanum.com/wiki/The_Empire",
                "https://whfb.lexicanum.com/wiki/Chaos_(Warhammer)",
            ],
            "extractor": "lexicanum",
            "language": "en",
            "display_name": "Warhammer Fantasy",
            "tags": ["dark_fantasy", "warhammer"],
        },
        "wh40k": {
            "urls": [
                "https://wh40k.lexicanum.com/wiki/Imperium_of_Man",
                "https://wh40k.lexicanum.com/wiki/Chaos",
            ],
            "extractor": "lexicanum",
            "language": "en",
            "display_name": "Warhammer 40,000",
            "tags": ["scifi", "grimdark", "warhammer40k"],
        },
        "shadowrun": {
            "urls": [
                "https://shadowrun.fandom.com/wiki/Sixth_World",
                "https://shadowrun.fandom.com/wiki/Seattle_metroplex",
            ],
            "extractor": "fandom",
            "language": "en",
            "display_name": "Shadowrun",
            "tags": ["cyberpunk", "fantasy", "shadowrun"],
        },
        "world_of_darkness": {
            "urls": [
                "https://whitewolf.fandom.com/wiki/World_of_Darkness",
                "https://whitewolf.fandom.com/wiki/Vampire:_The_Masquerade",
            ],
            "extractor": "fandom",
            "language": "en",
            "display_name": "World of Darkness",
            "tags": ["horror", "gothic", "world_of_darkness"],
        },
        "call_of_cthulhu": {
            "urls": [
                "https://lovecraft.fandom.com/wiki/Cthulhu_Mythos",
                "https://lovecraft.fandom.com/wiki/Arkham",
            ],
            "extractor": "fandom",
            "language": "en",
            "display_name": "Call of Cthulhu",
            "tags": ["horror", "cosmic", "lovecraft"],
        },
        "iron_kingdoms": {
            "urls": [
                "https://ironkingdoms.fandom.com/wiki/Iron_Kingdoms",
                "https://ironkingdoms.fandom.com/wiki/Western_Immoren",
            ],
            "extractor": "fandom",
            "language": "en",
            "display_name": "Iron Kingdoms",
            "tags": ["steampunk", "fantasy", "iron_kingdoms"],
        },
        "blades_in_the_dark": {
            "urls": [
                "https://bladesinthedark.com/doskvol",
                "https://bladesinthedark.com/streets-doskvol",
            ],
            "extractor": "generic",
            "language": "en",
            "display_name": "Blades in the Dark",
            "tags": ["steampunk", "heist", "blades_in_the_dark"],
        },
        "l5r": {
            "urls": [
                "https://l5r.fandom.com/wiki/Rokugan",
                "https://l5r.fandom.com/wiki/Great_Clans",
            ],
            "extractor": "fandom",
            "language": "en",
            "display_name": "Legend of the Five Rings",
            "tags": ["wuxia", "eastern_fantasy", "l5r"],
        },
        "deadlands": {
            "urls": [
                "https://peginc.com/savage-settings/deadlands/",
            ],
            "extractor": "generic",
            "language": "en",
            "display_name": "Deadlands",
            "tags": ["western", "horror", "deadlands"],
        },
        "mutant_year_zero": {
            "urls": [
                "https://mutant.fandom.com/wiki/Mutant:_Year_Zero",
            ],
            "extractor": "fandom",
            "language": "en",
            "display_name": "Mutant: Year Zero",
            "tags": ["post_apocalyptic", "scifi", "mutant"],
        },
        "gloomhaven": {
            "urls": [
                "https://gloomhaven.fandom.com/wiki/Gloomhaven",
            ],
            "extractor": "fandom",
            "language": "en",
            "display_name": "Gloomhaven",
            "tags": ["dungeon_crawler", "fantasy", "gloomhaven"],
        },
    },
    "webnovel": {
        "ao3_fantasy": {
            "urls": [],  # populated dynamically by sample_ao3()
            "extractor": "ao3",
            "language": "en",
            "display_name": "Archive of Our Own",
            "sample_strategy": "ao3",
            "sample_tags": ["Fantasy", "Science Fiction"],
            "sample_count": 30,
            "tags": ["fanfic", "original", "fantasy", "en"],
        },
        "royalroad_litrpg": {
            "urls": [],  # populated dynamically by sample_royalroad()
            "extractor": "royalroad",
            "language": "en",
            "display_name": "Royal Road",
            "sample_strategy": "royalroad",
            "sample_tags": ["litrpg"],
            "sample_count": 20,
            "tags": ["litrpg", "progression", "fantasy", "en"],
        },
        "syosetu_isekai": {
            "urls": [],  # populated dynamically by sample_syosetu()
            "extractor": "syosetu",
            "language": "ja",
            "display_name": "小説家になろう (Syosetu)",
            "sample_strategy": "syosetu",
            "sample_count": 15,
            "tags": ["isekai", "fantasy", "ja"],
        },
        "lexicanum_wh": {
            "urls": [
                "https://whfb.lexicanum.com/wiki/The_Empire",
                "https://wh40k.lexicanum.com/wiki/Imperium_of_Man",
            ],
            "extractor": "lexicanum",
            "language": "en",
            "display_name": "Lexicanum (Warhammer)",
            "tags": ["warhammer", "lore", "en"],
        },
    },
}

# ---------------------------------------------------------------------------
# HTTP HEADERS (reused from crawl_world_lore.py — proven effective)
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7,ja;q=0.6",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    log_dir = pathlib.Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"scrape_{datetime.date.today().strftime('%Y%m%d')}.log"

    fmt = "%(asctime)s [%(levelname)-8s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger("scraper")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(ch)

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(fh)

    return logger


log = setup_logging()

# ---------------------------------------------------------------------------
# HTTP LAYER
# ---------------------------------------------------------------------------

def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(_HEADERS)
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_ROBOTS_CACHE: dict[str, urllib.robotparser.RobotFileParser] = {}


def check_robots(url: str) -> bool:
    """Return True if crawling is allowed by robots.txt."""
    parsed = urllib.parse.urlparse(url)
    domain_key = f"{parsed.scheme}://{parsed.netloc}"
    if domain_key not in _ROBOTS_CACHE:
        rp = urllib.robotparser.RobotFileParser()
        robots_url = f"{domain_key}/robots.txt"
        try:
            rp.set_url(robots_url)
            rp.read()
        except Exception:
            rp = None
        _ROBOTS_CACHE[domain_key] = rp
    rp = _ROBOTS_CACHE[domain_key]
    if rp is None:
        return True
    return rp.can_fetch(_HEADERS["User-Agent"], url)


def _fetch_with_curl(url: str) -> Optional[str]:
    """Fallback using system curl, which often bypasses TLS fingerprinting."""
    try:
        cmd = [
            "curl", "-s", "-L",
            "-H", f"User-Agent: {_HEADERS['User-Agent']}",
            "-H", f"Accept: {_HEADERS['Accept']}",
            "-H", f"Accept-Language: {_HEADERS['Accept-Language']}",
            "--max-time", "20",
            url,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except Exception as e:
        log.debug(f"curl fallback failed for {url}: {e}")
    return None


def fetch_page(
    session: requests.Session,
    url: str,
    respect_robots: bool = True,
) -> tuple[Optional[str], int]:
    """Fetch a URL. Returns (html_text, http_status). Never raises."""
    if respect_robots and not check_robots(url):
        log.warning(f"ROBOTS_BLOCKED {url}")
        return None, 0

    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 403:
            html = _fetch_with_curl(url)
            if html:
                return html, 200
        resp.raise_for_status()
        # Handle Shift-JIS and other encodings (important for Syosetu)
        resp.encoding = resp.apparent_encoding
        return resp.text, resp.status_code
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        html = _fetch_with_curl(url)
        if html:
            return html, 200
        log.warning(f"FAILED  {url} | status={status} | reason={e}")
        return None, status
    except Exception as e:
        html = _fetch_with_curl(url)
        if html:
            return html, 200
        log.warning(f"FAILED  {url} | reason={e}")
        return None, 0


def polite_delay(min_s: float = 1.0, max_s: float = 2.0) -> None:
    time.sleep(random.uniform(min_s, max_s))

# ---------------------------------------------------------------------------
# EXTRACTOR FUNCTIONS
# ---------------------------------------------------------------------------

def extract_fandom(soup: BeautifulSoup) -> str:
    """Extract clean text from Fandom wiki pages (*.fandom.com)."""
    for tag in soup.find_all([
        "nav", "footer", "header", "script", "style", "aside",
        "div", "section",
    ], class_=re.compile(
        r"(portable-infobox|infobox|toc|wikia-ad|wds-tabs|gallery|"
        r"noprint|mw-editsection|pi-item|references|reflist|navbox|"
        r"thumb|caption|noexcerpt)"
    )):
        tag.decompose()

    for tag in soup.find_all(["sup", "span"], class_=re.compile(r"reference")):
        tag.decompose()

    content = soup.find("div", class_="mw-parser-output")
    if not content:
        content = soup.find("div", id="mw-content-text")
    if not content:
        return extract_generic(soup)

    parts = []
    for el in content.find_all(["h2", "h3", "h4", "p"]):
        text = el.get_text(separator=" ", strip=True)
        if text and len(text) > 20:
            parts.append(text)
    return "\n\n".join(parts)


def extract_mediawiki(soup: BeautifulSoup) -> str:
    """Extract clean text from plain MediaWiki pages (e.g. Pathfinder Wiki)."""
    for tag in soup.find_all([
        "nav", "footer", "header", "script", "style", "aside",
    ]):
        tag.decompose()

    for tag in soup.find_all(["table"], class_=re.compile(r"(infobox|navbox|wikitable)")):
        tag.decompose()

    for tag in soup.find_all(["div"], id=re.compile(r"(toc|mw-navigation|mw-head|mw-panel)")):
        tag.decompose()

    for tag in soup.find_all(class_=re.compile(r"(mw-editsection|noprint|reference)")):
        tag.decompose()

    for tag in soup.find_all(["sup"]):
        tag.decompose()

    content = soup.find("div", id="mw-content-text")
    if not content:
        content = soup.find("div", class_="mw-parser-output")
    if not content:
        return extract_generic(soup)

    parts = []
    for el in content.find_all(["h2", "h3", "h4", "p"]):
        text = el.get_text(separator=" ", strip=True)
        if text and len(text) > 20:
            parts.append(text)
    return "\n\n".join(parts)


def extract_lexicanum(soup: BeautifulSoup) -> str:
    """Extract clean text from Lexicanum wikis (Warhammer lore)."""
    # Remove notice boxes and source sections
    for tag in soup.find_all("table", class_=re.compile(r"notices?")):
        tag.decompose()

    text = extract_mediawiki(soup)

    # Strip inline citation markers like [1], [2a], [3b]
    text = re.sub(r"\[\d+[a-z]?\]", "", text)
    # Strip "Sources" and "See Also" trailing sections
    text = re.sub(r"\n\nSources\n\n.*$", "", text, flags=re.DOTALL)
    text = re.sub(r"\n\nSee Also\n\n.*$", "", text, flags=re.DOTALL)
    return text.strip()


def extract_ao3(soup: BeautifulSoup) -> str:
    """Extract story text from Archive of Our Own work pages."""
    for tag in soup.find_all(["div"], id=re.compile(r"(header|footer|kudos|comments|series|feedback)")):
        tag.decompose()

    for tag in soup.find_all(["dl"], class_=re.compile(r"work.meta")):
        tag.decompose()

    content = soup.find("div", class_="userstuff")
    if not content:
        return extract_generic(soup)

    parts = []
    for el in content.find_all(["p", "h2", "h3"]):
        text = el.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def extract_royalroad(soup: BeautifulSoup) -> str:
    """Extract chapter text from Royal Road fiction pages."""
    for tag in soup.find_all(class_=re.compile(r"(page-header|sidebar|author-info|chapter-list|comments)")):
        tag.decompose()

    content = soup.find("div", class_="chapter-content")
    if not content:
        return extract_generic(soup)

    parts = []
    for el in content.find_all(["p", "h2", "h3"]):
        text = el.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def extract_syosetu(soup: BeautifulSoup) -> str:
    """Extract chapter text from 小説家になろう (Syosetu) pages."""
    content = soup.find("div", id="novel_honbun")
    if not content:
        # Fallback for newer site layout
        content = soup.find("div", class_=re.compile(r"(novel_view|honbun)"))
    if not content:
        return extract_generic(soup)

    parts = []
    for el in content.find_all(["p"]):
        text = el.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def extract_generic(soup: BeautifulSoup) -> str:
    """Generic extractor — mirrors crawl_world_lore.py's _extract_text."""
    for tag in soup.find_all(["nav", "footer", "header", "script", "style", "aside"]):
        tag.decompose()

    parts = []
    for el in soup.find_all(["h2", "h3", "p"]):
        text = el.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


EXTRACTOR_MAP: dict[str, Callable[[BeautifulSoup], str]] = {
    "fandom": extract_fandom,
    "mediawiki": extract_mediawiki,
    "lexicanum": extract_lexicanum,
    "ao3": extract_ao3,
    "royalroad": extract_royalroad,
    "syosetu": extract_syosetu,
    "generic": extract_generic,
}

# ---------------------------------------------------------------------------
# SAMPLING STRATEGIES FOR LARGE PLATFORMS
# ---------------------------------------------------------------------------

def sample_ao3(
    session: requests.Session,
    tags: list[str],
    count: int,
) -> list[str]:
    """Sample top-kudos complete English works from AO3. Returns chapter URLs."""
    urls = []
    page = 1
    while len(urls) < count:
        tag_param = urllib.parse.quote(tags[0]) if tags else "Fantasy"
        listing_url = (
            f"https://archiveofourown.org/works"
            f"?tag={tag_param}"
            f"&work_search[sort_column]=kudos"
            f"&work_search[complete]=T"
            f"&work_search[language_id]=1"
            f"&page={page}"
        )
        log.info(f"SAMPLE  AO3 listing page {page}")
        html, status = fetch_page(session, listing_url)
        if not html:
            break

        soup = BeautifulSoup(html, "lxml")
        work_links = soup.select("ol#work-index-groups li.work h4 a[href^='/works/']")
        if not work_links:
            break

        for link in work_links:
            if len(urls) >= count:
                break
            work_url = "https://archiveofourown.org" + link["href"]
            # Fetch just the first chapter
            work_html, _ = fetch_page(session, work_url)
            if work_html:
                work_soup = BeautifulSoup(work_html, "lxml")
                # Try to find first chapter link
                chapter_link = work_soup.select_one("div#chapter-index a")
                if chapter_link:
                    chapter_url = "https://archiveofourown.org" + chapter_link["href"]
                    urls.append(chapter_url)
                else:
                    urls.append(work_url)
            polite_delay()

        page += 1
        if page > 5:
            break
    return urls[:count]


def sample_royalroad(
    session: requests.Session,
    genre_tags: list[str],
    count: int,
) -> list[str]:
    """Sample best-rated LitRPG fictions from Royal Road. Returns first chapter URLs."""
    urls = []
    genre = genre_tags[0] if genre_tags else "litrpg"
    listing_url = f"https://www.royalroad.com/fictions/best-rated?genre={genre}"
    log.info(f"SAMPLE  Royal Road listing: {listing_url}")
    html, status = fetch_page(session, listing_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    fiction_links = soup.select("div.fiction-list-item h2.fiction-title a")

    for link in fiction_links[:count * 2]:
        if len(urls) >= count:
            break
        fiction_url = "https://www.royalroad.com" + link["href"]
        polite_delay()
        fiction_html, _ = fetch_page(session, fiction_url)
        if not fiction_html:
            continue

        fsoup = BeautifulSoup(fiction_html, "lxml")
        # Find first chapter
        chapter_link = fsoup.select_one("table#chapters td.chapter-name a")
        if not chapter_link:
            # Fallback: any chapter link pattern
            chapter_link = fsoup.select_one(
                "a[href*='/fiction/'][href*='/chapter/']"
            )
        if chapter_link:
            href = chapter_link.get("href", "")
            if href.startswith("/"):
                href = "https://www.royalroad.com" + href
            urls.append(href)

    return urls[:count]


def sample_syosetu(
    session: requests.Session,
    count: int,
) -> list[str]:
    """Sample top-ranked works from Syosetu daily ranking. Returns first chapter URLs."""
    urls = []
    ranking_url = "https://yomou.syosetu.com/rank/list/type/daily_total/"
    log.info(f"SAMPLE  Syosetu ranking: {ranking_url}")
    html, status = fetch_page(session, ranking_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    # Each ranked work has a link pattern: /n{code}/
    novel_links = soup.select("div.ranking-box h3 a, div.rank_h a[href*='ncode.syosetu.com']")

    for link in novel_links[:count * 2]:
        if len(urls) >= count:
            break
        href = link.get("href", "")
        if not href:
            continue
        if not href.startswith("http"):
            href = "https://ncode.syosetu.com" + href

        polite_delay()
        index_html, _ = fetch_page(session, href)
        if not index_html:
            continue

        isoup = BeautifulSoup(index_html, "lxml")
        # First chapter is in index_box as the first dt > a
        first_chapter = isoup.select_one("div.index_box dt a")
        if first_chapter:
            chref = first_chapter.get("href", "")
            if not chref.startswith("http"):
                base = href.rstrip("/")
                chref = base + "/" + chref.lstrip("/")
            urls.append(chref)

    return urls[:count]


def resolve_urls(
    session: requests.Session,
    source_id: str,
    config: dict,
    dry_run: bool = False,
) -> list[str]:
    """Return list of URLs to scrape. Calls sampler if urls list is empty."""
    static_urls = config.get("urls", [])
    if static_urls:
        return static_urls

    strategy = config.get("sample_strategy")
    count = config.get("sample_count", 10)
    if dry_run:
        log.info(f"DRY-RUN SAMPLE  {source_id} | strategy={strategy} | count={count}")
        return [f"[would sample {count} URLs from {strategy}]"]

    if strategy == "ao3":
        return sample_ao3(session, config.get("sample_tags", []), count)
    elif strategy == "royalroad":
        return sample_royalroad(session, config.get("sample_tags", []), count)
    elif strategy == "syosetu":
        return sample_syosetu(session, count)
    else:
        log.warning(f"No sample strategy for {source_id}")
        return []

# ---------------------------------------------------------------------------
# OUTPUT / STATE
# ---------------------------------------------------------------------------

def _url_to_filename(url: str) -> str:
    """Convert URL to a safe filename (without extension)."""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip("/").replace("/", "_")
    if not path:
        path = parsed.netloc.replace(".", "_")
    # Sanitize
    path = re.sub(r"[^\w\-]", "_", path)
    # Limit length
    if len(path) > 80:
        path = path[:72] + "_" + hashlib.md5(url.encode()).hexdigest()[:7]
    return path or hashlib.md5(url.encode()).hexdigest()[:12]


def save_raw(
    text: str,
    url: str,
    category: str,
    source_id: str,
    config: dict,
    http_status: int,
    base_dir: pathlib.Path = pathlib.Path("data/raw"),
) -> pathlib.Path:
    """Write text + JSON metadata sidecar. Returns path to .txt file."""
    out_dir = base_dir / category / source_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = _url_to_filename(url)
    txt_path = out_dir / f"{fname}.txt"
    json_path = out_dir / f"{fname}.json"

    txt_path.write_text(text, encoding="utf-8")

    word_count = len(text.split())
    char_count = len(text)
    metadata = {
        "url": url,
        "source_id": source_id,
        "category": category,
        "extractor": config.get("extractor", "generic"),
        "language": config.get("language", "en"),
        "display_name": config.get("display_name", source_id),
        "tags": config.get("tags", []),
        "http_status": http_status,
        "word_count": word_count,
        "char_count": char_count,
        "scraped_at": datetime.datetime.now().isoformat(),
        "output_file": str(txt_path),
        "checksum_md5": hashlib.md5(text.encode("utf-8")).hexdigest(),
    }
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return txt_path


STATE_PATH = pathlib.Path("data/scrape_state.json")


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"version": 1, "scraped_urls": {}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def is_already_scraped(state: dict, url: str) -> bool:
    entry = state.get("scraped_urls", {}).get(url, {})
    return entry.get("status") == "ok"


def update_state(state: dict, url: str, status: str, http_code: int, txt_path: Optional[pathlib.Path], word_count: int, char_count: int) -> None:
    entry: dict = {
        "status": status,
        "http_code": http_code,
        "scraped_at": datetime.datetime.now().isoformat(),
    }
    if txt_path:
        entry["output_file"] = str(txt_path)
        entry["word_count"] = word_count
        entry["char_count"] = char_count
    state.setdefault("scraped_urls", {})[url] = entry

# ---------------------------------------------------------------------------
# MAIN SCRAPE LOGIC
# ---------------------------------------------------------------------------

def scrape_url(
    session: requests.Session,
    url: str,
    extractor_name: str,
    respect_robots: bool,
) -> tuple[Optional[str], int]:
    """Fetch URL and extract clean text. Returns (text_or_none, http_status)."""
    log.info(f"FETCH   {url}")
    html, status = fetch_page(session, url, respect_robots)
    if not html:
        return None, status

    soup = BeautifulSoup(html, "lxml")
    extractor = EXTRACTOR_MAP.get(extractor_name, extract_generic)
    text = extractor(soup)

    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or None, status


def run_source(
    session: requests.Session,
    category: str,
    source_id: str,
    config: dict,
    state: dict,
    force: bool,
    respect_robots: bool,
    dry_run: bool,
) -> None:
    log.info(f"=== {category.upper()} / {source_id} ===")
    urls = resolve_urls(session, source_id, config, dry_run)

    if not urls:
        log.warning(f"SKIP    {source_id} | no URLs resolved")
        return

    extractor_name = config.get("extractor", "generic")
    stored = 0
    skipped = 0
    failed = 0

    for url in urls:
        if dry_run:
            log.info(f"DRY-RUN WOULD_FETCH  {url}")
            continue

        if not force and is_already_scraped(state, url):
            log.info(f"SKIP    {url} | already scraped")
            skipped += 1
            continue

        text, status = scrape_url(session, url, extractor_name, respect_robots)
        polite_delay()

        if not text:
            log.warning(f"EMPTY   {url} | status={status} | words=0")
            update_state(state, url, "failed", status, None, 0, 0)
            failed += 1
            save_state(state)
            continue

        word_count = len(text.split())
        char_count = len(text)

        txt_path = save_raw(text, url, category, source_id, config, status)
        log.info(
            f"STORED  {txt_path} | status={status} | words={word_count} | chars={char_count}"
        )
        update_state(state, url, "ok", status, txt_path, word_count, char_count)
        stored += 1
        save_state(state)

    log.info(
        f"DONE    {source_id} | stored={stored} skipped={skipped} failed={failed}"
    )


def run_category(
    session: requests.Session,
    category: str,
    targets: dict,
    state: dict,
    force: bool,
    respect_robots: bool,
    dry_run: bool,
    sources_filter: Optional[list[str]] = None,
) -> None:
    for source_id, config in targets.items():
        if sources_filter and source_id not in sources_filter:
            continue
        run_source(session, category, source_id, config, state, force, respect_robots, dry_run)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape TRPG wikis and web novel platforms for LLM training data."
    )
    parser.add_argument(
        "--category", choices=["trpg", "webnovel"],
        help="Limit to one category."
    )
    parser.add_argument(
        "--sources", nargs="*",
        help="Specific source IDs to scrape (e.g. pathfinder wh40k)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-scrape even if already in state."
    )
    parser.add_argument(
        "--no-robots", action="store_true",
        help="Skip robots.txt checks."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be scraped without making HTTP requests."
    )
    args = parser.parse_args()

    session = build_session()
    state = load_state()
    respect_robots = not args.no_robots
    sources_filter = args.sources

    categories_to_run = (
        [args.category] if args.category
        else list(CRAWL_TARGETS.keys())
    )

    for cat in categories_to_run:
        if cat not in CRAWL_TARGETS:
            log.warning(f"Unknown category: {cat}")
            continue
        run_category(
            session,
            cat,
            CRAWL_TARGETS[cat],
            state,
            force=args.force,
            respect_robots=respect_robots,
            dry_run=args.dry_run,
            sources_filter=sources_filter,
        )

    log.info("All done.")


if __name__ == "__main__":
    main()
