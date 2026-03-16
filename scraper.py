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
import collections
import datetime
import hashlib
import html as _html
import itertools
import json
import logging
import os
import pathlib
import random
import re
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import urllib.robotparser
from typing import Callable, Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import cloudscraper as _cloudscraper
    _CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    _CLOUDSCRAPER_AVAILABLE = False

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# CONFIG FILE
# ---------------------------------------------------------------------------

CONFIG_PATH = pathlib.Path("scraper_config.yaml")

# Populated at startup by load_config() called from main().
# Contains the full parsed YAML dict.
_CFG: dict = {}

# Populated from _CFG["sources"] by main(); consumed by run_category() etc.
CRAWL_TARGETS: dict[str, dict] = {}


def load_config(path: pathlib.Path = CONFIG_PATH) -> dict:
    """Load scraper_config.yaml and return the full config dict.

    Exits with a helpful error if pyyaml is not installed or the file is
    missing.  Call this once at startup from main().
    """
    if not _YAML_AVAILABLE:
        print(
            "ERROR: pyyaml is required to read scraper_config.yaml.\n"
            "       Run:  pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(1)
    if not path.exists():
        print(
            f"ERROR: Config file not found: {path}\n"
            f"       Expected a scraper_config.yaml in the current directory.\n"
            f"       You can copy the default from the repository root.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        with open(path, encoding="utf-8") as fh:
            return _yaml.safe_load(fh) or {}
    except Exception as exc:
        print(f"ERROR: Failed to parse {path}: {exc}", file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# HTTP HEADERS (reused from crawl_world_lore.py — proven effective)
# ---------------------------------------------------------------------------

# Base headers: keys not present in a UA profile entry are always included.
_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
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
# USER-AGENT POOL
# Each entry overrides the matching keys in _HEADERS for the session lifetime.
# Firefox and Safari entries omit Sec-CH-UA (those browsers don't send it).
# Safari entries use a narrower Accept header matching real Safari behaviour.
# ---------------------------------------------------------------------------

_UA_POOL: list[dict] = [
    {   # Chrome 124 / Windows 10
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-CH-UA": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
    },
    {   # Chrome 122 / macOS 14
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-CH-UA": '"Chromium";v="122", "Google Chrome";v="122", "Not-A.Brand";v="99"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"macOS"',
    },
    {   # Firefox 124 / Windows 10
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    },
    {   # Firefox 123 / macOS 14
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:123.0) Gecko/20100101 Firefox/123.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    },
    {   # Edge 122 / Windows 11
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-CH-UA": '"Chromium";v="122", "Microsoft Edge";v="122", "Not-A.Brand";v="99"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
    },
    {   # Safari 17.3 / macOS 14
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    },
    {   # Chrome 121 / Linux
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-CH-UA": '"Chromium";v="121", "Google Chrome";v="121", "Not-A.Brand";v="99"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Linux"',
    },
    {   # Firefox 122 / Linux
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    },
    {   # Edge 121 / Windows 10
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.2277.128",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-CH-UA": '"Chromium";v="121", "Microsoft Edge";v="121", "Not-A.Brand";v="99"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
    },
    {   # Chrome 120 / Windows 10 (original profile, kept for diversity)
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7,ja;q=0.6",
        "Sec-CH-UA": '"Chromium";v="120", "Google Chrome";v="120", "Not-A.Brand";v="99"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
    },
]


def _pick_ua_profile() -> dict:
    """Pick one UA profile at random. Called once per build_session()."""
    return random.choice(_UA_POOL)


def _build_referer(url: str) -> str:
    """Return a plausible Referer for url, simulating natural site navigation."""
    netloc = urllib.parse.urlparse(url).netloc
    if "fandom.com" in netloc:
        return f"https://{netloc}/wiki/"
    if "lexicanum.com" in netloc:
        return f"https://{netloc}/wiki/Main_Page"
    if "royalroad.com" in netloc:
        return "https://www.royalroad.com/"
    if "archiveofourown.org" in netloc:
        return "https://archiveofourown.org/"
    if "syosetu.com" in netloc:
        return "https://yomou.syosetu.com/"
    return f"https://{netloc}/"

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

def build_session(use_cloudscraper: bool = True) -> requests.Session:
    ua_profile = _pick_ua_profile()

    if use_cloudscraper and _CLOUDSCRAPER_AVAILABLE:
        # cloudscraper is a requests.Session subclass that solves Cloudflare
        # JS challenges automatically (used by Fandom wikis, RoyalRoad, etc.)
        session = _cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
    else:
        session = requests.Session()

    # Overlay UA profile on top of base headers
    session.headers.update({**_HEADERS, **ua_profile})

    _http = _CFG.get("http", {})
    retry = Retry(
        total=_http.get("retry_total", 3),
        backoff_factor=_http.get("backoff_factor", 1),
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Restore cookies saved from previous runs
    for c in load_cookies():
        session.cookies.set(
            c["name"], c["value"],
            domain=c.get("domain", ""),
            path=c.get("path", "/"),
        )

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
    # Use "*" so the check applies to the most permissive robots.txt rule;
    # version-specific UA strings rarely appear in Disallow directives.
    return rp.can_fetch("*", url)


def _fetch_with_curl(url: str, referer: Optional[str] = None) -> Optional[str]:
    """Fallback using system curl, which often bypasses TLS fingerprinting."""
    try:
        cmd = [
            "curl", "-s", "-L",
            "-H", f"User-Agent: {random.choice(_UA_POOL)['User-Agent']}",
            "-H", f"Accept: {_HEADERS['Accept']}",
            "-H", "Accept-Language: en-US,en;q=0.9",
        ]
        if referer:
            cmd += ["-H", f"Referer: {referer}"]
        cmd += ["--max-time", "20", url]
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
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
) -> tuple[Optional[str], int, dict]:
    """Fetch a URL. Returns (html_text, http_status, cache_headers). Never raises.

    cache_headers = {"etag": str|None, "last_modified": str|None}
    On 304 Not Modified, html_text is None and http_status is 304.
    """
    if respect_robots and not check_robots(url):
        log.warning(f"ROBOTS_BLOCKED {url}")
        return None, 0, {}

    _no_cache: dict = {"etag": None, "last_modified": None}
    referer = _build_referer(url)

    # Build per-request headers: conditional GET validators + Referer
    req_headers: dict = {"Referer": referer}
    if etag:
        req_headers["If-None-Match"] = etag
    if last_modified:
        req_headers["If-Modified-Since"] = last_modified

    try:
        _timeout = _CFG.get("http", {}).get("timeout", 15)
        resp = session.get(url, timeout=_timeout, headers=req_headers)

        # 304 Not Modified — content unchanged since last visit
        if resp.status_code == 304:
            return None, 304, _no_cache

        if resp.status_code == 403:
            html = _fetch_with_curl(url, referer=referer)
            if html:
                return html, 200, _no_cache
        resp.raise_for_status()
        # Handle Shift-JIS and other encodings (important for Syosetu)
        resp.encoding = resp.apparent_encoding
        cache_headers = {
            "etag": resp.headers.get("ETag"),
            "last_modified": resp.headers.get("Last-Modified"),
        }
        return resp.text, resp.status_code, cache_headers
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        html = _fetch_with_curl(url, referer=referer)
        if html:
            return html, 200, _no_cache
        log.warning(f"FAILED  {url} | status={status} | reason={e}")
        return None, status, _no_cache
    except Exception as e:
        html = _fetch_with_curl(url, referer=referer)
        if html:
            return html, 200, _no_cache
        log.warning(f"FAILED  {url} | reason={e}")
        return None, 0, _no_cache


def polite_delay(min_s: float = 1.0, max_s: float = 2.0) -> None:
    time.sleep(random.uniform(min_s, max_s))


_DOMAIN_LAST_HIT: dict[str, float] = {}


def domain_polite_delay(url: str, min_s: float = 3.0, max_s: float = 6.0) -> None:
    """Per-domain rate limiter with random jitter.

    Ensures at least min_s seconds have elapsed since the last request to the
    same netloc. The random upper bound (max_s) prevents perfectly regular
    request intervals that automated systems can fingerprint.
    Call this immediately before each fetch.
    """
    domain = urllib.parse.urlparse(url).netloc
    elapsed = time.time() - _DOMAIN_LAST_HIT.get(domain, 0.0)
    target = random.uniform(min_s, max_s)
    if elapsed < target:
        time.sleep(target - elapsed)
    _DOMAIN_LAST_HIT[domain] = time.time()

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
# TEXT CLEANING (post-extraction)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Post-extraction text cleaning pipeline (no LLM required):
    1. HTML entity decode — catches any &amp; &lt; &#160; that BS4 left behind
    2. Unicode NFKC normalization — fullwidth ASCII→normal, ligatures→letters,
       non-breaking spaces→space, smart quotes→straight, etc.
    3. Collapse runs of spaces within a line (but preserve newlines)
    4. Collapse 3+ consecutive blank lines to 2
    5. Strip leading/trailing whitespace
    """
    # 1. HTML entity decode (belt-and-suspenders; BS4 handles most but not all)
    text = _html.unescape(text)

    # 2. Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # 3. Collapse intra-line whitespace runs (spaces/tabs) to single space
    lines = []
    for line in text.splitlines():
        lines.append(re.sub(r"[ \t]+", " ", line).strip())
    text = "\n".join(lines)

    # 4. Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Strip
    return text.strip()

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
        html, status, _ = fetch_page(session, listing_url)
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
    html, status, _ = fetch_page(session, listing_url)
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
    html, status, _ = fetch_page(session, ranking_url)
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
COOKIES_PATH = pathlib.Path("data/cookies.json")


def load_cookies() -> list[dict]:
    """Load persisted cookies, discarding any that have already expired."""
    if not COOKIES_PATH.exists():
        return []
    try:
        raw: list[dict] = json.loads(COOKIES_PATH.read_text(encoding="utf-8"))
        now = time.time()
        return [c for c in raw if c.get("expires") is None or c["expires"] > now]
    except Exception:
        return []


def save_cookies(session: requests.Session) -> None:
    """Persist all session cookies to disk (atomic write)."""
    cookies = []
    for cookie in session.cookies:
        cookies.append({
            "name": cookie.name,
            "value": cookie.value,
            "domain": cookie.domain,
            "path": cookie.path,
            "secure": cookie.secure,
            "expires": cookie.expires,
        })
    if not cookies:
        return
    COOKIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = COOKIES_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cookies, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, COOKIES_PATH)
    log.debug(f"Saved {len(cookies)} cookie(s) to {COOKIES_PATH}")


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


def update_state(
    state: dict,
    url: str,
    status: str,
    http_code: int,
    txt_path: Optional[pathlib.Path],
    word_count: int,
    char_count: int,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
    content_checksum: Optional[str] = None,
) -> None:
    entry: dict = {
        "status": status,
        "http_code": http_code,
        "scraped_at": datetime.datetime.now().isoformat(),
    }
    if txt_path:
        entry["output_file"] = str(txt_path)
        entry["word_count"] = word_count
        entry["char_count"] = char_count
    if etag:
        entry["etag"] = etag
    if last_modified:
        entry["last_modified"] = last_modified
    if content_checksum:
        entry["content_checksum"] = content_checksum
    state.setdefault("scraped_urls", {})[url] = entry

# ---------------------------------------------------------------------------
# MAIN SCRAPE LOGIC
# ---------------------------------------------------------------------------

def scrape_url(
    session: requests.Session,
    url: str,
    extractor_name: str,
    respect_robots: bool,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
) -> tuple[Optional[str], int, dict, Optional[str]]:
    """Fetch URL and extract clean text.

    Returns (text_or_none, http_status, cache_headers, content_checksum).
    text_or_none is None on failure *and* on 304 (check http_status to distinguish).
    """
    log.info(f"FETCH   {url}")
    html, status, cache_headers = fetch_page(session, url, respect_robots, etag, last_modified)
    if not html:
        return None, status, cache_headers, None

    soup = BeautifulSoup(html, "lxml")
    extractor = EXTRACTOR_MAP.get(extractor_name, extract_generic)
    text = clean_text(extractor(soup))
    if not text:
        return None, status, cache_headers, None
    checksum = hashlib.md5(text.encode("utf-8")).hexdigest()
    return text, status, cache_headers, checksum


def _process_url(
    session: requests.Session,
    category: str,
    source_id: str,
    config: dict,
    url: str,
    state: dict,
    respect_robots: bool,
    update_check: bool,
    counters: dict,
) -> None:
    """Fetch, evaluate, and persist one URL. Updates counters dict in-place.

    Applies domain_polite_delay() before fetching, so callers must NOT add
    their own sleep.
    """
    prev_entry = state.get("scraped_urls", {}).get(url, {})
    already_scraped = prev_entry.get("status") == "ok"

    cond_etag = prev_entry.get("etag") if (update_check and already_scraped) else None
    cond_lm = prev_entry.get("last_modified") if (update_check and already_scraped) else None
    prev_checksum = prev_entry.get("content_checksum") if (update_check and already_scraped) else None

    extractor_name = config.get("extractor", "generic")

    _http = _CFG.get("http", {})
    domain_polite_delay(
        url,
        min_s=config.get("min_delay", _http.get("min_delay", 3.0)),
        max_s=config.get("max_delay", _http.get("max_delay", 6.0)),
    )
    text, status, cache_hdrs, checksum = scrape_url(
        session, url, extractor_name, respect_robots, cond_etag, cond_lm
    )

    if status == 304:
        log.info(f"UNCHANGED (304)  {url}")
        counters["unchanged"] += 1
        return

    if not text:
        log.warning(f"EMPTY   {url} | status={status} | words=0")
        if not already_scraped:
            update_state(state, url, "failed", status, None, 0, 0)
            save_state(state)
        counters["failed"] += 1
        return

    if update_check and already_scraped and checksum == prev_checksum:
        log.info(f"UNCHANGED (md5)  {url}")
        counters["unchanged"] += 1
        return

    word_count = len(text.split())
    char_count = len(text)
    txt_path = save_raw(text, url, category, source_id, config, status)

    if update_check and already_scraped:
        log.info(f"UPDATED  {txt_path} | status={status} | words={word_count} | chars={char_count}")
        counters["updated"] += 1
    else:
        log.info(f"STORED  {txt_path} | status={status} | words={word_count} | chars={char_count}")
        counters["stored"] += 1

    update_state(
        state, url, "ok", status, txt_path, word_count, char_count,
        etag=cache_hdrs.get("etag"),
        last_modified=cache_hdrs.get("last_modified"),
        content_checksum=checksum,
    )
    save_state(state)
    save_cookies(session)


def _new_counters() -> dict:
    return {"stored": 0, "skipped": 0, "failed": 0, "unchanged": 0, "updated": 0}


def run_source(
    session: requests.Session,
    category: str,
    source_id: str,
    config: dict,
    state: dict,
    force: bool,
    respect_robots: bool,
    dry_run: bool,
    update_check: bool = False,
) -> None:
    log.info(f"=== {category.upper()} / {source_id} ===")
    urls = resolve_urls(session, source_id, config, dry_run)

    if not urls:
        log.warning(f"SKIP    {source_id} | no URLs resolved")
        return

    counters = _new_counters()
    for url in urls:
        if dry_run:
            log.info(f"DRY-RUN WOULD_FETCH  {url}")
            continue

        prev_entry = state.get("scraped_urls", {}).get(url, {})
        if prev_entry.get("status") == "ok" and not force and not update_check:
            log.info(f"SKIP    {url} | already scraped")
            counters["skipped"] += 1
            continue

        _process_url(session, category, source_id, config, url, state, respect_robots, update_check, counters)

    summary = f"stored={counters['stored']} skipped={counters['skipped']} failed={counters['failed']}"
    if update_check:
        summary += f" unchanged={counters['unchanged']} updated={counters['updated']}"
    log.info(f"DONE    {source_id} | {summary}")


def run_category(
    session: requests.Session,
    category: str,
    targets: dict,
    state: dict,
    force: bool,
    respect_robots: bool,
    dry_run: bool,
    sources_filter: Optional[list[str]] = None,
    update_check: bool = False,
) -> None:
    for source_id, config in targets.items():
        if sources_filter and source_id not in sources_filter:
            continue
        run_source(session, category, source_id, config, state, force, respect_robots, dry_run, update_check)


def run_all_interleaved(
    session: requests.Session,
    categories_and_targets: list[tuple[str, dict]],
    state: dict,
    force: bool,
    respect_robots: bool,
    dry_run: bool,
    sources_filter: Optional[list[str]],
    update_check: bool = False,
) -> None:
    """Collect all pending URLs from every source, then process them in
    round-robin domain order so no single server receives consecutive requests.

    Phase 1: resolve all URL lists (may call samplers for AO3/RoyalRoad/Syosetu)
    Phase 2: bucket URLs by netloc, interleave with itertools.zip_longest
    Phase 3: process the interleaved list through _process_url()
    """
    # --- Phase 1: resolve ---
    log.info("=== INTERLEAVED MODE: resolving work lists ===")
    all_items: list[tuple[str, str, dict, str]] = []  # (category, source_id, config, url)
    for category, targets in categories_and_targets:
        for source_id, config in targets.items():
            if sources_filter and source_id not in sources_filter:
                continue
            urls = resolve_urls(session, source_id, config, dry_run)
            for url in urls:
                if dry_run:
                    log.info(f"DRY-RUN WOULD_FETCH  {url}")
                    continue
                prev_entry = state.get("scraped_urls", {}).get(url, {})
                if prev_entry.get("status") == "ok" and not force and not update_check:
                    log.debug(f"SKIP    {url} | already scraped")
                    continue
                all_items.append((category, source_id, config, url))

    if dry_run or not all_items:
        return

    # --- Phase 2: bucket by domain, round-robin interleave ---
    buckets: dict[str, list] = collections.defaultdict(list)
    for item in all_items:
        domain = urllib.parse.urlparse(item[3]).netloc
        buckets[domain].append(item)

    interleaved: list[tuple] = [
        x
        for row in itertools.zip_longest(*buckets.values())
        for x in row
        if x is not None
    ]

    log.info(
        f"=== INTERLEAVED MODE: {len(interleaved)} URLs across "
        f"{len(buckets)} domains ==="
    )
    for domain, items in buckets.items():
        log.info(f"    {domain}: {len(items)} URL(s)")

    # --- Phase 3: process ---
    counters = _new_counters()
    for category, source_id, config, url in interleaved:
        _process_url(session, category, source_id, config, url, state, respect_robots, update_check, counters)

    summary = f"stored={counters['stored']} skipped={counters['skipped']} failed={counters['failed']}"
    if update_check:
        summary += f" unchanged={counters['unchanged']} updated={counters['updated']}"
    log.info(f"=== INTERLEAVED DONE | {summary} ===")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape TRPG wikis and web novel platforms for LLM training data."
    )
    parser.add_argument(
        "--config", type=pathlib.Path, default=CONFIG_PATH, metavar="FILE",
        help=f"Path to scraper_config.yaml (default: {CONFIG_PATH}).",
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
    parser.add_argument(
        "--update-check", action="store_true",
        help=(
            "Re-visit already-scraped URLs to detect changes. "
            "Uses HTTP conditional GET (ETag/Last-Modified) when available; "
            "falls back to content MD5 comparison. "
            "Only saves files that have actually changed."
        ),
    )
    parser.add_argument(
        "--interleave", action="store_true",
        help=(
            "Round-robin between sources instead of exhausting one site at a "
            "time. Reduces the risk of per-IP rate-limit blocks by spreading "
            "requests across domains. Combines with --update-check and --force."
        ),
    )
    args = parser.parse_args()

    # --- Load config and apply to module-level globals ---
    global _CFG, CRAWL_TARGETS, STATE_PATH, COOKIES_PATH
    _CFG = load_config(args.config)
    CRAWL_TARGETS = _CFG.get("sources", {})
    _paths = _CFG.get("paths", {})
    if "state_path" in _paths:
        STATE_PATH = pathlib.Path(_paths["state_path"])
    if "cookies_path" in _paths:
        COOKIES_PATH = pathlib.Path(_paths["cookies_path"])

    log.info(
        f"Config loaded from {args.config} | "
        f"categories={list(CRAWL_TARGETS)} | "
        f"sources={sum(len(v) for v in CRAWL_TARGETS.values())}"
    )

    _http = _CFG.get("http", {})
    session = build_session(use_cloudscraper=_http.get("use_cloudscraper", True))
    state = load_state()
    respect_robots = not args.no_robots
    sources_filter = args.sources

    categories_to_run = (
        [args.category] if args.category
        else list(CRAWL_TARGETS.keys())
    )

    if args.interleave:
        cats = [
            (cat, CRAWL_TARGETS[cat])
            for cat in categories_to_run
            if cat in CRAWL_TARGETS
        ]
        run_all_interleaved(
            session, cats, state,
            force=args.force,
            respect_robots=respect_robots,
            dry_run=args.dry_run,
            sources_filter=sources_filter,
            update_check=args.update_check,
        )
    else:
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
                update_check=args.update_check,
            )

    save_cookies(session)
    log.info("All done.")


if __name__ == "__main__":
    main()
