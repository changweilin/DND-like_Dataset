import sys
import yaml
import requests
import cloudscraper
from bs4 import BeautifulSoup
from ruamel.yaml import YAML
import time
import urllib.parse
from pathlib import Path

# ── API 用量追蹤 ───────────────────────────────────────────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from api_tracker import check_quota, record_call as _record_api
    _API_TRACKER = True
except ImportError:
    _API_TRACKER = False
    def check_quota(api, **kw): return True
    def _record_api(api): pass

CONFIG_PATH = r"c:\Users\user\Documents\Python\DND-like_Dataset\scraper_config.yaml"

class DiscoveryAgent:
    def __init__(self, config_path):
        self.config_path = config_path
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = self.yaml.load(f)
        self.known_urls = self._build_known_urls()

        # Initialize a session based on config
        if self.config.get('http', {}).get('use_cloudscraper', False):
            self.session = cloudscraper.create_scraper()
            print("[i] Cloudscraper is enabled.")
        else:
            self.session = requests.Session()
            print("[i] Using standard requests session.")
        
        # Set a default user-agent
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        })

    def _build_known_urls(self):
        """建立全域已存在網址集合，用於去重"""
        urls = set()
        self.domain_to_source = {}  # 用於追蹤哪個網域屬於哪個 source_id
        for cat_name, category in self.config.get('sources', {}).items():
            for src_id, source in category.items():
                for url in source.get('urls', []):
                    clean_url = url.rstrip('/')
                    urls.add(clean_url)
                    domain = urllib.parse.urlparse(clean_url).netloc
                    if domain:
                        self.domain_to_source[domain] = (cat_name, src_id)
        return urls

    def get_search_keywords(self):
        """從現有配置提取標籤作為搜尋種子"""
        all_tags = set()
        for category in self.config['sources'].values():
            for source in category.values():
                all_tags.update(source.get('tags', []))
        return list(all_tags)

    def discover_new_links(self, keyword):
        """
        呼叫 Google Custom Search API 尋找新的 Wiki 分類頁面。
        """
        api_key = self.config.get('api', {}).get('google_search_key')
        cx = self.config.get('api', {}).get('google_search_cx')

        if not api_key or "YOUR_" in api_key:
            print("[!] 錯誤: 未設定有效的 Google API Key。跳過搜尋。")
            return []

        # 建立針對 Wiki 分類頁面的搜尋語句
        query = f"{keyword} wiki category lore"
        api_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': api_key,
            'cx': cx,
            'q': query,
            'num': 10  # 每次搜尋取前 10 筆結果
        }

        try:
            if not check_quota("google-custom-search", raise_on_exceed=False):
                print("[!] Google Custom Search 今日配額已滿 (100/day)，跳過搜尋。")
                return []
            _record_api("google-custom-search")
            print(f"[*] 正在透過 Google API 搜尋關鍵字: {keyword}")
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # 提取連結
            links = [item['link'] for item in data.get('items', [])]
            return links
        except Exception as e:
            print(f"[!] Google Search API 調用失敗: {e}")
            return []

    def analyze_site_type(self, url):
        """
        分析網站結構以決定 extractor
        """
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            generator_tag = soup.find("meta", {"name": "generator"})
            
            if "fandom.com" in url:
                return "fandom"
            elif generator_tag and 'content' in generator_tag.attrs and generator_tag['content'].startswith("MediaWiki"):
                return "mediawiki"
            else:
                return "generic"
        except Exception as e:
            print(f"[!] 分析網站類型時發生錯誤: {e}")
            return None

    def crawl_fandom_category(self, category_url):
        """
        自動抓取 Fandom 分類頁面下的所有條目網址，支援分頁。
        例如: https://forgottenrealms.fandom.com/wiki/Category:Deities
        """
        found_links = []
        next_url = category_url
        while next_url:
            try:
                print(f"[*] 正在掃描分類頁面: {next_url}")
                response = self.session.get(next_url, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # 1. 抓取成員連結 (Fandom 標準 CSS class 為 .category-page__member-link)
                members = soup.select(".category-page__member-link")
                for member in members:
                    href = member.get("href")
                    if href and "/wiki/" in href:
                        # 過濾掉非條目頁面 (例如 Category:, File:, Template: 等包含冒號的頁面)
                        page_title = href.split("/wiki/")[-1]
                        if ":" not in page_title:
                            full_url = urllib.parse.urljoin(next_url, href)
                            found_links.append(full_url)

                # 2. 檢查分頁 (尋找「下一頁」按鈕)
                next_btn = soup.select_one(".category-page__pagination-next")
                next_url = next_btn.get("href") if next_btn and next_btn.get("href") else None
                
                if next_url:
                    time.sleep(1) # 禮貌延遲，避免被阻擋
            except Exception as e:
                print(f"[!] 抓取分類時發生錯誤: {e}")
                break

        return list(dict.fromkeys(found_links)) # 去重並保持順序

    def crawl_mediawiki_category(self, category_url):
        """
        自動抓取標準 MediaWiki 分類頁面下的所有條目網址。
        例如: https://pathfinderwiki.com/wiki/Category:Golarion
        """
        found_links = []
        next_url = category_url
        while next_url:
            try:
                print(f"[*] 正在掃描 MediaWiki 分類: {next_url}")
                response = self.session.get(next_url, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # 1. 在 MediaWiki 中，條目連結通常位於 id="mw-pages" 的 div 內
                pages_div = soup.find("div", id="mw-pages")
                if pages_div:
                    for link in pages_div.find_all("a"):
                        href = link.get("href")
                        if href and "/wiki/" in href:
                            # 過濾掉非內容頁面 (排除包含 ':' 的 Namespace)
                            page_title = href.split("/wiki/")[-1]
                            if ":" not in page_title:
                                full_url = urllib.parse.urljoin(next_url, href)
                                found_links.append(full_url)

                # 2. 尋找「下一頁」連結，MediaWiki 通常使用 class="mw-nextlink"
                next_btn = soup.select_one(".mw-nextlink")
                next_url = urllib.parse.urljoin(next_url, next_btn.get("href")) if next_btn else None
                
                if next_url:
                    time.sleep(1)
            except Exception as e:
                print(f"[!] 抓取 MediaWiki 分類時發生錯誤: {e}")
                break

        return list(dict.fromkeys(found_links))

    def _generate_source_id(self, url):
        """從網址生成簡單的 source_id (例如 pathfinderwiki.com -> pathfinder)"""
        domain = urllib.parse.urlparse(url).netloc
        name = domain.replace('www.', '').split('.')[0]
        return name.lower().replace('-', '_')

    def update_config(self, category_name, source_id, data):
        """
        更新 YAML 配置。如果 source_id 已存在，則合併新發現的 URLs。
        """
        sources = self.config['sources'].get(category_name, {})
        new_urls_count = 0

        if source_id in sources:
            # 增量更新：將不重複的網址加入現有列表
            existing_urls = set(sources[source_id].get('urls', []))
            for u in data['urls']:
                if u.rstrip('/') not in self.known_urls:
                    sources[source_id]['urls'].append(u)
                    self.known_urls.add(u.rstrip('/'))
                    new_urls_count += 1
        else:
            # 全新新增
            sources[source_id] = data
            self.known_urls.update([u.rstrip('/') for u in data['urls']])
            new_urls_count = len(data['urls'])

        if new_urls_count > 0:
            self.config['sources'][category_name] = sources
            with open(self.config_path, 'w', encoding='utf-8') as f:
                self.yaml.dump(self.config, f)
            print(f"[+] {source_id} 已更新，新增了 {new_urls_count} 個網址。")

    def run_cycle(self):
        """執行完整的發現循環"""
        keywords = self.get_search_keywords()
        print(f"[*] 開始自動發現循環，標籤種子數量: {len(keywords)}")
        
        for kw in keywords:
            # 1. 發現潛在的分類頁面連結
            potential_category_urls = self.discover_new_links(kw)
            
            for url in potential_category_urls:
                # 2. 評估網站類型
                site_type = self.analyze_site_type(url)
                found_links = []

                # 3. 根據類型調用對應的爬取函數
                if site_type == "fandom":
                    found_links = self.crawl_fandom_category(url)
                elif site_type == "mediawiki":
                    found_links = self.crawl_mediawiki_category(url)

                # 4. 如果有發現條目，更新設定檔
                if found_links:
                    src_id = self._generate_source_id(url)
                    new_entry = {
                        "display_name": f"{src_id.capitalize()} Lore",
                        "extractor": site_type,
                        "language": "en",
                        "tags": [kw],
                        "urls": found_links
                    }
                    self.update_config("trpg", src_id, new_entry)

if __name__ == "__main__":
    agent = DiscoveryAgent(CONFIG_PATH)
    # agent.run_cycle()
    print("Agent 初始化完成，準備掃描新數據源。")