from __future__ import annotations

import re
import ssl
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
SKIP_EXT = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js", ".zip", ".mp4", ".woff", ".woff2"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = False
        self.parts: list[str] = []
        self.title = ""
        self.links: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = True
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = False
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title and not self.title:
            self.title = text
        self.parts.append(text)


def extract_url(message: str) -> str:
    match = URL_RE.search(message or "")
    return match.group(0).rstrip(".,);") if match else ""


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl._create_unverified_context()


def _host(url: str) -> str:
    return urlparse(url).hostname.replace("www.", "") if urlparse(url).hostname else ""


def _allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    path = parsed.path.lower()
    return not any(path.endswith(ext) for ext in SKIP_EXT)


def _fetch(url: str) -> str:
    req = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 AI-Sales-Agent-Ingest"},
    )
    with urlopen(req, timeout=12, context=_ssl_context()) as resp:
        raw = resp.read(400_000)
        ctype = resp.headers.get("Content-Type", "")
        final = resp.geturl()
    if "html" not in ctype.lower() and not final.lower().endswith((".html", ".htm", "/")):
        return "", final, []
    parser = _TextExtractor()
    try:
        parser.feed(raw.decode("utf-8", errors="ignore"))
    except Exception:
        return "", final, []
    text = "\n".join(parser.parts)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    links = []
    for href in parser.links:
        try:
            links.append(urldefrag(urljoin(final, href))[0])
        except Exception:
            continue
    return parser.title or final, text, links, final


def _priority(url: str) -> int:
    path = urlparse(url).path.lower()
    if any(k in path for k in ("pricing", "price", "plans")):
        return 0
    if any(k in path for k in ("product", "feature", "about")):
        return 1
    return 2


def ingest_site(start_url: str, max_pages: int = 6) -> dict:
    seen: set[str] = set()
    queue = [start_url]
    pages: list[dict] = []
    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        url = urldefrag(url)[0]
        if url in seen or not _allowed(url) or _host(url) != _host(start_url):
            continue
        seen.add(url)
        try:
            title, text, links, final = _fetch(url)
        except (HTTPError, URLError, TimeoutError, OSError):
            continue
        if not text or len(text) < 40:
            continue
        pages.append({"url": final, "title": title or final, "text": text[:8000]})
        fresh = [link for link in links if link not in seen and _host(link) == _host(start_url)]
        fresh.sort(key=_priority)
        queue = [link for link in fresh if _priority(link) == 0] + queue + [link for link in fresh if _priority(link) != 0]

    context_parts = []
    for page in pages:
        context_parts.append(f"SOURCE {page['url']}\n{page['title']}\n{page['text']}")
    return {
        "url": start_url,
        "page_count": len(pages),
        "titles": [p["title"] for p in pages],
        "context": "\n\n".join(context_parts)[:24000],
    }
