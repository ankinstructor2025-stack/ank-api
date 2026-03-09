from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

try:
    import ulid
except ImportError:
    ulid = None


router = APIRouter(tags=["public_url"])


DB_PATH = "ank.db"


# =========================
# request model
# =========================
class PublicUrlRequest(BaseModel):
    source_key: str = Field(..., min_length=1)
    target_url: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


# =========================
# internal model
# =========================
@dataclass
class CrawlRule:
    same_domain_only: bool
    include_root_page: bool
    link_mode: str
    page_url_prefixes: list[str]
    exclude_exact_urls: set[str]


# =========================
# utility
# =========================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_id() -> str:
    if ulid is not None:
        return str(ulid.new())
    # ulid未導入でも落とさない
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    # query / fragment はURL管理上ノイズになりやすいので除外
    return f"{scheme}://{netloc}{path}"


def same_domain(url1: str, url2: str) -> bool:
    return urlparse(url1).netloc.lower() == urlparse(url2).netloc.lower()


def load_crawl_rule(config: dict[str, Any]) -> tuple[dict[str, Any], CrawlRule]:
    request_conf = config.get("request", {}) or {}
    crawl_conf = config.get("crawl", {}) or {}

    prefixes = crawl_conf.get("page_url_prefixes") or []
    exclude_exact_urls = set(crawl_conf.get("exclude_exact_urls") or [])

    return request_conf, CrawlRule(
        same_domain_only=bool(crawl_conf.get("same_domain_only", True)),
        include_root_page=bool(crawl_conf.get("include_root_page", True)),
        link_mode=str(crawl_conf.get("link_mode", "prefix")),
        page_url_prefixes=[normalize_url(x) for x in prefixes if x],
        exclude_exact_urls={normalize_url(x) for x in exclude_exact_urls if x},
    )


def is_target_page(url: str, root_url: str, rule: CrawlRule) -> bool:
    normalized = normalize_url(url)

    if rule.same_domain_only and not same_domain(normalized, root_url):
        return False

    if normalized in rule.exclude_exact_urls:
        return False

    if rule.link_mode == "prefix":
        if not rule.page_url_prefixes:
            return True
        return any(normalized.startswith(prefix) for prefix in rule.page_url_prefixes)

    return True


def fetch_html(url: str, request_conf: dict[str, Any]) -> str:
    method = str(request_conf.get("method", "GET")).upper()
    timeout_sec = int(request_conf.get("timeout_sec", 15))
    headers = request_conf.get("headers") or {}

    if method != "GET":
        raise HTTPException(status_code=400, detail=f"unsupported request.method: {method}")

    try:
        res = requests.get(url, headers=headers, timeout=timeout_sec)
        res.raise_for_status()
        res.encoding = res.apparent_encoding or res.encoding
        return res.text
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"public url fetch error: {e}")


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if href.startswith("#"):
            continue
        if href.lower().startswith("javascript:"):
            continue
        if href.lower().startswith("mailto:"):
            continue
        if href.lower().startswith("tel:"):
            continue

        abs_url = urljoin(base_url, href)
        norm = normalize_url(abs_url)

        if norm not in seen:
            seen.add(norm)
            urls.append(norm)

    return urls


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# =========================
# DB
# 既存テーブル前提
# =========================
def insert_url_root(
    conn: sqlite3.Connection,
    root_id: str,
    source_key: str,
    requested_url: str,
    target_url: str,
) -> None:
    now = now_iso()
    conn.execute(
        """
        INSERT INTO url_roots (
            root_id,
            source_key,
            requested_url,
            target_url,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (root_id, source_key, requested_url, target_url, now, now),
    )


def insert_url_page_if_not_exists(
    conn: sqlite3.Connection,
    page_id: str,
    root_id: str,
    source_key: str,
    page_url: str,
    row_index: int,
) -> bool:
    now = now_iso()

    cur = conn.execute(
        """
        INSERT OR IGNORE INTO url_pages (
            page_id,
            root_id,
            source_key,
            page_url,
            status,
            row_index,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (page_id, root_id, source_key, page_url, "discovered", row_index, now, now),
    )
    return cur.rowcount > 0


# =========================
# route
# =========================
@router.post("/public-url/register")
async def public_url_register(
    req: PublicUrlRequest,
    authorization: str | None = Header(default=None),
):
    # authorization は既存構成に合わせて受けるだけ
    # 認証チェックは既存共通処理があるならそちらへ寄せる

    config = req.config or {}
    request_conf, rule = load_crawl_rule(config)

    requested_url = (req.target_url or "").strip()
    config_url = str(request_conf.get("url") or "").strip()
    target_url = config_url or requested_url

    if not target_url:
        raise HTTPException(status_code=400, detail="target_url is required")

    target_url = normalize_url(target_url)

    html = fetch_html(target_url, request_conf)
    extracted = extract_links(html, target_url)

    pages: list[str] = []
    seen: set[str] = set()

    if rule.include_root_page and target_url not in rule.exclude_exact_urls:
        if is_target_page(target_url, target_url, rule):
            seen.add(target_url)
            pages.append(target_url)

    for url in extracted:
        if url in seen:
            continue
        if not is_target_page(url, target_url, rule):
            continue
        seen.add(url)
        pages.append(url)

    root_id = make_id()
    row_inserted = 0
    row_skipped = 0

    try:
        with get_conn() as conn:
            insert_url_root(
                conn=conn,
                root_id=root_id,
                source_key=req.source_key,
                requested_url=requested_url or target_url,
                target_url=target_url,
            )

            result_pages: list[dict[str, Any]] = []

            for idx, page_url in enumerate(pages, start=1):
                inserted = insert_url_page_if_not_exists(
                    conn=conn,
                    page_id=make_id(),
                    root_id=root_id,
                    source_key=req.source_key,
                    page_url=page_url,
                    row_index=idx,
                )

                if inserted:
                    row_inserted += 1
                    status = "discovered"
                else:
                    row_skipped += 1
                    status = "duplicate"

                result_pages.append(
                    {
                        "page_url": page_url,
                        "status": status,
                        "created_at": now_iso(),
                    }
                )

            conn.commit()

    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail=f"sqlite integrity error: {e}")
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"sqlite error: {e}")

    return {
        "mode": "public_url_register",
        "source_key": req.source_key,
        "requested_url": requested_url or target_url,
        "target_url": target_url,
        "root_id": root_id,
        "page_count": len(pages),
        "row_inserted": row_inserted,
        "row_skipped": row_skipped,
        "pages": result_pages,
        "message": f"{req.source_key} public url registered",
    }
