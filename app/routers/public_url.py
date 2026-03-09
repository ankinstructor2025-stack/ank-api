from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, Header, HTTPException
from google.cloud import storage
from pydantic import BaseModel, Field

try:
    import ulid
except ImportError:
    ulid = None


router = APIRouter(tags=["public_url"])

DB_PATH = "ank.db"
BUCKET_NAME = os.getenv("BUCKET_NAME", "ank-bucket")
TEMPLATE_PREFIX = os.getenv("TEMPLATE_PREFIX", "template")

SOURCE_FILE_MAP = {
    "url_egov": "egov.json",
    "url_caa": "caa.json",
}


class PublicUrlRequest(BaseModel):
    source_key: str = Field(..., min_length=1)


@dataclass
class CrawlRule:
    same_domain_only: bool
    include_root_page: bool


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_id() -> str:
    if ulid is not None:
        return str(ulid.new())
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    return f"{scheme}://{netloc}{path}"


def same_domain(url1: str, url2: str) -> bool:
    return urlparse(url1).netloc.lower() == urlparse(url2).netloc.lower()


def get_blob_path(source_key: str) -> str:
    file_name = SOURCE_FILE_MAP.get(source_key)
    if not file_name:
        raise HTTPException(status_code=400, detail=f"unsupported source_key: {source_key}")
    return f"{TEMPLATE_PREFIX}/{file_name}"


def load_source_config(source_key: str) -> dict[str, Any]:
    blob_path = get_blob_path(source_key)

    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_path)

        if not blob.exists():
            raise HTTPException(
                status_code=404,
                detail=f"config not found: gs://{BUCKET_NAME}/{blob_path}"
            )

        text = blob.download_as_text(encoding="utf-8")
        return json.loads(text)

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"invalid json in gs://{BUCKET_NAME}/{blob_path}: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"failed to load source config: {e}"
        )


def load_crawl_rule(config: dict[str, Any]) -> tuple[dict[str, Any], CrawlRule]:
    request_conf = config.get("request", {}) or {}
    crawl_conf = config.get("crawl", {}) or {}

    return request_conf, CrawlRule(
        same_domain_only=bool(crawl_conf.get("same_domain_only", True)),
        include_root_page=bool(crawl_conf.get("include_root_page", True)),
    )


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


def ensure_url_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS url_roots (
          root_id TEXT PRIMARY KEY,
          source_type TEXT NOT NULL,
          root_url TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_url_roots_root_url
        ON url_roots(root_url)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS url_pages (
          page_id TEXT PRIMARY KEY,
          root_id TEXT NOT NULL,
          page_url TEXT NOT NULL,
          status TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_url_pages_page_url
        ON url_pages(page_url)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_url_pages_root_id
        ON url_pages(root_id)
        """
    )


def insert_url_root_if_not_exists(
    conn: sqlite3.Connection,
    root_id: str,
    source_type: str,
    root_url: str,
) -> str:
    now = now_iso()

    cur = conn.execute(
        """
        SELECT root_id
        FROM url_roots
        WHERE root_url = ?
        """,
        (root_url,),
    )
    row = cur.fetchone()
    if row:
        return row["root_id"]

    conn.execute(
        """
        INSERT INTO url_roots (
            root_id,
            source_type,
            root_url,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (root_id, source_type, root_url, now),
    )
    return root_id


def insert_url_page_if_not_exists(
    conn: sqlite3.Connection,
    page_id: str,
    root_id: str,
    page_url: str,
) -> bool:
    now = now_iso()

    cur = conn.execute(
        """
        INSERT OR IGNORE INTO url_pages (
            page_id,
            root_id,
            page_url,
            status,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (page_id, root_id, page_url, "new", now),
    )
    return cur.rowcount > 0


@router.post("/public-url/register")
async def public_url_register(
    req: PublicUrlRequest,
    authorization: str | None = Header(default=None),
):
    config = load_source_config(req.source_key)
    request_conf, rule = load_crawl_rule(config)

    target_url = str(request_conf.get("url") or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="request.url is required in template json")

    target_url = normalize_url(target_url)

    # root
    root_html = fetch_html(target_url, request_conf)
    level1_urls_all = extract_links(root_html, target_url)

    level1_urls: list[str] = []
    seen_pages: set[str] = set()

    for url in level1_urls_all:
        if rule.same_domain_only and not same_domain(url, target_url):
            continue
        if url == target_url:
            continue
        if url not in seen_pages:
            seen_pages.add(url)
            level1_urls.append(url)

    # level2
    level2_urls: list[str] = []
    for page_url in level1_urls:
        try:
            html = fetch_html(page_url, request_conf)
            child_urls = extract_links(html, page_url)
        except HTTPException:
            continue

        for child_url in child_urls:
            if rule.same_domain_only and not same_domain(child_url, target_url):
                continue
            if child_url == target_url:
                continue
            if child_url not in seen_pages:
                seen_pages.add(child_url)
                level2_urls.append(child_url)

    pages = level1_urls + level2_urls

    root_id = make_id()
    row_inserted = 0
    row_skipped = 0

    try:
        with get_conn() as conn:
            ensure_url_tables(conn)

            root_id = insert_url_root_if_not_exists(
                conn=conn,
                root_id=root_id,
                source_type=req.source_key,
                root_url=target_url,
            )

            result_pages: list[dict[str, Any]] = []

            for page_url in pages:
                inserted = insert_url_page_if_not_exists(
                    conn=conn,
                    page_id=make_id(),
                    root_id=root_id,
                    page_url=page_url,
                )

                if inserted:
                    row_inserted += 1
                    status = "new"
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
        "root_url": target_url,
        "root_id": root_id,
        "page_count": len(pages),
        "row_inserted": row_inserted,
        "row_skipped": row_skipped,
        "pages": result_pages,
        "message": f"{req.source_key} public url registered",
    }
