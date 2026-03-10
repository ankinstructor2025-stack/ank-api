from __future__ import annotations

import json
import os
import re
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
    max_depth: int
    max_pages: int


@dataclass
class CrawlNode:
    url: str
    parent_url: str | None
    depth: int


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

    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{netloc}{path}{query}"


def same_domain(url1: str, url2: str) -> bool:
    return urlparse(url1).netloc.lower() == urlparse(url2).netloc.lower()


def is_html_like_url(url: str) -> bool:
    lower = url.lower()
    blocked_exts = (
        ".pdf", ".zip", ".xls", ".xlsx", ".csv", ".json",
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
        ".doc", ".docx", ".ppt", ".pptx", ".xml",
        ".mp3", ".mp4", ".avi", ".mov"
    )
    return not lower.endswith(blocked_exts)


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
        include_root_page=bool(crawl_conf.get("include_root_page", False)),
        max_depth=int(crawl_conf.get("max_depth", 3)),
        max_pages=int(crawl_conf.get("max_pages", 200)),
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

        lower = href.lower()
        if lower.startswith(("javascript:", "mailto:", "tel:")):
            continue

        abs_url = urljoin(base_url, href)
        norm = normalize_url(abs_url)

        if norm not in seen:
            seen.add(norm)
            urls.append(norm)

    return urls


def filter_child_urls(
    urls: list[str],
    target_url: str,
    rule: CrawlRule,
    seen_urls: set[str],
) -> list[str]:
    result: list[str] = []

    for url in urls:
        if rule.same_domain_only and not same_domain(url, target_url):
            continue
        if url == target_url:
            continue
        if not is_html_like_url(url):
            continue
        if url in seen_urls:
            continue

        seen_urls.add(url)
        result.append(url)

    return result


def extract_page_features(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    body = soup.body or soup
    text = " ".join(body.stripped_strings)
    text = re.sub(r"\s+", " ", text).strip()

    link_count = len(body.find_all("a", href=True))
    text_length = len(text)

    return {
        "title": title,
        "text": text,
        "text_length": text_length,
        "link_count": link_count,
    }


def judge_page(url: str, title: str, text: str, text_length: int, link_count: int) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []

    lower_url = url.lower()
    lower_title = title.lower()
    lower_text = text.lower()

    faq_keywords = ["faq", "q&a", "qanda", "よくある質問", "質問", "回答"]
    guide_keywords = ["手続き", "申請", "届出", "利用方法", "使い方", "必要書類", "準備", "案内", "方法"]
    notice_keywords = ["お知らせ", "新着", "更新情報", "障害", "報道発表", "発生について", "掲載", "公表"]
    list_keywords = ["一覧", "カテゴリ", "メニュー", "index", "list"]

    if any(k in lower_url for k in ["faq", "qanda", "guide", "help", "manual"]):
        score += 20
        reasons.append("url_keyword")

    if any(k in lower_title for k in faq_keywords):
        score += 40
        reasons.append("title_faq")

    if any(k in title for k in guide_keywords):
        score += 30
        reasons.append("title_guide")

    if any(k in title for k in notice_keywords):
        score -= 40
        reasons.append("title_notice")

    if any(k in title for k in list_keywords):
        score -= 25
        reasons.append("title_list")

    qa_pattern_count = 0
    for pat in [r"\bq\s*[:：]", r"\ba\s*[:：]", r"質問", r"回答"]:
        qa_pattern_count += len(re.findall(pat, lower_text if "q" in pat or "a" in pat else text))

    if qa_pattern_count >= 2:
        score += 35
        reasons.append("qa_pattern")

    if any(k in text for k in guide_keywords):
        score += 20
        reasons.append("body_guide")

    if text_length >= 1200:
        score += 20
        reasons.append("long_text")
    elif text_length >= 500:
        score += 10
        reasons.append("enough_text")
    elif text_length < 200:
        score -= 25
        reasons.append("too_short")

    if link_count >= 30 and text_length < 800:
        score -= 25
        reasons.append("link_heavy")

    if "pdf" in lower_text and text_length < 400:
        score -= 15
        reasons.append("pdf_only_like")

    page_type = "unknown"

    if any(k in lower_title for k in faq_keywords) or qa_pattern_count >= 2:
        page_type = "faq"
    elif any(k in title for k in guide_keywords):
        page_type = "guide"
    elif any(k in title for k in notice_keywords):
        page_type = "notice"
    elif any(k in title for k in list_keywords) or (link_count >= 30 and text_length < 800):
        page_type = "list"

    score = max(0, min(100, score))

    is_usable = 0
    if page_type in ("faq", "guide") and score >= 40:
        is_usable = 1
    elif page_type == "unknown" and score >= 60 and text_length >= 500:
        is_usable = 1

    return {
        "score": score,
        "page_type": page_type,
        "is_usable": is_usable,
        "judge_reason": ",".join(reasons) if reasons else "no_rule_match",
    }


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
          parent_page_id TEXT,
          page_url TEXT NOT NULL,
          depth INTEGER NOT NULL,
          status TEXT NOT NULL,
          child_count INTEGER NOT NULL DEFAULT 0,
          score INTEGER NOT NULL DEFAULT 0,
          page_type TEXT NOT NULL DEFAULT 'unknown',
          is_usable INTEGER NOT NULL DEFAULT 0,
          judge_reason TEXT,
          created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_url_pages_root_page_url
        ON url_pages(root_id, page_url)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_url_pages_root_id
        ON url_pages(root_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_url_pages_parent_page_id
        ON url_pages(parent_page_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_url_pages_root_depth
        ON url_pages(root_id, depth)
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


def find_page_row(
    conn: sqlite3.Connection,
    root_id: str,
    page_url: str,
) -> sqlite3.Row | None:
    cur = conn.execute(
        """
        SELECT *
        FROM url_pages
        WHERE root_id = ? AND page_url = ?
        """,
        (root_id, page_url),
    )
    return cur.fetchone()


def upsert_url_page(
    conn: sqlite3.Connection,
    root_id: str,
    parent_page_id: str | None,
    page_url: str,
    depth: int,
    status: str,
    child_count: int,
    score: int,
    page_type: str,
    is_usable: int,
    judge_reason: str | None,
) -> tuple[bool, str]:
    now = now_iso()
    existing = find_page_row(conn, root_id, page_url)

    if existing:
        page_id = existing["page_id"]
        conn.execute(
            """
            UPDATE url_pages
               SET parent_page_id = ?,
                   depth = ?,
                   status = ?,
                   child_count = ?,
                   score = ?,
                   page_type = ?,
                   is_usable = ?,
                   judge_reason = ?
             WHERE page_id = ?
            """,
            (
                parent_page_id,
                depth,
                "duplicate" if status == "new" else status,
                child_count,
                score,
                page_type,
                is_usable,
                judge_reason,
                page_id,
            ),
        )
        return False, page_id

    page_id = make_id()
    conn.execute(
        """
        INSERT INTO url_pages (
            page_id,
            root_id,
            parent_page_id,
            page_url,
            depth,
            status,
            child_count,
            score,
            page_type,
            is_usable,
            judge_reason,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            page_id,
            root_id,
            parent_page_id,
            page_url,
            depth,
            status,
            child_count,
            score,
            page_type,
            is_usable,
            judge_reason,
            now,
        ),
    )
    return True, page_id


def build_page_results(
    target_url: str,
    request_conf: dict[str, Any],
    rule: CrawlRule,
) -> list[dict[str, Any]]:
    root_html = fetch_html(target_url, request_conf)
    root_links_all = extract_links(root_html, target_url)

    seen_urls: set[str] = set()
    level1_urls = filter_child_urls(root_links_all, target_url, rule, seen_urls)

    queue: list[CrawlNode] = [
        CrawlNode(url=url, parent_url=None, depth=1)
        for url in level1_urls
    ]

    page_results: list[dict[str, Any]] = []
    idx = 0

    while idx < len(queue) and len(page_results) < rule.max_pages:
        node = queue[idx]
        idx += 1

        page_info = {
            "page_id": None,
            "page_url": node.url,
            "parent_url": node.parent_url,
            "parent_page_id": None,
            "depth": node.depth,
            "status": "new",
            "child_count": 0,
            "score": 0,
            "page_type": "unknown",
            "is_usable": 0,
            "judge_reason": None,
            "created_at": now_iso(),
        }

        try:
            html = fetch_html(node.url, request_conf)
            features = extract_page_features(html)
            child_urls_all = extract_links(html, node.url)

            child_urls = []
            if node.depth < rule.max_depth:
                child_urls = filter_child_urls(child_urls_all, target_url, rule, seen_urls)
                for child_url in child_urls:
                    if len(queue) >= rule.max_pages:
                        break
                    queue.append(
                        CrawlNode(
                            url=child_url,
                            parent_url=node.url,
                            depth=node.depth + 1,
                        )
                    )

            page_info["child_count"] = len(child_urls)

            judged = judge_page(
                url=node.url,
                title=features["title"],
                text=features["text"],
                text_length=features["text_length"],
                link_count=features["link_count"],
            )
            page_info["score"] = judged["score"]
            page_info["page_type"] = judged["page_type"]
            page_info["is_usable"] = judged["is_usable"]
            page_info["judge_reason"] = judged["judge_reason"]

        except HTTPException:
            page_info["status"] = "fetch_error"
            page_info["score"] = 0
            page_info["page_type"] = "unknown"
            page_info["is_usable"] = 0
            page_info["judge_reason"] = "fetch_error"

        page_results.append(page_info)

    return page_results


@router.post("/public-url/register")
async def public_url_register(
    req: PublicUrlRequest,
    authorization: str | None = Header(default=None),
):
    # authorization は現時点では未使用だが、既存APIとの整合のため残す
    _ = authorization

    config = load_source_config(req.source_key)
    request_conf, rule = load_crawl_rule(config)

    target_url = str(request_conf.get("url") or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="request.url is required in template json")

    target_url = normalize_url(target_url)

    root_id = make_id()
    row_inserted = 0
    row_skipped = 0
    result_pages: list[dict[str, Any]] = []

    try:
        page_results = build_page_results(
            target_url=target_url,
            request_conf=request_conf,
            rule=rule,
        )

        with get_conn() as conn:
            ensure_url_tables(conn)

            root_id = insert_url_root_if_not_exists(
                conn=conn,
                root_id=root_id,
                source_type=req.source_key,
                root_url=target_url,
            )

            url_to_page_id: dict[str, str] = {}

            for page in page_results:
                parent_page_id = None
                if page["parent_url"]:
                    parent_page_id = url_to_page_id.get(page["parent_url"])

                inserted, actual_page_id = upsert_url_page(
                    conn=conn,
                    root_id=root_id,
                    parent_page_id=parent_page_id,
                    page_url=page["page_url"],
                    depth=page["depth"],
                    status=page["status"],
                    child_count=page["child_count"],
                    score=page["score"],
                    page_type=page["page_type"],
                    is_usable=page["is_usable"],
                    judge_reason=page["judge_reason"],
                )

                url_to_page_id[page["page_url"]] = actual_page_id

                if inserted:
                    row_inserted += 1
                else:
                    row_skipped += 1
                    if page["status"] == "new":
                        page["status"] = "duplicate"

                page["page_id"] = actual_page_id
                page["parent_page_id"] = parent_page_id
                result_pages.append(page)

            conn.commit()

    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail=f"sqlite integrity error: {e}")
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"sqlite error: {e}")

    for page in result_pages:
        page.pop("parent_url", None)

    return {
        "mode": "public_url_register",
        "source_key": req.source_key,
        "root_url": target_url,
        "root_id": root_id,
        "max_depth": rule.max_depth,
        "max_pages": rule.max_pages,
        "page_count": len(result_pages),
        "row_inserted": row_inserted,
        "row_skipped": row_skipped,
        "pages": result_pages,
        "message": f"{req.source_key} public url registered",
    }
