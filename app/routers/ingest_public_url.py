from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, Header, HTTPException
from google.cloud import storage
from pydantic import BaseModel, Field

import firebase_admin
from firebase_admin import auth as fb_auth

import ulid


router = APIRouter(prefix="/public-url", tags=["public_url"])

JST = ZoneInfo("Asia/Tokyo")

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
TEMPLATE_PREFIX = os.getenv("TEMPLATE_PREFIX", "template")
PUBLIC_URL_CONFIG_NAME = "public_url.json"
DEFAULT_TIMEOUT_SEC = 15
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ank-bot/1.0)"
}


class PublicUrlRequest(BaseModel):
    source_key: str = Field(..., min_length=1)


class PublicUrlDecomposeRequest(BaseModel):
    page_url: str = Field(..., min_length=1)


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


def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


def get_public_url_blob_path() -> str:
    return f"{TEMPLATE_PREFIX}/{PUBLIC_URL_CONFIG_NAME}"


def ensure_firebase_initialized():
    if firebase_admin._apps:
        return
    firebase_admin.initialize_app(options={"projectId": "ank-firebase"})


def get_uid_from_auth_header(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    token = authorization.replace("Bearer ", "", 1).strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")

    ensure_firebase_initialized()
    try:
        decoded = fb_auth.verify_id_token(token)
        uid = decoded.get("uid")
        if not uid:
            raise HTTPException(status_code=401, detail="uid not found in token")
        return uid
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid ID token: {e}")


def now_iso() -> str:
    return datetime.now(tz=JST).isoformat()


def make_id() -> str:
    return str(ulid.new())


def normalize_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
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
    lower = (url or "").lower()
    blocked_exts = (
        ".pdf", ".zip", ".xls", ".xlsx", ".csv", ".json",
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
        ".doc", ".docx", ".ppt", ".pptx", ".xml",
        ".mp3", ".mp4", ".avi", ".mov"
    )
    return not lower.endswith(blocked_exts)


def load_public_url_config() -> dict[str, Any]:
    blob_path = get_public_url_blob_path()

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
        data = json.loads(text)
        if not isinstance(data, dict):
            raise HTTPException(status_code=500, detail="public_url.json root must be object")
        return data

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
            detail=f"failed to load public_url config: {e}"
        )


def get_public_url_source(config: dict[str, Any], source_key: str) -> dict[str, Any]:
    sources = config.get("sources") or []
    if not isinstance(sources, list):
        raise HTTPException(status_code=500, detail="public_url.json: sources must be array")

    for source in sources:
        if not isinstance(source, dict):
            continue
        if str(source.get("source_key") or "").strip() == source_key:
            return source

    raise HTTPException(status_code=400, detail=f"unsupported source_key: {source_key}")


def load_crawl_rule(config: dict[str, Any]) -> CrawlRule:
    crawl_conf = config.get("crawl", {}) or {}

    return CrawlRule(
        same_domain_only=bool(crawl_conf.get("same_domain_only", True)),
        include_root_page=bool(crawl_conf.get("include_root_page", False)),
        max_depth=int(crawl_conf.get("max_depth", 3)),
        max_pages=int(crawl_conf.get("max_pages", 200)),
    )


def get_source_url(source: dict[str, Any]) -> str:
    url = str(source.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required in public_url.json sources[]")
    return normalize_url(url)


def fetch_html(url: str) -> str:
    try:
        res = requests.get(url, headers=DEFAULT_HEADERS, timeout=DEFAULT_TIMEOUT_SEC)
        res.raise_for_status()
        res.encoding = res.apparent_encoding or res.encoding
        return res.text
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"public url fetch error: {e}")


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

    qa_pattern_count = 0
    for pat in [r"\bq\s*[:：]", r"\ba\s*[:：]"]:
        qa_pattern_count += len(re.findall(pat, lower_text))
    qa_pattern_count += text.count("質問")
    qa_pattern_count += text.count("回答")

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

    if link_count >= 20 and text_length < 1200:
        score -= 20
        reasons.append("link_heavy")

    if "pdf" in lower_text and text_length < 400:
        score -= 15
        reasons.append("pdf_only_like")

    page_type = "unknown"

    if any(k in title for k in notice_keywords):
        page_type = "notice"
    elif (
        any(k in lower_title for k in ["一覧", "カテゴリ", "index", "list"])
        or (link_count >= 10 and text_length < 1500)
        or ("ヘルプ" in title and link_count >= 5)
    ):
        page_type = "list"
    elif any(k in lower_title for k in faq_keywords) or qa_pattern_count >= 2:
        page_type = "faq"
    elif any(k in title for k in guide_keywords):
        page_type = "guide"

    if page_type == "list":
        score = min(score, 60)

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
) -> tuple[bool, str, str]:
    now = now_iso()
    existing = find_page_row(conn, root_id, page_url)

    if existing:
        page_id = existing["page_id"]

        current_status = str(existing["status"] or "new")
        if current_status == "done":
            next_status = "done"
        elif status == "fetch_error":
            next_status = "fetch_error"
        else:
            next_status = "new"

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
                next_status,
                child_count,
                score,
                page_type,
                is_usable,
                judge_reason,
                page_id,
            ),
        )
        return False, page_id, next_status

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
    return True, page_id, status


def build_page_results(
    target_url: str,
    rule: CrawlRule,
) -> list[dict[str, Any]]:
    root_html = fetch_html(target_url)
    root_links_all = extract_links(root_html, target_url)

    seen_urls: set[str] = set()
    level1_urls = filter_child_urls(root_links_all, target_url, rule, seen_urls)

    stack: list[CrawlNode] = [
        CrawlNode(url=url, parent_url=None, depth=1)
        for url in reversed(level1_urls)
    ]

    page_results: list[dict[str, Any]] = []

    if rule.include_root_page:
        root_features = extract_page_features(root_html)
        root_judged = judge_page(
            url=target_url,
            title=root_features["title"],
            text=root_features["text"],
            text_length=root_features["text_length"],
            link_count=root_features["link_count"],
        )
        page_results.append(
            {
                "page_id": None,
                "page_url": target_url,
                "parent_url": None,
                "parent_page_id": None,
                "depth": 0,
                "status": "new",
                "child_count": len(level1_urls),
                "score": root_judged["score"],
                "page_type": root_judged["page_type"],
                "is_usable": root_judged["is_usable"],
                "judge_reason": root_judged["judge_reason"],
                "created_at": now_iso(),
            }
        )

    while stack and len(page_results) < rule.max_pages:
        node = stack.pop()

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
            html = fetch_html(node.url)
            features = extract_page_features(html)

            child_urls: list[str] = []
            if node.depth < rule.max_depth:
                child_urls_all = extract_links(html, node.url)
                child_urls = filter_child_urls(child_urls_all, target_url, rule, seen_urls)

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

            if node.depth < rule.max_depth:
                for child_url in reversed(child_urls):
                    if len(page_results) + len(stack) >= rule.max_pages:
                        break
                    stack.append(
                        CrawlNode(
                            url=child_url,
                            parent_url=node.url,
                            depth=node.depth + 1,
                        )
                    )

        except HTTPException:
            page_info["status"] = "fetch_error"
            page_info["score"] = 0
            page_info["page_type"] = "unknown"
            page_info["is_usable"] = 0
            page_info["judge_reason"] = "fetch_error"

        page_results.append(page_info)

    return page_results


def extract_main_content_blocks(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile("main|content|contents", re.I))
        or soup.find(class_=re.compile("main|content|contents", re.I))
        or soup.body
        or soup
    )

    blocks: list[str] = []
    seen: set[str] = set()

    for tag in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "dt", "dd", "th", "td"]):
        text = " ".join(tag.stripped_strings)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            continue
        if len(text) < 8:
            continue
        if text in seen:
            continue

        seen.add(text)
        blocks.append(text)

    if blocks:
        return blocks

    fallback_text = main.get_text("\n", strip=True)
    lines = [re.sub(r"\s+", " ", x).strip() for x in fallback_text.splitlines()]
    lines = [x for x in lines if len(x) >= 8]

    return lines


def merge_blocks_for_row_data(blocks: list[str], min_len: int = 80, max_len: int = 500) -> list[str]:
    rows: list[str] = []
    buffer = ""

    for block in blocks:
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue

        if not buffer:
            buffer = block
            continue

        if len(buffer) + 1 + len(block) <= max_len:
            buffer += " " + block
            continue

        rows.append(buffer)
        buffer = block

    if buffer:
        rows.append(buffer)

    cleaned = [x.strip() for x in rows if x and len(x.strip()) >= 20]
    return cleaned


def find_page_with_root(conn: sqlite3.Connection, page_url: str) -> sqlite3.Row | None:
    cur = conn.execute(
        """
        SELECT
            p.page_id,
            p.root_id,
            p.page_url,
            p.depth,
            p.status,
            p.page_type,
            p.is_usable,
            r.source_type AS source_key,
            r.root_url
        FROM url_pages p
        JOIN url_roots r
          ON p.root_id = r.root_id
        WHERE p.page_url = ?
        ORDER BY p.created_at DESC
        LIMIT 1
        """,
        (page_url,),
    )
    return cur.fetchone()


def replace_row_data_for_public_url(
    conn: sqlite3.Connection,
    file_id: str,
    source_key: str,
    source_item_id: str,
    contents: list[str],
) -> tuple[int, int]:
    conn.execute(
        """
        DELETE FROM row_data
        WHERE file_id = ?
          AND source_type = 'public_url'
        """,
        (file_id,),
    )

    inserted = 0
    skipped = 0
    created_at = now_iso()

    for idx, content in enumerate(contents, start=1):
        row_source_item_id = f"{source_item_id}:{idx}"

        conn.execute(
            """
            INSERT INTO row_data
              (row_id, file_id, source_type, source_key, source_item_id, row_index, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                make_id(),
                file_id,
                "public_url",
                source_key,
                row_source_item_id,
                idx,
                content,
                created_at,
            ),
        )
        inserted += 1

    return inserted, skipped


@router.post("/register")
async def public_url_register(
    req: PublicUrlRequest,
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)

    config = load_public_url_config()
    source = get_public_url_source(config, req.source_key)
    rule = load_crawl_rule(config)
    target_url = get_source_url(source)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}"
        )

    local_db_path = f"/tmp/ank_{uid}_public_url.db"
    db_blob.download_to_filename(local_db_path)

    root_id = make_id()
    row_inserted = 0
    row_skipped = 0
    result_pages: list[dict[str, Any]] = []

    try:
        page_results = build_page_results(
            target_url=target_url,
            rule=rule,
        )

        conn = sqlite3.connect(local_db_path)
        conn.row_factory = sqlite3.Row
        try:
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

                inserted, actual_page_id, actual_status = upsert_url_page(
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

                page["page_id"] = actual_page_id
                page["parent_page_id"] = parent_page_id
                page["status"] = actual_status
                result_pages.append(page)

            conn.commit()
        finally:
            conn.close()

        db_blob.upload_from_filename(local_db_path)

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


@router.post("/decompose")
async def public_url_decompose(
    req: PublicUrlDecomposeRequest,
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)
    page_url = normalize_url(req.page_url)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}"
        )

    local_db_path = f"/tmp/ank_{uid}_public_url.db"
    db_blob.download_to_filename(local_db_path)

    try:
        conn = sqlite3.connect(local_db_path)
        conn.row_factory = sqlite3.Row
        try:
            ensure_url_tables(conn)

            page_row = find_page_with_root(conn, page_url)
            if not page_row:
                raise HTTPException(status_code=404, detail="page not found")

            html = fetch_html(page_url)
            raw_blocks = extract_main_content_blocks(html)
            contents = merge_blocks_for_row_data(raw_blocks)

            if not contents:
                raise HTTPException(status_code=400, detail="no content extracted")

            row_count, skipped_count = replace_row_data_for_public_url(
                conn=conn,
                file_id=str(page_row["page_id"]),
                source_key=str(page_row["source_key"]),
                source_item_id=page_url,
                contents=contents,
            )

            conn.execute(
                """
                UPDATE url_pages
                   SET status = 'done'
                 WHERE page_id = ?
                """,
                (str(page_row["page_id"]),),
            )

            conn.commit()
        finally:
            conn.close()

        db_blob.upload_from_filename(local_db_path)

        return {
            "mode": "public_url_decompose",
            "page_url": page_url,
            "file_id": str(page_row["page_id"]),
            "source_key": str(page_row["source_key"]),
            "row_count": row_count,
            "qa_count": 0,
            "text_count": row_count,
            "skipped_count": skipped_count,
            "message": "public url decomposed into row_data",
        }

    except HTTPException:
        raise
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail=f"sqlite integrity error: {e}")
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"sqlite error: {e}")
