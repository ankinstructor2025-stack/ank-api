from fastapi import APIRouter, HTTPException, Header, Request
import os
import json
import sqlite3
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth

import ulid

router = APIRouter(prefix="/caa", tags=["caa"])

JST = ZoneInfo("Asia/Tokyo")

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
CAA_TEMPLATE_PATH = "template/caa.json"


def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


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
        raise HTTPException(status_code=401, detail=str(e))


def load_caa_template() -> dict:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(CAA_TEMPLATE_PATH)

    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"{CAA_TEMPLATE_PATH} not found")

    text = blob.download_as_text(encoding="utf-8")
    try:
        return json.loads(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"invalid template json: {e}")


def fetch_html(url: str, headers: dict | None = None, timeout_sec: int = 15) -> tuple[str, str, int]:
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout_sec)
        r.raise_for_status()
        return r.url, r.text or "", r.status_code
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"caa fetch error: {str(e)}")


def normalize_lines(text: str) -> list[str]:
    lines = [x.strip() for x in text.splitlines()]
    return [x for x in lines if x]


def is_same_domain(base_url: str, target_url: str) -> bool:
    try:
        return urlparse(base_url).netloc == urlparse(target_url).netloc
    except Exception:
        return False


def is_question_start(line: str) -> bool:
    patterns = [
        r"^Q[\.．:：]?\s*",
        r"^問[0-9０-９]*[\.．:：]?\s*",
        r"^質問[0-9０-９]*[\.．:：]?\s*",
    ]
    return any(re.match(p, line) for p in patterns)


def is_answer_start(line: str) -> bool:
    patterns = [
        r"^A[\.．:：]?\s*",
        r"^答[0-9０-９]*[\.．:：]?\s*",
        r"^回答[0-9０-９]*[\.．:：]?\s*",
    ]
    return any(re.match(p, line) for p in patterns)


def strip_question_prefix(line: str) -> str:
    return re.sub(r"^(Q|問|質問)[0-9０-９]*[\.．:：]?\s*", "", line).strip()


def strip_answer_prefix(line: str) -> str:
    return re.sub(r"^(A|答|回答)[0-9０-９]*[\.．:：]?\s*", "", line).strip()


def dedupe_qa(items: list[dict]) -> list[dict]:
    unique = []
    seen = set()

    for item in items:
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if not q or not a:
            continue

        key = (q, a)
        if key in seen:
            continue

        seen.add(key)
        unique.append({
            "question": q,
            "answer": a,
        })

    return unique


def extract_qa_from_text_lines(lines: list[str]) -> list[dict]:
    results = []
    q = None
    a_lines = []
    mode = None

    for line in lines:
        if is_question_start(line):
            if q and a_lines:
                results.append({
                    "question": strip_question_prefix(q).strip(),
                    "answer": "\n".join(a_lines).strip()
                })
            q = line
            a_lines = []
            mode = "q"
            continue

        if is_answer_start(line):
            mode = "a"
            ans = strip_answer_prefix(line)
            if ans:
                a_lines.append(ans)
            continue

        if mode == "q" and q:
            q += " " + line
        elif mode == "a":
            a_lines.append(line)

    if q and a_lines:
        results.append({
            "question": strip_question_prefix(q).strip(),
            "answer": "\n".join(a_lines).strip()
        })

    return dedupe_qa(results)


def extract_qa_from_dl(soup: BeautifulSoup) -> list[dict]:
    results = []

    for dl in soup.find_all("dl"):
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        if not dts or not dds:
            continue

        pair_count = min(len(dts), len(dds))
        for i in range(pair_count):
            q = dts[i].get_text(" ", strip=True)
            a = dds[i].get_text("\n", strip=True)
            if q and a:
                results.append({
                    "question": q,
                    "answer": a,
                })

    return dedupe_qa(results)


def extract_qa_from_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    qa_from_dl = extract_qa_from_dl(soup)
    if qa_from_dl:
        return qa_from_dl

    text = soup.get_text("\n")
    lines = normalize_lines(text)
    return extract_qa_from_text_lines(lines)


def extract_caa_qa_links(index_html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(index_html, "html.parser")
    urls = []

    base_prefix = "https://www.caa.go.jp/policies/policy/consumer_policy/"

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        label = a.get_text(" ", strip=True)

        if not href:
            continue

        full_url = urljoin(base_url, href)

        if not is_same_domain(base_url, full_url):
            continue

        text_hit = any(word in label for word in [
            "FAQ", "Q&A", "Q＆A", "よくある質問", "質問", "相談"
        ])
        url_hit = any(word in full_url.lower() for word in [
            "/faq", "faq", "qanda", "question"
        ])
        prefix_hit = full_url.startswith(base_prefix)

        if text_hit or url_hit or prefix_hit:
            if full_url.rstrip("/") != base_url.rstrip("/"):
                urls.append(full_url)

    unique_urls = []
    seen = set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)

    return unique_urls


def ensure_url_tables(cur: sqlite3.Cursor):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS url_roots (
          root_id TEXT PRIMARY KEY,
          source_type TEXT NOT NULL,
          root_url TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_url_roots_root_url
        ON url_roots(root_url)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS url_pages (
          page_id TEXT PRIMARY KEY,
          root_id TEXT NOT NULL,
          page_url TEXT NOT NULL,
          status TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_url_pages_page_url
        ON url_pages(page_url)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS ix_url_pages_root_id
        ON url_pages(root_id)
    """)


def get_or_create_root(cur: sqlite3.Cursor, root_url: str, created_at: str) -> str:
    cur.execute("""
        SELECT root_id
        FROM url_roots
        WHERE root_url = ?
        LIMIT 1
    """, (root_url,))
    row = cur.fetchone()
    if row:
        return row[0]

    root_id = str(ulid.new())
    cur.execute("""
        INSERT INTO url_roots (root_id, source_type, root_url, created_at)
        VALUES (?, ?, ?, ?)
    """, (root_id, "caa", root_url, created_at))
    return root_id


def get_or_create_page(cur: sqlite3.Cursor, root_id: str, page_url: str, created_at: str) -> str:
    cur.execute("""
        SELECT page_id
        FROM url_pages
        WHERE page_url = ?
        LIMIT 1
    """, (page_url,))
    row = cur.fetchone()
    if row:
        return row[0]

    page_id = str(ulid.new())
    cur.execute("""
        INSERT INTO url_pages (page_id, root_id, page_url, status, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (page_id, root_id, page_url, "new", created_at))
    return page_id


def update_page_status(cur: sqlite3.Cursor, page_url: str, status: str):
    cur.execute("""
        UPDATE url_pages
        SET status = ?
        WHERE page_url = ?
    """, (status, page_url))


def build_target_url(override_target_url: str | None = None) -> tuple[str, dict]:
    tpl = load_caa_template()

    request_cfg = tpl.get("request") or {}
    method = (request_cfg.get("method") or "GET").upper()
    url = (override_target_url or request_cfg.get("url") or "").strip()

    if not url:
        raise HTTPException(status_code=400, detail="template url is empty")

    if method != "GET":
        raise HTTPException(status_code=400, detail=f"unsupported method: {method}")

    return url, tpl


def fetch_index_and_detail_qa(target_url: str, tpl: dict) -> tuple[str, list[dict], int, list[str]]:
    request_cfg = tpl.get("request") or {}
    timeout_sec = request_cfg.get("timeout_sec", 15)
    headers = request_cfg.get("headers") or {}

    index_url, index_html, status_code = fetch_html(target_url, headers=headers, timeout_sec=timeout_sec)
    detail_urls = extract_caa_qa_links(index_html, index_url)

    all_qa = []
    crawled_pages = [index_url]

    for i, qa in enumerate(extract_qa_from_html(index_html), start=1):
        all_qa.append({
            "question": qa.get("question", ""),
            "answer": qa.get("answer", ""),
            "source_url": index_url,
            "source_no": i,
        })

    for detail_url in detail_urls:
        fetched_url, detail_html, _ = fetch_html(detail_url, headers=headers, timeout_sec=timeout_sec)
        crawled_pages.append(fetched_url)
        qa_list = extract_qa_from_html(detail_html)

        for i, qa in enumerate(qa_list, start=1):
            all_qa.append({
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "source_url": fetched_url,
                "source_no": i,
            })

    unique = []
    seen = set()
    for item in all_qa:
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        s = (item.get("source_url") or "").strip()

        if not q or not a:
            continue

        key = (q, a, s)
        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    unique_pages = []
    seen_pages = set()
    for u in crawled_pages:
        if u not in seen_pages:
            seen_pages.add(u)
            unique_pages.append(u)

    return index_url, unique, status_code, unique_pages


@router.get("/test")
def caa_test():
    target_url, tpl = build_target_url()
    requested_url, qa_list, status_code, page_urls = fetch_index_and_detail_qa(target_url, tpl)

    return {
        "count": len(qa_list),
        "status": status_code,
        "page_count": len(page_urls),
        "sample": qa_list[:3],
        "requested_url": requested_url,
    }


def _caa_fetch_and_register_impl(authorization: str | None, target_url: str | None = None):
    uid = get_uid_from_auth_header(authorization)

    effective_target_url, tpl = build_target_url(target_url)
    requested_url, qa_list, status_code, page_urls = fetch_index_and_detail_qa(effective_target_url, tpl)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}",
        )

    local_db_path = f"/tmp/ank_{uid}_caa.db"
    db_blob.download_to_filename(local_db_path)

    created_at = datetime.now(tz=JST).isoformat()
    inserted = 0
    skipped = 0

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()
        ensure_url_tables(cur)

        root_id = get_or_create_root(cur, requested_url, created_at)

        for page_url in page_urls:
            get_or_create_page(cur, root_id, page_url, created_at)

        row_no_by_page = {}
        for qa in qa_list:
            question = (qa.get("question") or "").strip()
            answer = (qa.get("answer") or "").strip()
            source_url = (qa.get("source_url") or requested_url).strip()

            if not question or not answer:
                skipped += 1
                update_page_status(cur, source_url, "skipped")
                continue

            page_id = get_or_create_page(cur, root_id, source_url, created_at)
            row_no_by_page[source_url] = row_no_by_page.get(source_url, 0) + 1
            page_row_no = row_no_by_page[source_url]

            row_id = str(ulid.new())
            source_item_id = f"{source_url}#qa-{page_row_no}"
            content = json.dumps(
                {
                    "page_id": page_id,
                    "root_url": requested_url,
                    "page_url": source_url,
                    "question": question,
                    "answer": answer,
                },
                ensure_ascii=False,
            )

            cur.execute(
                """
                INSERT OR IGNORE INTO row_data
                  (row_id, file_id, source_type, source_key, source_item_id, row_index, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    root_id,
                    "caa",
                    requested_url,
                    source_item_id,
                    page_row_no,
                    content,
                    created_at,
                ),
            )

            if cur.rowcount == 1:
                inserted += 1
                update_page_status(cur, source_url, "done")
            else:
                skipped += 1
                update_page_status(cur, source_url, "done")

        if len(qa_list) == 0:
            for page_url in page_urls:
                update_page_status(cur, page_url, "skipped")

        conn.commit()
    finally:
        conn.close()

    db_blob.upload_from_filename(local_db_path)

    return {
        "mode": "fetch_and_register",
        "target_url": effective_target_url,
        "requested_url": requested_url,
        "root_id": root_id,
        "count": len(qa_list),
        "fetched": len(qa_list),
        "page_count": len(page_urls),
        "row_inserted": inserted,
        "row_skipped": skipped,
        "status": status_code,
        "sample": qa_list[:3],
    }


@router.get("/fetch_and_register")
def caa_fetch_and_register_get(authorization: str | None = Header(default=None)):
    return _caa_fetch_and_register_impl(authorization)


@router.post("/fetch_and_register")
async def caa_fetch_and_register_post(
    request: Request,
    authorization: str | None = Header(default=None)
):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    target_url = (body.get("target_url") or "").strip() or None
    return _caa_fetch_and_register_impl(authorization, target_url)
