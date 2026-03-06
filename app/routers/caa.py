from fastapi import APIRouter, HTTPException, Header
import os
import json
import sqlite3
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urljoin

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


def extract_qa_from_text_lines(lines: list[str]) -> list[dict]:
    results = []
    q = None
    a_lines = []
    mode = None

    for line in lines:
        if re.match(r"^Q[\.．]?\s*", line):
            if q and a_lines:
                results.append({
                    "question": re.sub(r"^Q[\.．]?\s*", "", q).strip(),
                    "answer": "\n".join(a_lines).strip()
                })
            q = line
            a_lines = []
            mode = "q"
            continue

        if re.match(r"^A[\.．]?\s*", line):
            mode = "a"
            ans = re.sub(r"^A[\.．]?\s*", "", line).strip()
            if ans:
                a_lines.append(ans)
            continue

        if mode == "q" and q:
            q += " " + line
        elif mode == "a":
            a_lines.append(line)

    if q and a_lines:
        results.append({
            "question": re.sub(r"^Q[\.．]?\s*", "", q).strip(),
            "answer": "\n".join(a_lines).strip()
        })

    return results


def extract_qa_from_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    lines = normalize_lines(text)
    return extract_qa_from_text_lines(lines)


def extract_caa_qa_links(index_html: str, base_url: str) -> list[str]:
    """
    一覧ページから Q&A ページのリンクを抽出
    """
    soup = BeautifulSoup(index_html, "html.parser")
    urls = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        label = a.get_text(" ", strip=True)

        full_url = urljoin(base_url, href)

        if "Q&A" in label or "Q＆A" in label or "FAQ" in label:
            urls.append(full_url)
            continue

        if "/faq/" in full_url:
            urls.append(full_url)
            continue

    # 同一URL除去
    unique_urls = []
    seen = set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)

    return unique_urls


def fetch_index_and_detail_qa_from_template() -> tuple[str, list[dict], int]:
    tpl = load_caa_template()

    request_cfg = tpl.get("request") or {}
    method = (request_cfg.get("method") or "GET").upper()
    url = request_cfg.get("url")
    timeout_sec = request_cfg.get("timeout_sec", 15)
    headers = request_cfg.get("headers") or {}

    if not url:
        raise HTTPException(status_code=400, detail="template url is empty")

    if method != "GET":
        raise HTTPException(status_code=400, detail=f"unsupported method: {method}")

    index_url, index_html, status_code = fetch_html(url, headers=headers, timeout_sec=timeout_sec)
    detail_urls = extract_caa_qa_links(index_html, index_url)

    all_qa = []

    for detail_url in detail_urls:
        fetched_url, detail_html, _ = fetch_html(detail_url, headers=headers, timeout_sec=timeout_sec)
        qa_list = extract_qa_from_html(detail_html)

        for i, qa in enumerate(qa_list, start=1):
            all_qa.append({
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "source_url": fetched_url,
                "source_no": i,
            })

    return index_url, all_qa, status_code


@router.get("/test")
def caa_test():
    requested_url, qa_list, status_code = fetch_index_and_detail_qa_from_template()

    return {
        "count": len(qa_list),
        "status": status_code,
        "sample": qa_list[:3],
        "requested_url": requested_url,
    }


def _caa_fetch_and_register_impl(authorization: str | None):
    uid = get_uid_from_auth_header(authorization)

    requested_url, qa_list, status_code = fetch_index_and_detail_qa_from_template()

    fetched = len(qa_list)
    if fetched == 0:
        return {
            "mode": "fetch_and_register",
            "requested_url": requested_url,
            "count": 0,
            "fetched": 0,
            "inserted": 0,
            "skipped": 0,
            "status": status_code,
        }

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

    file_id = str(ulid.new())
    created_at = datetime.now(tz=JST).isoformat()

    inserted = 0
    skipped = 0

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()

        row_index = 1
        for qa in qa_list:
            question = (qa.get("question") or "").strip()
            answer = (qa.get("answer") or "").strip()
            source_url = (qa.get("source_url") or requested_url).strip()

            if not question or not answer:
                skipped += 1
                row_index += 1
                continue

            row_id = str(ulid.new())
            source_item_id = f"{source_url}#qa-{row_index}"
            content = json.dumps(
                {
                    "question": question,
                    "answer": answer,
                    "source_url": source_url,
                },
                ensure_ascii=False,
            )

            cur.execute(
                """
                INSERT OR IGNORE INTO row_data
                  (row_id, file_id, source_type, source_key, source_item_id, row_index, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (row_id, file_id, "caa", CAA_TEMPLATE_PATH, source_item_id, row_index, content, created_at),
            )

            if cur.rowcount == 1:
                inserted += 1
            else:
                skipped += 1

            row_index += 1

        conn.commit()
    finally:
        conn.close()

    db_blob.upload_from_filename(local_db_path)

    return {
        "mode": "fetch_and_register",
        "requested_url": requested_url,
        "file_id": file_id,
        "count": fetched,
        "fetched": fetched,
        "inserted": inserted,
        "skipped": skipped,
        "status": status_code,
        "sample": qa_list[:3],
    }


@router.get("/fetch_and_register")
def caa_fetch_and_register_get(authorization: str | None = Header(default=None)):
    return _caa_fetch_and_register_impl(authorization)


@router.post("/fetch_and_register")
def caa_fetch_and_register_post(authorization: str | None = Header(default=None)):
    return _caa_fetch_and_register_impl(authorization)
