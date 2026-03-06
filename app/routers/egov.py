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

router = APIRouter(prefix="/egov", tags=["egov"])

JST = ZoneInfo("Asia/Tokyo")

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
EGOV_TEMPLATE_PATH = "template/egov.json"


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


def load_egov_template() -> dict:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(EGOV_TEMPLATE_PATH)

    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"{EGOV_TEMPLATE_PATH} not found")

    text = blob.download_as_text(encoding="utf-8")
    try:
        return json.loads(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"invalid template json: {e}")


def fetch_html_from_template() -> tuple[str, str, int]:
    tpl = load_egov_template()

    request_cfg = tpl.get("request") or {}
    method = (request_cfg.get("method") or "GET").upper()
    url = request_cfg.get("url")
    timeout_sec = request_cfg.get("timeout_sec", 15)
    headers = request_cfg.get("headers") or {}

    if not url:
        raise HTTPException(status_code=400, detail="template url is empty")

    if method != "GET":
        raise HTTPException(status_code=400, detail=f"unsupported method: {method}")

    try:
        r = requests.get(url, headers=headers, timeout=timeout_sec)
        r.raise_for_status()
        html = r.text or ""
        return r.url, html, r.status_code
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"egov fetch error: {str(e)}")


def normalize_lines(text: str) -> list[str]:
    lines = [x.strip() for x in text.splitlines()]
    return [x for x in lines if x]


def extract_qa_from_text_lines(lines: list[str]) -> list[dict]:
    """
    Q. / A. 形式のFAQを抽出
    """
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


@router.get("/test")
def egov_test():
    tpl = load_egov_template()
    request_cfg = tpl.get("request") or {}
    url = request_cfg.get("url")
    timeout_sec = request_cfg.get("timeout_sec", 15)
    headers = request_cfg.get("headers") or {}

    if not url:
        raise HTTPException(status_code=400, detail="template url is empty")

    r = requests.get(url, headers=headers, timeout=timeout_sec)
    r.raise_for_status()

    html = r.text or ""
    qa_list = extract_qa_from_html(html)

    return {
        "count": len(qa_list),
        "bytes": len(html.encode("utf-8")),
        "status": r.status_code,
        "sample": qa_list[:3],
    }


def _egov_fetch_and_register_impl(authorization: str | None):
    uid = get_uid_from_auth_header(authorization)

    requested_url, html, status_code = fetch_html_from_template()
    qa_list = extract_qa_from_html(html)

    fetched = len(qa_list)
    if fetched == 0:
        return {
            "mode": "fetch_and_register",
            "requested_url": requested_url,
            "count": 0,
            "fetched": 0,
            "inserted": 0,
            "skipped": 0,
            "bytes": len(html.encode("utf-8")),
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

    local_db_path = f"/tmp/ank_{uid}_egov.db"
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

            if not question or not answer:
                skipped += 1
                row_index += 1
                continue

            row_id = str(ulid.new())
            source_item_id = f"{requested_url}#qa-{row_index}"
            content = json.dumps(
                {
                    "question": question,
                    "answer": answer,
                    "source_url": requested_url,
                },
                ensure_ascii=False,
            )

            cur.execute(
                """
                INSERT OR IGNORE INTO row_data
                  (row_id, file_id, source_type, source_key, source_item_id, row_index, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (row_id, file_id, "egov", EGOV_TEMPLATE_PATH, source_item_id, row_index, content, created_at),
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
        "bytes": len(html.encode("utf-8")),
        "status": status_code,
        "sample": qa_list[:3],
    }


@router.get("/fetch_and_register")
def egov_fetch_and_register_get(authorization: str | None = Header(default=None)):
    return _egov_fetch_and_register_impl(authorization)


@router.post("/fetch_and_register")
def egov_fetch_and_register_post(authorization: str | None = Header(default=None)):
    return _egov_fetch_and_register_impl(authorization)
