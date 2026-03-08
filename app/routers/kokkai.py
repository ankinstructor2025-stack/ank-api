# app/routers/kokkai.py

import os
import json
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Header
import requests
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth

import ulid

router = APIRouter(prefix="/kokkai", tags=["kokkai"])

JST = ZoneInfo("Asia/Tokyo")

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
KOKKAI_TEMPLATE_PATH = "template/kokkai.json"


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
        raise HTTPException(status_code=401, detail=f"Invalid ID token: {e}")


def load_kokkai_template() -> dict:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(KOKKAI_TEMPLATE_PATH)

    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"{KOKKAI_TEMPLATE_PATH} not found")

    text = blob.download_as_text(encoding="utf-8")
    try:
        return json.loads(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"invalid template json: {e}")


def _is_speech(url: str) -> bool:
    return "/api/speech" in (url or "").lower()


def _record_key(url: str) -> str:
    return "speechRecord" if _is_speech(url) else "meetingRecord"


def _clamp_maximum_records(url: str, value: Any) -> int:
    try:
        n = int(value)
    except Exception:
        n = 10

    if _is_speech(url):
        return max(1, min(100, n))
    return max(1, min(10, n))


def _fetch_json(url: str, params: dict) -> Tuple[dict, str]:
    try:
        res = requests.get(url, params=params, timeout=20)
        res.raise_for_status()
        return res.json(), res.url
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"kokkai api error: {str(e)}")


def _extract_records(data: dict, key: str) -> List[Dict[str, Any]]:
    v = data.get(key)
    return v if isinstance(v, list) else []


def _next_record_position(data: dict) -> Optional[int]:
    v = data.get("nextRecordPosition")
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def fetch_records_from_template() -> Tuple[str, str, List[Dict[str, Any]]]:
    tpl = load_kokkai_template()

    url = tpl.get("endpoint") or "https://kokkai.ndl.go.jp/api/meeting"
    params = dict(tpl.get("params") or {})
    fetch_cfg = dict(tpl.get("fetch") or {})

    params.setdefault("recordPacking", "json")
    params["maximumRecords"] = _clamp_maximum_records(url, params.get("maximumRecords", 10))

    key = _record_key(url)

    all_mode = bool(fetch_cfg.get("all", False))
    try:
        max_total = int(fetch_cfg.get("max_total", params["maximumRecords"]))
    except Exception:
        max_total = params["maximumRecords"]
    if max_total < 1:
        max_total = 1

    if not all_mode:
        data, requested_url = _fetch_json(url, params)
        records = _extract_records(data, key)
        return requested_url, key, records

    collected: List[Dict[str, Any]] = []
    start = int(params.get("startRecord", 1) or 1)
    requested_url_last = ""

    while len(collected) < max_total:
        params["startRecord"] = start

        data, requested_url_last = _fetch_json(url, params)
        records = _extract_records(data, key)

        if not records:
            break

        remain = max_total - len(collected)
        collected.extend(records[:remain])

        nxt = _next_record_position(data)
        if not nxt:
            break
        start = nxt

    return requested_url_last, key, collected


def speech_id_of(r: Dict[str, Any]) -> str:
    return str(r.get("speechID") or r.get("speechId") or r.get("speech_id") or "")


def row_source_item_id_of(r: Dict[str, Any]) -> str:
    """
    row_data 側の一意識別子
    国会議事録は speechID を使う
    """
    sid = speech_id_of(r)
    if sid:
        return sid
    return ""


def build_source_key(records: List[Dict[str, Any]]) -> str:
    """
    row_data.source_key に入れる取得条件の簡易表現
    """
    if not records:
        return "kokkai"

    first = records[0]
    house = str(first.get("nameOfHouse") or "")
    meeting = str(first.get("nameOfMeeting") or "")

    parts = [p for p in [house, meeting] if p]
    return " / ".join(parts) if parts else "kokkai"


@router.post("/fetch_and_register")
def kokkai_fetch_and_register(
    authorization: str | None = Header(default=None),
):
    """
    取得テスト + row_data登録
    - template/kokkai.json の条件で取得
    - source_documents は使わない
    - row_data のみ登録
    - source_item_id には speechID を入れる
    """
    uid = get_uid_from_auth_header(authorization)

    requested_url, record_type, records = fetch_records_from_template()

    if record_type != "speechRecord":
        raise HTTPException(
            status_code=400,
            detail=f"record_type must be speechRecord. got={record_type}"
        )

    fetched = len(records)
    if fetched == 0:
        return {
            "mode": "fetch_and_register",
            "requested_url": requested_url,
            "fetched": 0,
            "inserted": 0,
            "skipped": 0,
        }

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}"
        )

    local_db_path = f"/tmp/ank_{uid}_kokkai.db"
    db_blob.download_to_filename(local_db_path)

    file_id = str(ulid.new())
    created_at = datetime.now(tz=JST).isoformat()
    source_type = "kokkai"
    source_key = build_source_key(records)

    inserted = 0
    skipped = 0

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()

        row_index = 1
        for r in records:
            item_id = row_source_item_id_of(r)
            if not item_id:
                skipped += 1
                row_index += 1
                continue

            row_id = str(ulid.new())
            content = json.dumps(r, ensure_ascii=False)

            cur.execute(
                """
                INSERT OR IGNORE INTO row_data
                  (row_id, file_id, source_type, source_key, source_item_id, row_index, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    file_id,
                    source_type,
                    source_key,
                    item_id,
                    row_index,
                    content,
                    created_at,
                ),
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
        "fetched": fetched,
        "count": fetched,
        "inserted": inserted,
        "skipped": skipped,
        "source_key": source_key,
    }
