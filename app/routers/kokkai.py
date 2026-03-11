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


def now_iso() -> str:
    return datetime.now(tz=JST).isoformat()


def make_id() -> str:
    return str(ulid.new())


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
    return str(r.get("speechID") or r.get("speechId") or r.get("speech_id") or "").strip()


def row_source_item_id_of(r: Dict[str, Any]) -> str:
    sid = speech_id_of(r)
    return sid or ""


def house_of(r: Dict[str, Any]) -> str:
    return str(r.get("nameOfHouse") or "").strip()


def meeting_of(r: Dict[str, Any]) -> str:
    return str(r.get("nameOfMeeting") or "").strip()


def logical_name_of(house: str, meeting: str) -> str:
    parts = [x for x in [house, meeting] if x]
    return " / ".join(parts) if parts else "国会議事録"


def build_parent_source_key(house: str, meeting: str) -> str:
    return json.dumps(
        {
            "nameOfHouse": house,
            "nameOfMeeting": meeting,
        },
        ensure_ascii=False,
    )


def group_records_by_house_meeting(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for r in records:
        house = house_of(r)
        meeting = meeting_of(r)
        if not house or not meeting:
            continue

        key = (house, meeting)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    return grouped


def ensure_kokkai_documents_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS kokkai_documents (
          source_id TEXT PRIMARY KEY,
          status TEXT NOT NULL,
          logical_name TEXT NOT NULL,
          source_key TEXT,
          name_of_house TEXT NOT NULL,
          name_of_meeting TEXT NOT NULL,
          row_count INTEGER,
          source_url TEXT,
          created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_kokkai_documents_house_meeting
        ON kokkai_documents(name_of_house, name_of_meeting)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_kokkai_documents_status
        ON kokkai_documents(status)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_kokkai_documents_created_at
        ON kokkai_documents(created_at)
        """
    )


def upsert_kokkai_document(
    conn: sqlite3.Connection,
    house: str,
    meeting: str,
    requested_url: str,
    row_count: int,
) -> str:
    logical_name = logical_name_of(house, meeting)
    source_key = build_parent_source_key(house, meeting)
    created_at = now_iso()

    cur = conn.execute(
        """
        SELECT source_id
        FROM kokkai_documents
        WHERE name_of_house = ?
          AND name_of_meeting = ?
        LIMIT 1
        """,
        (house, meeting),
    )
    row = cur.fetchone()

    if row:
        source_id = row[0]
        conn.execute(
            """
            UPDATE kokkai_documents
               SET status = ?,
                   logical_name = ?,
                   source_key = ?,
                   row_count = ?,
                   source_url = ?,
                   created_at = ?
             WHERE source_id = ?
            """,
            (
                "new",
                logical_name,
                source_key,
                row_count,
                requested_url,
                created_at,
                source_id,
            ),
        )
        return str(source_id)

    source_id = make_id()
    conn.execute(
        """
        INSERT INTO kokkai_documents (
          source_id,
          status,
          logical_name,
          source_key,
          name_of_house,
          name_of_meeting,
          row_count,
          source_url,
          created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_id,
            "new",
            logical_name,
            source_key,
            house,
            meeting,
            row_count,
            requested_url,
            created_at,
        ),
    )
    return source_id


def replace_row_data_for_kokkai(
    conn: sqlite3.Connection,
    file_id: str,
    source_key: str,
    records: List[Dict[str, Any]],
) -> Tuple[int, int]:
    conn.execute(
        """
        DELETE FROM row_data
        WHERE file_id = ?
          AND source_type = 'kokkai'
        """,
        (file_id,),
    )

    inserted = 0
    skipped = 0
    created_at = now_iso()

    row_index = 1
    for r in records:
        item_id = row_source_item_id_of(r)
        if not item_id:
            skipped += 1
            continue

        row_id = make_id()
        content = json.dumps(r, ensure_ascii=False)

        conn.execute(
            """
            INSERT OR IGNORE INTO row_data
              (row_id, file_id, source_type, source_key, source_item_id, row_index, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row_id,
                file_id,
                "kokkai",
                source_key,
                item_id,
                row_index,
                content,
                created_at,
            ),
        )

        if conn.execute("SELECT changes()").fetchone()[0] == 1:
            inserted += 1
            row_index += 1
        else:
            skipped += 1

    return inserted, skipped


@router.post("/fetch_and_register")
def kokkai_fetch_and_register(
    authorization: str | None = Header(default=None),
):
    """
    template/kokkai.json の条件で speechRecord を取得し、
    nameOfHouse + nameOfMeeting 単位で kokkai_documents に親登録したうえで、
    row_data.file_id = kokkai_documents.source_id として入れ直す
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
            "count": 0,
            "document_count": 0,
            "inserted": 0,
            "skipped": 0,
        }

    grouped = group_records_by_house_meeting(records)
    if not grouped:
        return {
            "mode": "fetch_and_register",
            "requested_url": requested_url,
            "fetched": fetched,
            "count": fetched,
            "document_count": 0,
            "inserted": 0,
            "skipped": fetched,
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

    total_inserted = 0
    total_skipped = 0
    document_count = 0
    source_ids: List[str] = []

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()

        ensure_kokkai_documents_table(conn)

        for (house, meeting), group_records in grouped.items():
            source_id = upsert_kokkai_document(
                conn=conn,
                house=house,
                meeting=meeting,
                requested_url=requested_url,
                row_count=len(group_records),
            )
            source_ids.append(source_id)
            document_count += 1

            source_key = build_parent_source_key(house, meeting)

            inserted, skipped = replace_row_data_for_kokkai(
                conn=conn,
                file_id=source_id,
                source_key=source_key,
                records=group_records,
            )

            total_inserted += inserted
            total_skipped += skipped

            cur.execute(
                """
                UPDATE kokkai_documents
                   SET status = ?,
                       row_count = ?
                 WHERE source_id = ?
                """,
                ("done", inserted, source_id),
            )

        conn.commit()

    finally:
        conn.close()

    db_blob.upload_from_filename(local_db_path)

    return {
        "mode": "fetch_and_register",
        "requested_url": requested_url,
        "fetched": fetched,
        "count": fetched,
        "document_count": document_count,
        "inserted": total_inserted,
        "skipped": total_skipped,
        "source_ids": source_ids,
    }

@router.get("/documents")
def kokkai_documents(
    authorization: str | None = Header(default=None),
):
    """
    kokkai_documents の親一覧を返す
    """
    uid = get_uid_from_auth_header(authorization)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}"
        )

    local_db_path = f"/tmp/ank_{uid}_kokkai_documents.db"
    db_blob.download_to_filename(local_db_path)

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row
    try:
        ensure_kokkai_documents_table(conn)

        cur = conn.execute(
            """
            SELECT
              source_id,
              status,
              logical_name,
              source_key,
              name_of_house,
              name_of_meeting,
              row_count,
              source_url,
              created_at
            FROM kokkai_documents
            ORDER BY created_at DESC, name_of_house, name_of_meeting
            """
        )
        rows = [dict(r) for r in cur.fetchall()]

    finally:
        conn.close()

    return {
        "rows": rows,
        "count": len(rows),
    }

@router.get("/rows")
def kokkai_rows(
    name_of_house: str,
    name_of_meeting: str,
    authorization: str | None = Header(default=None),
):
    """
    name_of_house + name_of_meeting に対応する row_data を返す
    """
    uid = get_uid_from_auth_header(authorization)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}"
        )

    local_db_path = f"/tmp/ank_{uid}_kokkai_rows.db"
    db_blob.download_to_filename(local_db_path)

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row
    try:
        ensure_kokkai_documents_table(conn)

        cur = conn.execute(
            """
            SELECT source_id
            FROM kokkai_documents
            WHERE name_of_house = ?
              AND name_of_meeting = ?
            LIMIT 1
            """,
            (name_of_house, name_of_meeting),
        )
        doc = cur.fetchone()

        if not doc:
            return {"rows": [], "count": 0}

        source_id = doc["source_id"]

        cur = conn.execute(
            """
            SELECT
              row_id,
              file_id,
              source_type,
              source_key,
              source_item_id,
              row_index,
              content,
              created_at
            FROM row_data
            WHERE file_id = ?
              AND source_type = 'kokkai'
            ORDER BY row_index
            """,
            (source_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]

    finally:
        conn.close()

    return {
        "rows": rows,
        "count": len(rows),
    }
