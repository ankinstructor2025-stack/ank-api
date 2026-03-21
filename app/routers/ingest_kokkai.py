# app/routers/ingest_kokkai.py

import os
import json
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Header, Query
import requests
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth

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


def _fetch_json(url: str, params: dict) -> Tuple[dict, str]:
    try:
        res = requests.get(url, params=params, timeout=30)
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


def _is_speech(url: str) -> bool:
    return "/api/speech" in (url or "").lower()


def _is_meeting_list(url: str) -> bool:
    return "/api/meeting_list" in (url or "").lower()


def _record_key_from_url(url: str) -> str:
    if _is_speech(url):
        return "speechRecord"
    if _is_meeting_list(url):
        return "meetingRecord"
    return "meetingRecord"


def _clamp_maximum_records(url: str, value: Any) -> int:
    try:
        n = int(value)
    except Exception:
        n = 10

    if _is_speech(url):
        return max(1, min(100, n))
    if _is_meeting_list(url):
        return max(1, min(100, n))
    return max(1, min(10, n))


def fetch_records(url: str, params: dict, fetch_cfg: dict) -> Tuple[str, List[Dict[str, Any]]]:
    safe_params = dict(params or {})
    safe_fetch = dict(fetch_cfg or {})

    safe_params.setdefault("recordPacking", "json")
    safe_params["maximumRecords"] = _clamp_maximum_records(url, safe_params.get("maximumRecords", 10))

    record_key = _record_key_from_url(url)

    all_mode = bool(safe_fetch.get("all", False))
    try:
        max_total = int(safe_fetch.get("max_total", safe_params["maximumRecords"]))
    except Exception:
        max_total = safe_params["maximumRecords"]
    if max_total < 1:
        max_total = 1

    if not all_mode:
        data, requested_url = _fetch_json(url, safe_params)
        return requested_url, _extract_records(data, record_key)

    collected: List[Dict[str, Any]] = []
    requested_url_last = ""
    start = int(safe_params.get("startRecord", 1) or 1)

    while len(collected) < max_total:
        safe_params["startRecord"] = start
        data, requested_url_last = _fetch_json(url, safe_params)

        records = _extract_records(data, record_key)
        if not records:
            break

        remain = max_total - len(collected)
        collected.extend(records[:remain])

        nxt = _next_record_position(data)
        if not nxt:
            break
        start = nxt

    return requested_url_last, collected


def issue_id_of(r: Dict[str, Any]) -> str:
    return str(r.get("issueID") or r.get("issueId") or r.get("issue_id") or "").strip()


def speech_id_of(r: Dict[str, Any]) -> str:
    return str(r.get("speechID") or r.get("speechId") or r.get("speech_id") or "").strip()


def house_of(r: Dict[str, Any]) -> str:
    return str(r.get("nameOfHouse") or "").strip()


def meeting_of(r: Dict[str, Any]) -> str:
    return str(r.get("nameOfMeeting") or "").strip()


def speaker_of(r: Dict[str, Any]) -> str:
    return str(r.get("speaker") or "").strip()


def speech_of(r: Dict[str, Any]) -> str:
    return str(r.get("speech") or "").strip()


def speech_order_of(r: Dict[str, Any]) -> int:
    v = r.get("speechOrder")
    try:
        return int(v)
    except Exception:
        return 0


def speech_date_of(r: Dict[str, Any]) -> str:
    return str(r.get("date") or "").strip()


def logical_name_of(house: str, meeting: str, issue_id: str) -> str:
    parts = [x for x in [house, meeting] if x]
    if parts:
        return " / ".join(parts)
    return issue_id or "国会議事録"


def ensure_kokkai_documents_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS kokkai_documents (
          issue_id TEXT PRIMARY KEY,
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
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_kokkai_documents_house_meeting
        ON kokkai_documents(name_of_house, name_of_meeting)
        """
    )


def ensure_kokkai_document_rows_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS kokkai_document_rows (
          issue_id TEXT NOT NULL,
          speech_id TEXT NOT NULL,
          status TEXT NOT NULL,
          speaker TEXT,
          speech TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT,
          PRIMARY KEY (issue_id, speech_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_kokkai_document_rows_status
        ON kokkai_document_rows(status)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_kokkai_document_rows_speaker
        ON kokkai_document_rows(speaker)
        """
    )


def ensure_schema(conn: sqlite3.Connection) -> None:
    ensure_kokkai_documents_table(conn)
    ensure_kokkai_document_rows_table(conn)

    # 旧UNIQUEインデックスがあれば除去
    conn.execute("DROP INDEX IF EXISTS ux_kokkai_documents_house_meeting")

    # 旧row_data前提のインデックス名が残っていても無視できるように削除
    conn.execute("DROP INDEX IF EXISTS ix_kokkai_document_rows_source_id")
    conn.execute("DROP INDEX IF EXISTS ix_kokkai_document_rows_issue_id")
    conn.execute("DROP INDEX IF EXISTS ix_kokkai_document_rows_meeting_date")
    conn.execute("DROP INDEX IF EXISTS ux_kokkai_document_rows_source_order")


def upsert_kokkai_document(
    conn: sqlite3.Connection,
    issue_id: str,
    house: str,
    meeting: str,
    requested_url: str,
    row_count: int,
) -> None:
    created_at = now_iso()
    logical_name = logical_name_of(house, meeting, issue_id)
    source_key = json.dumps(
        {
            "issueID": issue_id,
            "nameOfHouse": house,
            "nameOfMeeting": meeting,
        },
        ensure_ascii=False,
    )

    conn.execute(
        """
        INSERT INTO kokkai_documents (
          issue_id,
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
        ON CONFLICT(issue_id) DO UPDATE SET
          status = excluded.status,
          logical_name = excluded.logical_name,
          source_key = excluded.source_key,
          name_of_house = excluded.name_of_house,
          name_of_meeting = excluded.name_of_meeting,
          row_count = excluded.row_count,
          source_url = excluded.source_url,
          created_at = excluded.created_at
        """,
        (
            issue_id,
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


def replace_kokkai_document_rows(
    conn: sqlite3.Connection,
    issue_id: str,
    records: List[Dict[str, Any]],
) -> Tuple[int, int]:
    conn.execute(
        """
        DELETE FROM kokkai_document_rows
        WHERE issue_id = ?
        """,
        (issue_id,),
    )

    inserted = 0
    skipped = 0
    created_at = now_iso()

    sorted_records = sorted(records, key=lambda r: (speech_order_of(r), speech_id_of(r)))

    for r in sorted_records:
        speech_id = speech_id_of(r)
        speech_text = speech_of(r)
        speech_order = speech_order_of(r)

        if not speech_id or not speech_text:
            skipped += 1
            continue

        conn.execute(
            """
            INSERT INTO kokkai_document_rows (
              issue_id,
              speech_id,
              speech_order,
              status,
              speaker,
              speech,
              created_at,
              updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                issue_id,
                speech_id,
                speech_order,
                "new",
                speaker_of(r),
                speech_text,
                created_at,
                created_at,
            ),
        )
        inserted += 1

    return inserted, skipped


def load_template_parts() -> Tuple[dict, dict]:
    tpl = load_kokkai_template()

    list_cfg = dict(tpl.get("list") or {})
    detail_cfg = dict(tpl.get("detail") or {})

    if not list_cfg:
        raise HTTPException(status_code=400, detail="kokkai.json missing list")
    if not detail_cfg:
        raise HTTPException(status_code=400, detail="kokkai.json missing detail")

    if not list_cfg.get("endpoint"):
        raise HTTPException(status_code=400, detail="kokkai.json list.endpoint missing")
    if not detail_cfg.get("endpoint"):
        raise HTTPException(status_code=400, detail="kokkai.json detail.endpoint missing")
    if not detail_cfg.get("key"):
        raise HTTPException(status_code=400, detail="kokkai.json detail.key missing")

    return list_cfg, detail_cfg


def fetch_list_records_from_template() -> Tuple[str, List[Dict[str, Any]]]:
    list_cfg, _ = load_template_parts()
    url = str(list_cfg.get("endpoint"))
    params = dict(list_cfg.get("params") or {})
    fetch_cfg = dict(list_cfg.get("fetch") or {})
    return fetch_records(url, params, fetch_cfg)


def fetch_detail_records_from_template(issue_id: str) -> Tuple[str, List[Dict[str, Any]]]:
    _, detail_cfg = load_template_parts()
    url = str(detail_cfg.get("endpoint"))
    key = str(detail_cfg.get("key"))
    params = dict(detail_cfg.get("params") or {})
    params[key] = issue_id
    fetch_cfg = dict(detail_cfg.get("fetch") or {})
    return fetch_records(url, params, fetch_cfg)


@router.post("/fetch_and_register")
def kokkai_fetch_and_register(
    authorization: str | None = Header(default=None),
):
    """
    template/kokkai.json の list/detail 設定で
    meeting_list -> speech を取得し、issue_idベースで親子テーブルへ登録する
    """
    uid = get_uid_from_auth_header(authorization)

    requested_url_list, meetings = fetch_list_records_from_template()

    if not meetings:
        return {
            "mode": "fetch_and_register",
            "requested_url": requested_url_list,
            "meeting_count": 0,
            "document_count": 0,
            "row_inserted": 0,
            "row_skipped": 0,
            "issue_ids": [],
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

    document_count = 0
    total_inserted = 0
    total_skipped = 0
    issue_ids: List[str] = []

    conn = sqlite3.connect(local_db_path)
    try:
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)

        for meeting_rec in meetings:
            issue_id = issue_id_of(meeting_rec)
            if not issue_id:
                continue

            house = house_of(meeting_rec)
            meeting = meeting_of(meeting_rec)

            requested_url_detail, speech_records = fetch_detail_records_from_template(issue_id)

            upsert_kokkai_document(
                conn=conn,
                issue_id=issue_id,
                house=house,
                meeting=meeting,
                requested_url=requested_url_detail or requested_url_list,
                row_count=len(speech_records),
            )

            inserted, skipped = replace_kokkai_document_rows(
                conn=conn,
                issue_id=issue_id,
                records=speech_records,
            )

            actual_row_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM kokkai_document_rows
                WHERE issue_id = ?
                """,
                (issue_id,),
            ).fetchone()[0]

            conn.execute(
                """
                UPDATE kokkai_documents
                   SET status = ?,
                       row_count = ?
                 WHERE issue_id = ?
                """,
                ("done", actual_row_count, issue_id),
            )

            document_count += 1
            total_inserted += inserted
            total_skipped += skipped
            issue_ids.append(issue_id)

        conn.commit()

    finally:
        conn.close()

    db_blob.upload_from_filename(local_db_path)

    return {
        "mode": "fetch_and_register",
        "requested_url": requested_url_list,
        "meeting_count": len(meetings),
        "document_count": document_count,
        "row_inserted": total_inserted,
        "row_skipped": total_skipped,
        "issue_ids": issue_ids,
    }


@router.get("/documents")
def kokkai_documents(
    authorization: str | None = Header(default=None),
):
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
        ensure_schema(conn)

        cur = conn.execute(
            """
            SELECT
              d.issue_id,
              d.status,
              d.logical_name,
              d.source_key,
              d.name_of_house,
              d.name_of_meeting,
              d.row_count,
              d.source_url,
              d.created_at
            FROM kokkai_documents d
            ORDER BY d.created_at DESC
            """
        )
        rows = [dict(r) for r in cur.fetchall()]

    finally:
        conn.close()

    return {
        "rows": rows,
        "count": len(rows)
    }


@router.get("/rows")
def kokkai_rows(
    issue_id: str | None = Query(default=None),
    name_of_house: str | None = Query(default=None),
    name_of_meeting: str | None = Query(default=None),
    authorization: str | None = Header(default=None),
):
    """
    issue_id 指定を優先して、その会議の speech 一覧を返す
    旧UI互換として name_of_house + name_of_meeting 指定も許容
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
        ensure_schema(conn)

        target_issue_id = (issue_id or "").strip()

        if not target_issue_id:
            if not name_of_house or not name_of_meeting:
                raise HTTPException(
                    status_code=400,
                    detail="issue_id or (name_of_house and name_of_meeting) is required"
                )

            cur = conn.execute(
                """
                SELECT issue_id
                FROM kokkai_documents
                WHERE name_of_house = ?
                  AND name_of_meeting = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (name_of_house, name_of_meeting),
            )
            doc = cur.fetchone()
            if not doc:
                return {"rows": [], "count": 0}
            target_issue_id = str(doc["issue_id"])

        cur = conn.execute(
            """
            SELECT
              issue_id,
              speech_id,
              speech_order,
              status,
              speaker,
              speech,
              created_at,
              updated_at
            FROM kokkai_document_rows
            WHERE issue_id = ?
            ORDER BY COALESCE(speech_order, 0), speech_id
            """,
            (target_issue_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]

    finally:
        conn.close()

    return {
        "rows": rows,
        "count": len(rows),
        "issue_id": target_issue_id,
    }
