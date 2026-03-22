from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import firebase_admin
from fastapi import HTTPException
from firebase_admin import auth as fb_auth
from google.cloud import storage

logger = logging.getLogger(__name__)
JST = ZoneInfo("Asia/Tokyo")
STATUS_JSON_FILENAME = "knowledge_generate.json"


# ---------- basic helpers ----------

def now_iso() -> str:
    return datetime.now(tz=JST).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


def user_status_path(uid: str) -> str:
    return f"users/{uid}/{STATUS_JSON_FILENAME}"


def local_user_db_path(uid: str) -> str:
    return f"/tmp/ank_{uid}.db"


def local_status_json_path(uid: str) -> str:
    return f"/tmp/knowledge_generate_{uid}.json"


def load_json_safe(text: str | bytes | None) -> Any | None:
    if text is None:
        return None
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except Exception:
            return None
    text = str(text).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


# ---------- auth helpers ----------

def ensure_firebase_initialized() -> None:
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


# ---------- gcs text/json helpers ----------

def load_template_text(bucket_name: str, path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path)

    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"{path} not found")

    return blob.download_as_bytes().decode("utf-8").strip()


def load_chunk_config(bucket_name: str, config_path: str) -> dict[str, Any]:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(config_path)

    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"{config_path} not found")

    try:
        obj = json.loads(blob.download_as_bytes().decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("chunk config root is not object")
        return obj
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to parse {config_path}: {e}")


# ---------- db file helpers ----------

def open_user_db(local_db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(local_db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def replace_local_db_from_blob(db_blob: storage.Blob, local_db_path: str) -> None:
    local_dir = os.path.dirname(local_db_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)

    if os.path.exists(local_db_path):
        os.remove(local_db_path)

    db_blob.download_to_filename(local_db_path)


def upload_local_db(db_blob: storage.Blob, local_db_path: str) -> None:
    if not os.path.exists(local_db_path):
        raise FileNotFoundError(f"local db not found: {local_db_path}")

    db_blob.upload_from_filename(local_db_path)
    logger.info("db uploaded to gcs: %s", db_blob.name)


# ---------- status json helpers ----------

def default_status_payload() -> dict[str, Any]:
    return {
        "updated_at": None,
        "job_id": None,
        "source_type": None,
        "status": "idle",
        "phase": None,
        "message": None,
        "error_message": None,
        "started_at": None,
        "finished_at": None,
        "dataset_id": None,
        "dataset_name": None,
        "row_count": 0,
        "knowledge_count": 0,
        "qa_current": 0,
        "qa_total": 0,
        "plain_current": 0,
        "plain_total": 0,
        "chunk_current": 0,
        "chunk_total": 0,
    }


def _validate_status_payload_shape(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="status json root must be object")

    base = default_status_payload()
    merged = dict(base)
    merged.update(payload)
    return merged


def load_status_payload(bucket: storage.Bucket, uid: str) -> dict[str, Any]:
    blob = bucket.blob(user_status_path(uid))
    if not blob.exists():
        return default_status_payload()

    payload = load_json_safe(blob.download_as_bytes())
    if payload is None:
        raise HTTPException(status_code=500, detail=f"invalid json: {user_status_path(uid)}")

    return _validate_status_payload_shape(payload)


def save_status_payload(bucket: storage.Bucket, uid: str, payload: dict[str, Any]) -> None:
    merged = _validate_status_payload_shape(payload)
    merged["updated_at"] = now_iso()

    blob = bucket.blob(user_status_path(uid))
    blob.upload_from_string(
        json.dumps(merged, ensure_ascii=False, indent=2),
        content_type="application/json; charset=utf-8",
    )


def ensure_status_payload(bucket: storage.Bucket, uid: str) -> dict[str, Any]:
    payload = load_status_payload(bucket, uid)
    blob = bucket.blob(user_status_path(uid))
    if not blob.exists():
        save_status_payload(bucket, uid, payload)
    return payload


def try_read_status_payload(bucket: storage.Bucket, uid: str) -> dict[str, Any]:
    payload = load_status_payload(bucket, uid)
    return _validate_status_payload_shape(payload)


def update_status_payload(
    bucket: storage.Bucket,
    uid: str,
    *,
    job_id: str | None = None,
    source_type: str | None = None,
    status: str | None = None,
    phase: str | None = None,
    message: str | None = None,
    error_message: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    row_count: int | None = None,
    knowledge_count: int | None = None,
    qa_current: int | None = None,
    qa_total: int | None = None,
    plain_current: int | None = None,
    plain_total: int | None = None,
    chunk_current: int | None = None,
    chunk_total: int | None = None,
) -> dict[str, Any]:
    current = ensure_status_payload(bucket, uid)

    if job_id is not None:
        current["job_id"] = job_id
    if source_type is not None:
        current["source_type"] = source_type
    if status is not None:
        current["status"] = status
    if phase is not None:
        current["phase"] = phase
    if message is not None:
        current["message"] = message
    if error_message is not None:
        current["error_message"] = error_message
    if started_at is not None:
        current["started_at"] = started_at
    if finished_at is not None:
        current["finished_at"] = finished_at
    if dataset_id is not None:
        current["dataset_id"] = dataset_id
    if dataset_name is not None:
        current["dataset_name"] = dataset_name
    if row_count is not None:
        current["row_count"] = row_count
    if knowledge_count is not None:
        current["knowledge_count"] = knowledge_count
    if qa_current is not None:
        current["qa_current"] = qa_current
    if qa_total is not None:
        current["qa_total"] = qa_total
    if plain_current is not None:
        current["plain_current"] = plain_current
    if plain_total is not None:
        current["plain_total"] = plain_total
    if chunk_current is not None:
        current["chunk_current"] = chunk_current
    if chunk_total is not None:
        current["chunk_total"] = chunk_total

    save_status_payload(bucket, uid, current)
    return current


def set_job_running(
    bucket: storage.Bucket,
    uid: str,
    job_id: str,
    source_type: str,
    *,
    phase: str | None = None,
    message: str | None = None,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    row_count: int | None = None,
    qa_total: int | None = None,
    plain_total: int | None = None,
    chunk_total: int | None = None,
) -> dict[str, Any]:
    return update_status_payload(
        bucket,
        uid,
        job_id=job_id,
        source_type=source_type,
        status="running",
        phase=phase,
        message=message,
        error_message=None,
        started_at=now_iso(),
        finished_at=None,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        row_count=row_count,
        knowledge_count=0,
        qa_current=0,
        qa_total=qa_total if qa_total is not None else 0,
        plain_current=0,
        plain_total=plain_total if plain_total is not None else 0,
        chunk_current=0,
        chunk_total=chunk_total if chunk_total is not None else 0,
    )


def set_job_progress(
    bucket: storage.Bucket,
    uid: str,
    *,
    phase: str | None = None,
    message: str | None = None,
    knowledge_count: int | None = None,
    qa_current: int | None = None,
    qa_total: int | None = None,
    plain_current: int | None = None,
    plain_total: int | None = None,
    chunk_current: int | None = None,
    chunk_total: int | None = None,
) -> dict[str, Any]:
    return update_status_payload(
        bucket,
        uid,
        status="running",
        phase=phase,
        message=message,
        knowledge_count=knowledge_count,
        qa_current=qa_current,
        qa_total=qa_total,
        plain_current=plain_current,
        plain_total=plain_total,
        chunk_current=chunk_current,
        chunk_total=chunk_total,
    )


def set_job_done(
    bucket: storage.Bucket,
    uid: str,
    job_id: str,
    *,
    phase: str | None = None,
    message: str | None = None,
    knowledge_count: int | None = None,
    qa_current: int | None = None,
    qa_total: int | None = None,
    plain_current: int | None = None,
    plain_total: int | None = None,
    chunk_current: int | None = None,
    chunk_total: int | None = None,
) -> dict[str, Any]:
    return update_status_payload(
        bucket,
        uid,
        job_id=job_id,
        status="done",
        phase=phase,
        message=message,
        error_message=None,
        finished_at=now_iso(),
        knowledge_count=knowledge_count,
        qa_current=qa_current,
        qa_total=qa_total,
        plain_current=plain_current,
        plain_total=plain_total,
        chunk_current=chunk_current,
        chunk_total=chunk_total,
    )


def set_job_error(
    bucket: storage.Bucket,
    uid: str,
    job_id: str | None,
    error_message: str,
    *,
    phase: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    return update_status_payload(
        bucket,
        uid,
        job_id=job_id,
        status="error",
        phase=phase,
        message=message,
        error_message=error_message,
        finished_at=now_iso(),
    )


def clear_job_status(bucket: storage.Bucket, uid: str) -> dict[str, Any]:
    payload = default_status_payload()
    save_status_payload(bucket, uid, payload)
    return payload


# ---------- db helpers kept for job items ----------
def fetch_job_row(local_db_path: str, job_id: str) -> sqlite3.Row | None:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT *
            FROM knowledge_jobs
            WHERE job_id = ?
            LIMIT 1
            """,
            (job_id,),
        )
        return cur.fetchone()
    finally:
        conn.close()

def fetch_job_items(local_db_path: str, job_id: str) -> list[sqlite3.Row]:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT
                job_item_id,
                job_id,
                source_type,
                parent_source_id,
                parent_key1,
                parent_key2,
                parent_label,
                row_count,
                status,
                knowledge_count,
                error_message,
                created_at,
                started_at,
                finished_at
            FROM knowledge_job_items
            WHERE job_id = ?
            ORDER BY created_at, job_item_id
            """,
            (job_id,),
        )
        return cur.fetchall()
    finally:
        conn.close()


def fetch_next_new_job_item(local_db_path: str, job_id: str) -> sqlite3.Row | None:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT
                job_item_id,
                job_id,
                source_type,
                parent_source_id,
                parent_key1,
                parent_key2,
                parent_label,
                row_count,
                status,
                knowledge_count,
                error_message,
                created_at,
                started_at,
                finished_at
            FROM knowledge_job_items
            WHERE job_id = ?
              AND status = 'new'
            ORDER BY created_at, job_item_id
            LIMIT 1
            """,
            (job_id,),
        )
        return cur.fetchone()
    finally:
        conn.close()
