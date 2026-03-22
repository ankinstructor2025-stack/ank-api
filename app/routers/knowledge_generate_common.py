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

def default_source_status() -> dict[str, Any]:
    return {
        "job_id": None,
        "status": "idle",
        "phase": None,
        "message": None,
        "error_message": None,
        "started_at": None,
        "finished_at": None,
    }


def default_status_payload() -> dict[str, Any]:
    return {
        "updated_at": None,
        "sources": {
            "kokkai": default_source_status(),
            "opendata": default_source_status(),
            "file_upload": default_source_status(),
        },
    }


def _validate_status_payload_shape(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="status json root must be object")

    sources = payload.get("sources")
    if not isinstance(sources, dict):
        raise HTTPException(status_code=500, detail="status json 'sources' must be object")

    return payload


def load_status_payload(bucket: storage.Bucket, uid: str) -> dict[str, Any]:
    blob = bucket.blob(user_status_path(uid))
    if not blob.exists():
        return default_status_payload()

    payload = load_json_safe(blob.download_as_bytes())
    if payload is None:
        raise HTTPException(status_code=500, detail=f"invalid json: {user_status_path(uid)}")

    return _validate_status_payload_shape(payload)


def save_status_payload(bucket: storage.Bucket, uid: str, payload: dict[str, Any]) -> None:
    payload = _validate_status_payload_shape(payload)
    payload["updated_at"] = now_iso()

    blob = bucket.blob(user_status_path(uid))
    blob.upload_from_string(
        json.dumps(payload, ensure_ascii=False, indent=2),
        content_type="application/json; charset=utf-8",
    )


def ensure_status_payload(bucket: storage.Bucket, uid: str) -> dict[str, Any]:
    payload = load_status_payload(bucket, uid)
    blob = bucket.blob(user_status_path(uid))
    if not blob.exists():
        save_status_payload(bucket, uid, payload)
    return payload


def get_source_status_or_error(payload: dict[str, Any], source_type: str) -> dict[str, Any]:
    sources = payload.get("sources", {})
    if source_type not in sources:
        return {
            "job_id": None,
            "status": "error",
            "phase": None,
            "message": None,
            "error_message": f"source_type '{source_type}' not found in status json",
            "started_at": None,
            "finished_at": None,
        }

    source = sources.get(source_type)
    if not isinstance(source, dict):
        return {
            "job_id": None,
            "status": "error",
            "phase": None,
            "message": None,
            "error_message": f"source_type '{source_type}' is not object in status json",
            "started_at": None,
            "finished_at": None,
        }

    merged = default_source_status()
    merged.update(source)
    return merged


def try_read_status_payload(bucket: storage.Bucket, uid: str, source_type: str) -> dict[str, Any]:
    status = get_source_status_or_error(load_status_payload(bucket, uid), source_type)
    return {
        "job_id": status.get("job_id"),
        "status": status.get("status"),
        "phase": status.get("phase"),
        "message": status.get("message"),
        "error_message": status.get("error_message"),
        "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
    }


def _update_source_status(
    bucket: storage.Bucket,
    uid: str,
    source_type: str,
    *,
    job_id: str | None = None,
    status: str | None = None,
    phase: str | None = None,
    message: str | None = None,
    error_message: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    payload = ensure_status_payload(bucket, uid)
    sources = payload["sources"]

    if source_type not in sources or not isinstance(sources[source_type], dict):
        raise HTTPException(status_code=500, detail=f"source_type '{source_type}' not found in status json")

    current = default_source_status()
    current.update(sources[source_type])

    if job_id is not None:
        current["job_id"] = job_id
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

    sources[source_type] = current
    save_status_payload(bucket, uid, payload)
    return current


def set_source_running(
    bucket: storage.Bucket,
    uid: str,
    source_type: str,
    job_id: str,
    *,
    phase: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    return _update_source_status(
        bucket,
        uid,
        source_type,
        job_id=job_id,
        status="running",
        phase=phase,
        message=message,
        error_message=None,
        started_at=now_iso(),
        finished_at=None,
    )


def set_source_finished(
    bucket: storage.Bucket,
    uid: str,
    source_type: str,
    job_id: str,
    *,
    phase: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    return _update_source_status(
        bucket,
        uid,
        source_type,
        job_id=job_id,
        status="finished",
        phase=phase,
        message=message,
        error_message=None,
        finished_at=now_iso(),
    )


def set_source_error(
    bucket: storage.Bucket,
    uid: str,
    source_type: str,
    job_id: str | None,
    error_message: str,
    *,
    phase: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    return _update_source_status(
        bucket,
        uid,
        source_type,
        job_id=job_id,
        status="error",
        phase=phase,
        message=message,
        error_message=error_message,
        finished_at=now_iso(),
    )


def clear_source_status(bucket: storage.Bucket, uid: str, source_type: str) -> dict[str, Any]:
    payload = ensure_status_payload(bucket, uid)
    sources = payload["sources"]

    if source_type not in sources or not isinstance(sources[source_type], dict):
        raise HTTPException(status_code=500, detail=f"source_type '{source_type}' not found in status json")

    sources[source_type] = default_source_status()
    save_status_payload(bucket, uid, payload)
    return sources[source_type]


# ---------- compatibility wrappers ----------

def build_status_payload_from_db(
    local_db_path: str,
    job_id: str,
    *,
    bucket: storage.Bucket | None = None,
    uid: str | None = None,
    source_type: str | None = None,
) -> dict[str, Any]:
    if bucket is None or uid is None or source_type is None:
        raise HTTPException(
            status_code=500,
            detail="build_status_payload_from_db now requires bucket, uid and source_type",
        )
    return try_read_status_payload(bucket, uid, source_type)


def fetch_other_running_job_id(
    local_db_path: str,
    source_type: str,
    job_id: str,
    *,
    bucket: storage.Bucket | None = None,
    uid: str | None = None,
) -> str | None:
    if bucket is None or uid is None:
        raise HTTPException(
            status_code=500,
            detail="fetch_other_running_job_id now requires bucket and uid",
        )
    status = try_read_status_payload(bucket, uid, source_type)
    current_job_id = status.get("job_id")
    if status.get("status") == "running" and current_job_id and current_job_id != job_id:
        return str(current_job_id)
    return None


# ---------- db helpers kept for job items ----------
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
