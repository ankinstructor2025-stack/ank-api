from __future__ import annotations

import json
import os
import sqlite3
import uuid
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

from fastapi import HTTPException
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth


logger = logging.getLogger(__name__)
JST = ZoneInfo("Asia/Tokyo")
LOCK_TTL_SECONDS = 60 * 60 * 6


def now_iso() -> str:
    return datetime.now(tz=JST).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


def local_user_db_path(uid: str) -> str:
    return f"/tmp/ank_{uid}.db"


def row_to_status_item(row: sqlite3.Row) -> dict[str, Any]:
    qa_chunk_total = int(row["qa_chunk_total"] or 0)
    qa_chunk_done = int(row["qa_chunk_done"] or 0)
    plain_chunk_total = int(row["plain_chunk_total"] or 0)
    plain_chunk_done = int(row["plain_chunk_done"] or 0)

    return {
        "job_item_id": row["job_item_id"],
        "parent_source_id": row["parent_source_id"],
        "parent_label": row["parent_label"],
        "status": row["status"] or "",
        "knowledge_count": int(row["knowledge_count"] or 0),
        "error_message": row["error_message"],
        "row_count": int(row["row_count"] or 0),
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "qa_chunk_total": qa_chunk_total,
        "qa_chunk_done": qa_chunk_done,
        "plain_chunk_total": plain_chunk_total,
        "plain_chunk_done": plain_chunk_done,
        "chunk_total": qa_chunk_total + plain_chunk_total,
        "chunk_done": qa_chunk_done + plain_chunk_done,
    }

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


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def load_json_safe(text: str) -> dict | list | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def flatten_json_like(value: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []

    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            lines.extend(flatten_json_like(v, key))
        return lines

    if isinstance(value, list):
        for idx, item in enumerate(value):
            key = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            lines.extend(flatten_json_like(item, key))
        return lines

    text = normalize_text(str(value) if value is not None else "")
    if not text:
        return []

    if prefix:
        return [f"{prefix}: {text}"]
    return [text]


def extract_row_text(content_raw: str | None) -> str:
    src = (content_raw or "").strip()
    if not src:
        return ""

    parsed = load_json_safe(src)
    if parsed is None:
        return normalize_text(src)

    lines = flatten_json_like(parsed)
    if not lines:
        return normalize_text(src)

    return "\n".join(lines).strip()


def load_template_text(bucket_name: str, path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path)

    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"{path} not found")

    return blob.download_as_bytes().decode("utf-8").strip()


def load_chunk_config(bucket_name: str, config_path: str) -> dict:
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


def open_user_db(local_db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(local_db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def upload_local_db(db_blob: storage.Blob, local_db_path: str) -> None:
    if not os.path.exists(local_db_path):
        raise FileNotFoundError(f"local db not found: {local_db_path}")

    snapshot_path = f"{local_db_path}.upload.sqlite"

    src_conn = None
    dst_conn = None

    try:
        src_conn = sqlite3.connect(local_db_path, timeout=30)
        src_conn.row_factory = sqlite3.Row
        src_conn.execute("PRAGMA busy_timeout = 30000")

        journal_mode = src_conn.execute("PRAGMA journal_mode").fetchone()[0]
        logger.info("upload_local_db journal_mode=%s path=%s", journal_mode, local_db_path)

        if str(journal_mode).lower() == "wal":
            src_conn.execute("PRAGMA wal_checkpoint(FULL)")
            logger.info("upload_local_db wal checkpoint done: %s", local_db_path)

        dst_conn = sqlite3.connect(snapshot_path, timeout=30)
        src_conn.backup(dst_conn)
        dst_conn.commit()

    finally:
        if dst_conn is not None:
            dst_conn.close()
        if src_conn is not None:
            src_conn.close()

    db_blob.upload_from_filename(snapshot_path)
    logger.info("db uploaded to gcs: %s", db_blob.name)

    try:
        os.remove(snapshot_path)
    except Exception:
        logger.warning("failed to remove snapshot file: %s", snapshot_path)


def ensure_job_locks_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_locks (
            lock_key TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            locked_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
        """
    )


def build_lock_key(uid: str, source_type: str) -> str:
    return f"{uid}:{source_type}"


def try_acquire_job_lock(local_db_path: str, lock_key: str, job_id: str) -> bool:
    conn = open_user_db(local_db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        ensure_job_locks_table(conn)
        now = now_iso()
        expires_at = datetime.fromisoformat(now).timestamp() + LOCK_TTL_SECONDS
        expires_iso = datetime.fromtimestamp(expires_at, tz=JST).isoformat()
        conn.execute("DELETE FROM job_locks WHERE expires_at < ?", (now,))
        try:
            conn.execute(
                "INSERT INTO job_locks (lock_key, job_id, locked_at, expires_at) VALUES (?, ?, ?, ?)",
                (lock_key, job_id, now, expires_iso),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            conn.rollback()
            return False
    finally:
        conn.close()


def release_job_lock(local_db_path: str, lock_key: str, job_id: str | None = None) -> None:
    conn = open_user_db(local_db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        ensure_job_locks_table(conn)
        if job_id:
            conn.execute("DELETE FROM job_locks WHERE lock_key = ? AND job_id = ?", (lock_key, job_id))
        else:
            conn.execute("DELETE FROM job_locks WHERE lock_key = ?", (lock_key,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_running_lock_job_id(local_db_path: str, lock_key: str) -> str | None:
    conn = open_user_db(local_db_path)
    try:
        ensure_job_locks_table(conn)
        now = now_iso()
        conn.execute("DELETE FROM job_locks WHERE expires_at < ?", (now,))
        conn.commit()
        cur = conn.execute("SELECT job_id FROM job_locks WHERE lock_key = ? LIMIT 1", (lock_key,))
        row = cur.fetchone()
        return row["job_id"] if row else None
    finally:
        conn.close()


def fetch_job_row(local_db_path: str, job_id: str) -> sqlite3.Row | None:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT
                job_id,
                source_type,
                source_name,
                request_type,
                status,
                selected_count,
                qa_count,
                plain_count,
                error_count,
                requested_at,
                started_at,
                finished_at,
                error_message
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
                qa_chunk_total,
                qa_chunk_done,
                plain_chunk_total,
                plain_chunk_done,
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


def build_status_payload_from_db(local_db_path: str, job_id: str) -> dict[str, Any]:
    job_row = fetch_job_row(local_db_path, job_id)
    if not job_row:
        raise HTTPException(status_code=404, detail=f"knowledge_jobs not found: {job_id}")

    item_rows = fetch_job_items(local_db_path, job_id)
    items = [row_to_status_item(row) for row in item_rows]

    total_qa_chunks = sum(int(item["qa_chunk_total"] or 0) for item in items)
    done_qa_chunks = sum(int(item["qa_chunk_done"] or 0) for item in items)
    total_plain_chunks = sum(int(item["plain_chunk_total"] or 0) for item in items)
    done_plain_chunks = sum(int(item["plain_chunk_done"] or 0) for item in items)

    return {
        "job_id": job_row["job_id"],
        "status": job_row["status"] or "",
        "selected_count": int(job_row["selected_count"] or 0),
        "qa_count": int(job_row["qa_count"] or 0),
        "plain_count": int(job_row["plain_count"] or 0),
        "error_count": int(job_row["error_count"] or 0),
        "requested_at": job_row["requested_at"],
        "started_at": job_row["started_at"],
        "finished_at": job_row["finished_at"],
        "error_message": job_row["error_message"],
        "total_qa_chunks": total_qa_chunks,
        "processed_qa_chunks": done_qa_chunks,
        "total_plain_chunks": total_plain_chunks,
        "processed_plain_chunks": done_plain_chunks,
        "total_chunks": total_qa_chunks + total_plain_chunks,
        "processed_chunks": done_qa_chunks + done_plain_chunks,
        "items": items,
    }


def fetch_other_running_job_id(local_db_path: str, source_type: str, job_id: str) -> str | None:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT job_id
            FROM knowledge_jobs
            WHERE source_type = ?
              AND status = 'running'
              AND job_id <> ?
            ORDER BY requested_at
            LIMIT 1
            """,
            (source_type, job_id),
        )
        row = cur.fetchone()
        return row["job_id"] if row else None
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
