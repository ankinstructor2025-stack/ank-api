from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

from app.core.common import local_user_db_path

JST = ZoneInfo("Asia/Tokyo")
BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
CHUNK_CONFIG_PATH = "template/openai_chunk.json"

PROMPT_TEMPLATE_PATHS = {
    "opendata": {
        "qa": "template/opendata_qa_prompt.txt",
        "plain": "template/opendata_plain_prompt.txt",
    },
    "kokkai": {
        "qa": "template/kokkai_qa_prompt.txt",
        "plain": "template/kokkai_plain_prompt.txt",
    },
    "upload": {
        "qa": "template/upload_qa_prompt.txt",
        "plain": "template/upload_plain_prompt.txt",
    },
    "public_url": {
        "qa": "template/public_url_qa_prompt.txt",
        "plain": "template/public_url_plain_prompt.txt",
    },
}

def now_iso() -> str:
    return datetime.now(tz=JST).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def open_user_db(local_db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(local_db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def open_user_db_by_uid(uid: str) -> sqlite3.Connection:
    local_db_path = local_user_db_path(uid)
    return open_user_db(local_db_path)


def load_template_text(path: str) -> str:
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)

    if not blob.exists():
        return ""

    return blob.download_as_text(encoding="utf-8")


def load_chunk_config() -> dict[str, Any]:
    from google.cloud import storage
    import json

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(CHUNK_CONFIG_PATH)

    if not blob.exists():
        return {}

    text = blob.download_as_text(encoding="utf-8")
    return json.loads(text)

# -----------------------------
# JOB
# -----------------------------
def create_job_record(
    uid: str,
    source_type: str,
    source_name: str,
    request_type: str,
    selected_count: int,
    preview_only: bool = False,
) -> tuple[str, str]:

    job_id = new_id()
    requested_at = now_iso()

    conn = open_user_db_by_uid(uid)
    try:
        conn.execute("BEGIN")
        conn.execute(
            """
            INSERT INTO knowledge_jobs (
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
            )
            VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, ?, NULL, NULL, NULL)
            """,
            (
                job_id,
                source_type,
                source_name,
                request_type,
                "done" if preview_only else "new",
                selected_count,
                requested_at,
            ),
        )
        conn.commit()
        return job_id, requested_at
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# -----------------------------
# JOB ITEM
# -----------------------------
def create_job_item_record(
    conn: sqlite3.Connection,
    job_id: str,
    source_type: str,
    parent_source_id: str | None,
    parent_key1: str | None,
    parent_key2: str | None,
    parent_label: str | None,
    row_count: int = 0,
    status: str = "new",
) -> str:

    job_item_id = new_id()

    conn.execute(
        """
        INSERT INTO knowledge_job_items (
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
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, NULL, NULL)
        """,
        (
            job_item_id,
            job_id,
            source_type,
            parent_source_id,
            parent_key1,
            parent_key2,
            parent_label,
            row_count,
            status,
            now_iso(),
        ),
    )

    return job_item_id


def insert_opendata_contents_from_files(
    conn: sqlite3.Connection,
    local_db_path: str,
    job_id: str,
    job_item_id: str,
    source_id: str,
) -> int:

    src_conn = open_user_db(local_db_path)
    try:
        cur = src_conn.execute(
            """
            SELECT
                file_id,
                file_no,
                logical_name,
                gcs_path,
                ext
            FROM opendata_document_files
            WHERE source_id = ?
            ORDER BY file_no
            """,
            (source_id,),
        )

        rows = cur.fetchall()
    finally:
        src_conn.close()

    if not rows:
        return 0

    count = 0

    for r in rows:
        gcs_path = r["gcs_path"]
        if not gcs_path:
            continue

        conn.execute(
            """
            INSERT INTO knowledge_contents (
                job_id,
                job_item_id,
                source_type,
                content_text,
                sort_no,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                job_item_id,
                "opendata",
                gcs_path,
                r["file_no"],
                now_iso(),
            ),
        )

        count += 1

    return count


def insert_upload_contents_from_files(
    conn: sqlite3.Connection,
    local_db_path: str,
    job_id: str,
    job_item_id: str,
    source_id: str,
) -> int:

    src_conn = open_user_db(local_db_path)
    try:
        row = src_conn.execute(
            """
            SELECT
                file_id,
                file_name,
                file_path,
                ext
            FROM upload_files
            WHERE file_id = ?
            LIMIT 1
            """,
            (source_id,),
        ).fetchone()
    finally:
        src_conn.close()

    if not row:
        return 0

    file_id = str(row["file_id"] or "").strip()
    file_path = str(row["file_path"] or "").strip()
    ext = str(row["ext"] or "").strip().lower()

    if not file_id:
        raise RuntimeError(f"upload_files.file_id is empty: source_id={source_id}")

    if not file_path:
        raise RuntimeError(f"upload_files.file_path is empty: source_id={source_id}")

    if ext == "text":
        ext = "txt"

    if ext not in {"pdf", "csv", "txt"}:
        raise RuntimeError(f"unsupported upload ext: {ext}")

    conn.execute(
        """
        INSERT INTO knowledge_contents (
            job_id,
            job_item_id,
            source_type,
            source_id,
            source_item_id,
            row_id,
            content_type,
            content_text,
            sort_no,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            job_item_id,
            "upload",
            file_id,
            file_id,
            file_id,
            ext,
            file_path,
            1,
            now_iso(),
            now_iso(),
        ),
    )

    return 1


# -----------------------------
# CHUNK
# -----------------------------
def insert_job_chunks(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
    source_type: str,
    chunk_rows: list[dict[str, Any]],
) -> int:

    count = 0

    for row in chunk_rows:
        conn.execute(
            """
            INSERT INTO knowledge_job_chunks (
                chunk_id,
                job_id,
                job_item_id,
                source_type,
                chunk_no,
                prompt,
                prompt_type,
                row_count,
                status,
                task_name,
                queue_id,
                retry_count,
                response_text,
                result_json,
                error_message,
                created_at,
                queued_at,
                started_at,
                finished_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("chunk_id") or new_id(),
                job_id,
                job_item_id,
                source_type,
                row.get("chunk_no", 0),
                row.get("prompt", ""),
                row.get("prompt_type", ""),
                row.get("row_count", 0),
                row.get("status", "new"),
                row.get("task_name"),
                row.get("queue_id"),
                row.get("retry_count", 0),
                row.get("response_text"),
                row.get("result_json"),
                row.get("error_message"),
                row.get("created_at", now_iso()),
                row.get("queued_at"),
                row.get("started_at"),
                row.get("finished_at"),
            ),
        )
        count += 1

    return count
