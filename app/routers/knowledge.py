from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth


router = APIRouter(prefix="/knowledge", tags=["knowledge"])

JST = ZoneInfo("Asia/Tokyo")
BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")


def now_iso() -> str:
    return datetime.now(tz=JST).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


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


def ensure_knowledge_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_jobs (
            job_id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            source_name TEXT,
            request_type TEXT NOT NULL,
            status TEXT NOT NULL,
            selected_count INTEGER NOT NULL DEFAULT 0,
            qa_count INTEGER NOT NULL DEFAULT 0,
            plain_count INTEGER NOT NULL DEFAULT 0,
            error_count INTEGER NOT NULL DEFAULT 0,
            requested_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            error_message TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_knowledge_jobs_status
        ON knowledge_jobs(status)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_knowledge_jobs_requested_at
        ON knowledge_jobs(requested_at)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_job_items (
            job_item_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            parent_source_id TEXT,
            parent_key1 TEXT,
            parent_key2 TEXT,
            parent_label TEXT,
            row_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            knowledge_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            FOREIGN KEY (job_id) REFERENCES knowledge_jobs(job_id)
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_knowledge_job_items_job_id
        ON knowledge_job_items(job_id)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_knowledge_job_items_status
        ON knowledge_job_items(status)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_contents (
            job_id TEXT NOT NULL,
            job_item_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_id TEXT,
            source_item_id TEXT,
            row_id TEXT,
            content_type TEXT NOT NULL,
            content_text TEXT NOT NULL,
            sort_no INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_knowledge_contents_job_item
        ON knowledge_contents(job_item_id)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_items (
            knowledge_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            job_item_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_id TEXT,
            source_item_id TEXT,
            row_id TEXT,
            knowledge_type TEXT NOT NULL,
            title TEXT,
            question TEXT,
            answer TEXT,
            content TEXT,
            summary TEXT,
            keywords TEXT,
            language TEXT DEFAULT 'ja',
            sort_no INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active',
            review_status TEXT NOT NULL DEFAULT 'new',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_knowledge_items_job_id
        ON knowledge_items(job_id)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_knowledge_items_job_item_id
        ON knowledge_items(job_item_id)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_knowledge_items_type
        ON knowledge_items(knowledge_type)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_knowledge_items_source
        ON knowledge_items(source_type, source_id)
        """
    )


def load_json_safe(text: str) -> dict:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def extract_speech_text(content_obj: dict) -> str:
    return normalize_text(content_obj.get("speech"))


def extract_speech_id(content_obj: dict) -> str | None:
    v = content_obj.get("speechID")
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def insert_kokkai_contents(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
    source_id: str,
) -> int:
    cur = conn.execute(
        """
        SELECT row_id, row_index, content
        FROM row_data
        WHERE source_type = 'kokkai'
          AND file_id = ?
        ORDER BY row_index
        """,
        (source_id,),
    )
    rows = cur.fetchall()

    inserted_count = 0
    now = now_iso()

    for idx, row in enumerate(rows, start=1):
        content_obj = load_json_safe(row["content"] or "")
        speech_text = extract_speech_text(content_obj)
        if not speech_text:
            continue

        source_item_id = extract_speech_id(content_obj)

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
                "kokkai",
                source_id,
                source_item_id,
                row["row_id"],
                "speech",
                speech_text,
                idx,
                now,
                now,
            ),
        )
        inserted_count += 1

    return inserted_count


def insert_plain_items_from_contents(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
    source_type: str,
    source_id: str,
) -> int:
    cur = conn.execute(
        """
        SELECT
            source_item_id,
            row_id,
            content_type,
            content_text,
            sort_no
        FROM knowledge_contents
        WHERE job_id = ?
          AND job_item_id = ?
        ORDER BY sort_no
        """,
        (job_id, job_item_id),
    )
    rows = cur.fetchall()

    inserted_count = 0
    now = now_iso()

    for row in rows:
        if (row["content_type"] or "") != "speech":
            continue

        conn.execute(
            """
            INSERT INTO knowledge_items (
                knowledge_id,
                job_id,
                job_item_id,
                source_type,
                source_id,
                source_item_id,
                row_id,
                knowledge_type,
                title,
                question,
                answer,
                content,
                summary,
                keywords,
                language,
                sort_no,
                status,
                review_status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id(),
                job_id,
                job_item_id,
                source_type,
                source_id,
                row["source_item_id"],
                row["row_id"],
                "plain",
                None,
                None,
                None,
                row["content_text"],
                None,
                None,
                "ja",
                row["sort_no"],
                "active",
                "new",
                now,
                now,
            ),
        )
        inserted_count += 1

    return inserted_count


class KnowledgeTargetItem(BaseModel):
    source_type: str = Field(..., description="kokkai / opendata / public_url / upload")
    parent_source_id: Optional[str] = None
    parent_key1: Optional[str] = None
    parent_key2: Optional[str] = None
    parent_label: Optional[str] = None
    row_count: int = 0


class KnowledgeJobCreateRequest(BaseModel):
    source_type: str = Field(..., description="kokkai / opendata / public_url / upload")
    source_name: Optional[str] = None
    request_type: str = "extract_knowledge"
    items: List[KnowledgeTargetItem]


class KnowledgeJobCreateResponse(BaseModel):
    job_id: str
    selected_count: int
    created_item_count: int
    status: str


@router.post("/jobs", response_model=KnowledgeJobCreateResponse)
def create_knowledge_job(
    body: KnowledgeJobCreateRequest,
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)

    if not body.items:
        raise HTTPException(status_code=400, detail="items is empty")

    if body.source_type != "kokkai":
        raise HTTPException(
            status_code=400,
            detail="currently only source_type='kokkai' is supported",
        )

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}",
        )

    local_db_path = f"/tmp/ank_{uid}_knowledge.db"
    db_blob.download_to_filename(local_db_path)

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row

    try:
        ensure_knowledge_tables(conn)

        job_id = new_id()
        requested_at = now_iso()

        unique_items: List[KnowledgeTargetItem] = []
        seen_keys = set()

        for item in body.items:
            key = (
                item.source_type or "",
                item.parent_source_id or "",
                item.parent_key1 or "",
                item.parent_key2 or "",
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_items.append(item)

        selected_count = len(unique_items)

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
                body.source_type,
                body.source_name,
                body.request_type,
                "running",
                selected_count,
                requested_at,
            ),
        )

        created_item_count = 0
        total_plain_count = 0
        total_qa_count = 0
        total_error_count = 0

        for item in unique_items:
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
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, NULL)
                """,
                (
                    job_item_id,
                    job_id,
                    item.source_type,
                    item.parent_source_id,
                    item.parent_key1,
                    item.parent_key2,
                    item.parent_label,
                    item.row_count,
                    "running",
                    requested_at,
                    requested_at,
                ),
            )

            try:
                contents_count = 0
                plain_count = 0

                if item.source_type == "kokkai":
                    source_id = item.parent_source_id or ""
                    contents_count = insert_kokkai_contents(
                        conn=conn,
                        job_id=job_id,
                        job_item_id=job_item_id,
                        source_id=source_id,
                    )
                    plain_count = insert_plain_items_from_contents(
                        conn=conn,
                        job_id=job_id,
                        job_item_id=job_item_id,
                        source_type="kokkai",
                        source_id=source_id,
                    )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"unsupported item.source_type: {item.source_type}",
                    )

                finished_at = now_iso()

                conn.execute(
                    """
                    UPDATE knowledge_job_items
                    SET status = ?,
                        knowledge_count = ?,
                        finished_at = ?,
                        error_message = NULL
                    WHERE job_item_id = ?
                    """,
                    (
                        "done",
                        plain_count,
                        finished_at,
                        job_item_id,
                    ),
                )

                created_item_count += 1
                total_plain_count += plain_count

            except Exception as e:
                finished_at = now_iso()
                total_error_count += 1

                conn.execute(
                    """
                    UPDATE knowledge_job_items
                    SET status = ?,
                        finished_at = ?,
                        error_message = ?
                    WHERE job_item_id = ?
                    """,
                    (
                        "error",
                        finished_at,
                        str(e),
                        job_item_id,
                    ),
                )

        finished_at = now_iso()
        final_status = "done" if total_error_count == 0 else "partial_error"

        conn.execute(
            """
            UPDATE knowledge_jobs
            SET status = ?,
                qa_count = ?,
                plain_count = ?,
                error_count = ?,
                started_at = COALESCE(started_at, ?),
                finished_at = ?
            WHERE job_id = ?
            """,
            (
                final_status,
                total_qa_count,
                total_plain_count,
                total_error_count,
                requested_at,
                finished_at,
                job_id,
            ),
        )

        conn.commit()
        db_blob.upload_from_filename(local_db_path)

        return KnowledgeJobCreateResponse(
            job_id=job_id,
            selected_count=selected_count,
            created_item_count=created_item_count,
            status=final_status,
        )

    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise HTTPException(status_code=409, detail=f"duplicate or integrity error: {e}")

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()
