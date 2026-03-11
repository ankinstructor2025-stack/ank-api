from __future__ import annotations

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
import json


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


def insert_kokkai_contents(conn: sqlite3.Connection, job_item_id: str, source_id: str):

    cur = conn.execute(
        """
        SELECT row_id, content
        FROM row_data
        WHERE source_type = 'kokkai'
        AND file_id = ?
        ORDER BY row_index
        """,
        (source_id,),
    )

    rows = cur.fetchall()

    sort_no = 1
    now = now_iso()

    for row in rows:

        try:
            data = json.loads(row["content"])
        except Exception:
            continue

        speech = data.get("speech")

        if not speech:
            continue

        conn.execute(
            """
            INSERT INTO knowledge_contents (
                job_id,
                job_item_id,
                source_type,
                source_id,
                row_id,
                content_type,
                content_text,
                sort_no,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                None,
                job_item_id,
                "kokkai",
                source_id,
                row["row_id"],
                "speech",
                speech,
                sort_no,
                now,
                now,
            ),
        )

        sort_no += 1


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
                "queued",
                selected_count,
                requested_at,
            ),
        )

        created_item_count = 0

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
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, NULL, NULL)
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
                    "queued",
                    requested_at,
                ),
            )

            created_item_count += 1
            if item.source_type == "kokkai":
                insert_kokkai_contents(conn, job_item_id, item.parent_source_id)

        conn.commit()
        db_blob.upload_from_filename(local_db_path)

        return KnowledgeJobCreateResponse(
            job_id=job_id,
            selected_count=selected_count,
            created_item_count=created_item_count,
            status="queued",
        )

    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise HTTPException(status_code=409, detail=f"duplicate or integrity error: {e}")

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()
