from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import sqlite3
import uuid

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect("ank.db")
    conn.row_factory = sqlite3.Row
    return conn


def require_auth_user(authorization: str | None) -> dict:
    if not authorization:
        raise HTTPException(status_code=401, detail="authorization header is required")
    return {"ok": True}


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
    require_auth_user(authorization)

    if not body.items:
        raise HTTPException(status_code=400, detail="items is empty")

    if body.source_type != "kokkai":
        raise HTTPException(
            status_code=400,
            detail="currently only source_type='kokkai' is supported",
        )

    conn = get_db_connection()

    try:
        job_id = new_id()
        requested_at = now_iso()

        unique_items = []
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

        conn.commit()

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
