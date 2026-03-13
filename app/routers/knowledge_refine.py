from __future__ import annotations

import os
import sqlite3
import logging
from typing import Optional, List
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge/refine", tags=["knowledge_refine"])

JST = ZoneInfo("Asia/Tokyo")
BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")


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


class RefineJobRow(BaseModel):
    job_id: str
    source_type: str
    source_name: Optional[str] = None
    request_type: Optional[str] = None
    status: str
    selected_count: int = 0
    qa_count: int = 0
    plain_count: int = 0
    error_count: int = 0
    requested_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None


class RefineJobListResponse(BaseModel):
    jobs: List[RefineJobRow]
    total_count: int


@router.get("/jobs", response_model=RefineJobListResponse)
def list_refine_jobs(
    authorization: str | None = Header(default=None),
    source_type: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    uid = get_uid_from_auth_header(authorization)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}",
        )

    local_db_path = f"/tmp/ank_{uid}_knowledge_refine.db"
    db_blob.download_to_filename(local_db_path)

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row

    try:
        where_clauses = []
        params: list[object] = []

        if source_type:
            where_clauses.append("source_type = ?")
            params.append(source_type)

        if status:
            where_clauses.append("status = ?")
            params.append(status)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        sql = f"""
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
            {where_sql}
            ORDER BY requested_at DESC, job_id DESC
            LIMIT ?
        """
        params.append(limit)

        cur = conn.execute(sql, params)
        rows = cur.fetchall()

        jobs = [
            RefineJobRow(
                job_id=row["job_id"],
                source_type=row["source_type"],
                source_name=row["source_name"],
                request_type=row["request_type"],
                status=row["status"],
                selected_count=row["selected_count"] or 0,
                qa_count=row["qa_count"] or 0,
                plain_count=row["plain_count"] or 0,
                error_count=row["error_count"] or 0,
                requested_at=row["requested_at"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
                error_message=row["error_message"],
            )
            for row in rows
        ]

        return RefineJobListResponse(
            jobs=jobs,
            total_count=len(jobs),
        )

    except sqlite3.Error as e:
        logger.exception("list_refine_jobs sqlite error")
        raise HTTPException(status_code=500, detail=f"sqlite error: {e}")

    except Exception as e:
        logger.exception("list_refine_jobs failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()
