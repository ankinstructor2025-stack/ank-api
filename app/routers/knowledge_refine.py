from __future__ import annotations

import os
import sqlite3
import logging
from typing import Optional, List
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


class RefineJobItemRow(BaseModel):
    job_item_id: str
    job_id: str
    source_item_id: Optional[str] = None
    source_type: Optional[str] = None
    title: Optional[str] = None
    status: str = "new"
    qa_count: int = 0
    plain_count: int = 0
    error_count: int = 0
    error_message: Optional[str] = None
    requested_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class RefineJobItemListResponse(BaseModel):
    job_id: str
    items: List[RefineJobItemRow]
    total_count: int


def download_user_db(uid: str) -> str:
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
    return local_db_path


def get_existing_table_name(conn: sqlite3.Connection, candidates: list[str]) -> Optional[str]:
    for table_name in candidates:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (table_name,),
        )
        if cur.fetchone():
            return table_name
    return None


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    rows = cur.fetchall()
    return {row["name"] for row in rows}


@router.get("/jobs", response_model=RefineJobListResponse)
def list_refine_jobs(
    authorization: str | None = Header(default=None),
    source_type: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    uid = get_uid_from_auth_header(authorization)
    local_db_path = download_user_db(uid)

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


@router.get("/jobs/{job_id}/items", response_model=RefineJobItemListResponse)
def list_refine_job_items(
    job_id: str,
    authorization: str | None = Header(default=None),
    limit: int = Query(default=500, ge=1, le=2000),
):
    uid = get_uid_from_auth_header(authorization)
    local_db_path = download_user_db(uid)

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row

    try:
        table_name = get_existing_table_name(
            conn,
            ["knowledge_job_items", "job_items", "knowledge_items"],
        )
        if not table_name:
            raise HTTPException(
                status_code=404,
                detail="job item table not found. expected one of: knowledge_job_items, job_items, knowledge_items",
            )

        columns = get_table_columns(conn, table_name)
        if "job_id" not in columns:
            raise HTTPException(
                status_code=500,
                detail=f"{table_name} does not have job_id column",
            )

        select_parts = []

        if "job_item_id" in columns:
            select_parts.append("job_item_id")
        else:
            select_parts.append("rowid AS job_item_id")

        select_parts.append("job_id")

        if "source_item_id" in columns:
            select_parts.append("source_item_id")
        else:
            select_parts.append("NULL AS source_item_id")

        if "source_type" in columns:
            select_parts.append("source_type")
        else:
            select_parts.append("NULL AS source_type")

        if "title" in columns:
            select_parts.append("title")
        elif "source_title" in columns:
            select_parts.append("source_title AS title")
        elif "row_title" in columns:
            select_parts.append("row_title AS title")
        elif "display_title" in columns:
            select_parts.append("display_title AS title")
        else:
            select_parts.append("NULL AS title")

        if "status" in columns:
            select_parts.append("status")
        else:
            select_parts.append("'new' AS status")

        if "qa_count" in columns:
            select_parts.append("qa_count")
        else:
            select_parts.append("0 AS qa_count")

        if "plain_count" in columns:
            select_parts.append("plain_count")
        else:
            select_parts.append("0 AS plain_count")

        if "error_count" in columns:
            select_parts.append("error_count")
        else:
            select_parts.append("0 AS error_count")

        if "error_message" in columns:
            select_parts.append("error_message")
        else:
            select_parts.append("NULL AS error_message")

        if "requested_at" in columns:
            select_parts.append("requested_at")
        else:
            select_parts.append("NULL AS requested_at")

        if "started_at" in columns:
            select_parts.append("started_at")
        else:
            select_parts.append("NULL AS started_at")

        if "finished_at" in columns:
            select_parts.append("finished_at")
        else:
            select_parts.append("NULL AS finished_at")

        order_by = []
        if "requested_at" in columns:
            order_by.append("requested_at DESC")
        if "row_index" in columns:
            order_by.append("row_index ASC")
        if "job_item_id" in columns:
            order_by.append("job_item_id ASC")
        else:
            order_by.append("rowid ASC")

        sql = f"""
            SELECT
                {", ".join(select_parts)}
            FROM {table_name}
            WHERE job_id = ?
            ORDER BY {", ".join(order_by)}
            LIMIT ?
        """

        cur = conn.execute(sql, (job_id, limit))
        rows = cur.fetchall()

        items = [
            RefineJobItemRow(
                job_item_id=str(row["job_item_id"]),
                job_id=row["job_id"],
                source_item_id=row["source_item_id"],
                source_type=row["source_type"],
                title=row["title"],
                status=row["status"] or "new",
                qa_count=row["qa_count"] or 0,
                plain_count=row["plain_count"] or 0,
                error_count=row["error_count"] or 0,
                error_message=row["error_message"],
                requested_at=row["requested_at"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
            )
            for row in rows
        ]

        return RefineJobItemListResponse(
            job_id=job_id,
            items=items,
            total_count=len(items),
        )

    except sqlite3.Error as e:
        logger.exception("list_refine_job_items sqlite error")
        raise HTTPException(status_code=500, detail=f"sqlite error: {e}")

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("list_refine_job_items failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()
