from __future__ import annotations

import os
from typing import Any, Optional
from google.cloud import storage

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .knowledge_generate_common import (
    create_job_record,
    create_job_item_record,
    open_user_db,
)

from .knowledge_generate_kokkai import SOURCE_TYPE as KOKKAI_SOURCE_TYPE
from .knowledge_generate_kokkai import (
    fetch_kokkai_source_rows,
    insert_kokkai_contents,
    build_kokkai_chunk_rows,
)
from .knowledge_generate_opendata import SOURCE_TYPE as OPENDATA_SOURCE_TYPE
from .knowledge_generate_opendata import (
    fetch_opendata_file_rows,
    insert_opendata_contents,
    build_opendata_chunk_rows,
)
from .knowledge_generate_upload import SOURCE_TYPE as UPLOAD_SOURCE_TYPE
from .knowledge_generate_upload import (
    fetch_upload_file_row,
    insert_upload_contents,
    build_upload_chunk_rows,
)
from .knowledge_generate_public_url import SOURCE_TYPE as PUBLIC_URL_SOURCE_TYPE
from .knowledge_generate_public_url import (
    fetch_url_page_rows,
    insert_url_contents,
    build_public_url_chunk_rows,
)

from app.core.common import (
    local_user_db_path,
    user_task_db_path,
    local_task_db_path,
)
from app.routers.user_init import get_uid_from_auth_header

router = APIRouter(prefix="/knowledge", tags=["knowledge_generate"])


BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
TASK_TEMPLATE_DB_PATH = "template/task_template.sqlite"


SOURCE_TYPES = {
    KOKKAI_SOURCE_TYPE,
    OPENDATA_SOURCE_TYPE,
    UPLOAD_SOURCE_TYPE,
    PUBLIC_URL_SOURCE_TYPE,
}


class KnowledgeTargetItem(BaseModel):
    source_type: str
    parent_source_id: Optional[str] = None
    parent_key1: Optional[str] = None
    parent_key2: Optional[str] = None
    parent_label: Optional[str] = None
    row_count: int = 0


class KnowledgeJobCreateRequest(BaseModel):
    source_type: str
    source_name: Optional[str] = None
    request_type: str = "extract_knowledge"
    items: list[KnowledgeTargetItem] = Field(default_factory=list)
    preview_only: bool = False

class KnowledgeRunRequest(BaseModel):
    job_id: str


class KnowledgeJobCreateResponse(BaseModel):
    job_id: str
    requested_at: str
    selected_count: int
    status: str


class KnowledgeJobStatusResponse(BaseModel):
    job_id: str
    source_type: Optional[str] = None
    status: str
    phase: Optional[str] = None
    selected_count: int = 0
    qa_count: int = 0
    plain_count: int = 0
    error_count: int = 0
    requested_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None


def validate_source_type(source_type: str) -> None:
    if source_type not in SOURCE_TYPES:
        raise HTTPException(status_code=400, detail=f"unsupported source_type: {source_type}")


def insert_job_chunks(
    conn,
    job_id: str,
    job_item_id: str,
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
                chunk_no,
                prompt_type,
                prompt,
                status,
                retry_count,
                task_name,
                queue_id,
                response_text,
                result_json,
                error_message,
                created_at,
                queued_at,
                started_at,
                finished_at
            )
            VALUES (
                hex(randomblob(16)),
                ?, ?, ?, ?, ?, ?, 0,
                NULL, NULL, NULL, NULL, NULL,
                CURRENT_TIMESTAMP, NULL, NULL, NULL
            )
            """,
            (
                job_id,
                job_item_id,
                row.get("chunk_no", 0),
                row.get("prompt_type", ""),
                row.get("prompt", ""),
                row.get("status", "new"),
            ),
        )
        count += 1
    return count


def prepare_job_item(conn, local_db_path: str, job_id: str, item: KnowledgeTargetItem) -> str:
    source_type = item.source_type
    validate_source_type(source_type)

    job_item_id = create_job_item_record(
        conn=conn,
        job_id=job_id,
        source_type=source_type,
        parent_source_id=item.parent_source_id,
        parent_key1=item.parent_key1,
        parent_key2=item.parent_key2,
        parent_label=item.parent_label,
        row_count=item.row_count,
        status="new",
    )

    if source_type == KOKKAI_SOURCE_TYPE:
        source_rows = fetch_kokkai_source_rows(local_db_path, item.parent_source_id or "")
        insert_kokkai_contents(conn, job_id, job_item_id, source_rows)
        chunk_rows = build_kokkai_chunk_rows(conn, job_item_id)

    elif source_type == OPENDATA_SOURCE_TYPE:
        source_rows = fetch_opendata_file_rows(local_db_path, item.parent_source_id or "")
        insert_opendata_contents(conn, job_id, job_item_id, source_rows)
        chunk_rows = build_opendata_chunk_rows(conn, job_item_id)

    elif source_type == UPLOAD_SOURCE_TYPE:
        file_row = fetch_upload_file_row(local_db_path, item.parent_source_id or "")
        insert_upload_contents(conn, job_id, job_item_id, file_row)
        chunk_rows = build_upload_chunk_rows(conn, job_item_id)

    elif source_type == PUBLIC_URL_SOURCE_TYPE:
        source_rows = fetch_url_page_rows(local_db_path, item.parent_source_id or "")
        insert_url_contents(conn, job_id, job_item_id, source_rows)
        chunk_rows = build_public_url_chunk_rows(conn, job_item_id)

    else:
        raise HTTPException(status_code=400, detail=f"unsupported source_type: {source_type}")

    insert_job_chunks(conn, job_id, job_item_id, chunk_rows)
    return job_item_id


def ensure_job_task_db(uid: str, job_id: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    src_blob = bucket.blob(TASK_TEMPLATE_DB_PATH)
    if not src_blob.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Task template DB not found: gs://{BUCKET_NAME}/{TASK_TEMPLATE_DB_PATH}"
        )

    task_gcs_path = user_task_db_path(uid, job_id)
    task_blob = bucket.blob(task_gcs_path)

    if not task_blob.exists():
        bucket.copy_blob(src_blob, bucket, new_name=task_gcs_path)

    local_task_path = local_task_db_path(uid, job_id)
    os.makedirs(os.path.dirname(local_task_path), exist_ok=True)

    bucket.blob(task_gcs_path).download_to_filename(local_task_path)
    return local_task_path


def upload_job_task_db(uid: str, job_id: str, local_path: str) -> None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    task_gcs_path = user_task_db_path(uid, job_id)
    blob = bucket.blob(task_gcs_path)

    blob.upload_from_filename(local_path)


def insert_task_job_record(
    conn,
    job_id: str,
    source_type: str,
    source_name: str,
    request_type: str,
    selected_count: int,
    preview_only: bool,
    requested_at: str,
) -> None:
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


@router.post("/job", response_model=KnowledgeJobCreateResponse)
def create_job(request: Request, body: KnowledgeJobCreateRequest):
    validate_source_type(body.source_type)

    if not body.items:
        raise HTTPException(status_code=400, detail="items is empty")

    uid = get_uid_from_auth_header(request.headers.get("Authorization"))
    local_user_path = local_user_db_path(uid)

    job_id, requested_at = create_job_record(
        uid=uid,
        source_type=body.source_type,
        source_name=body.source_name or body.source_type,
        request_type=body.request_type,
        selected_count=len(body.items),
        preview_only=body.preview_only,
    )

    local_task_path = ensure_job_task_db(uid, job_id)

    conn = open_user_db(local_task_path)
    try:
        conn.execute("BEGIN")

        insert_task_job_record(
            conn=conn,
            job_id=job_id,
            source_type=body.source_type,
            source_name=body.source_name or body.source_type,
            request_type=body.request_type,
            selected_count=len(body.items),
            preview_only=body.preview_only,
            requested_at=requested_at,
        )

        for item in body.items:
            prepare_job_item(conn, local_user_path, job_id, item)
        conn.commit()

        upload_job_task_db(uid, job_id, local_task_path)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return KnowledgeJobCreateResponse(
        job_id=job_id,
        requested_at=requested_at,
        selected_count=len(body.items),
        status="done" if body.preview_only else "new",
    )


@router.post("/run")
def run_job(body: KnowledgeRunRequest):
    # 第一段階: まだ実行しない
    return {
        "job_id": body.job_id,
        "status": "accepted",
        "message": "第一段階の仮実装です。CHUNK作成後の実行は未実装です。",
    }


@router.get("/status", response_model=KnowledgeJobStatusResponse)
def get_status(uid: str, job_id: str):
    local_db_path = local_user_db_path(uid)

    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT
                job_id,
                source_type,
                status,
                phase,
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
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"job not found: {job_id}")

        return KnowledgeJobStatusResponse(
            job_id=row["job_id"],
            source_type=row["source_type"],
            status=row["status"],
            phase=row["phase"],
            selected_count=row["selected_count"],
            qa_count=row["qa_count"],
            plain_count=row["plain_count"],
            error_count=row["error_count"],
            requested_at=row["requested_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            error_message=row["error_message"],
        )
    finally:
        conn.close()
