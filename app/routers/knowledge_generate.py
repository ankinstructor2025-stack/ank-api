from __future__ import annotations

import json
import os
from typing import Any, Optional

from google.cloud import storage

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.common import CLOUD_RUN_BASE_URL

from .knowledge_generate_common import (
    create_job_item_record,
    open_user_db,
    insert_opendata_contents_from_files,
    insert_upload_contents_from_files,
    insert_job_chunks,
    now_iso,
    new_id,
    download_user_db_from_gcs,
)

from .knowledge_generate_kokkai import SOURCE_TYPE as KOKKAI_SOURCE_TYPE
from .knowledge_generate_kokkai import (
    build_kokkai_chunk_rows,
)

from .knowledge_generate_opendata import SOURCE_TYPE as OPENDATA_SOURCE_TYPE
from .knowledge_generate_opendata import (
    build_opendata_chunk_rows,
)

from .knowledge_generate_upload import SOURCE_TYPE as UPLOAD_SOURCE_TYPE
from .knowledge_generate_upload import (
    build_upload_chunk_rows,
)

from .knowledge_generate_public_url import SOURCE_TYPE as PUBLIC_URL_SOURCE_TYPE
from .knowledge_generate_public_url import (
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
    source_name: Optional[str] = None
    status: str
    phase: Optional[str] = None
    selected_count: int = 0
    total_chunks: int = 0
    done_chunks: int = 0
    error_chunks: int = 0
    qa_count: int = 0
    plain_count: int = 0
    requested_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    updated_at: Optional[str] = None
    error_message: Optional[str] = None


def validate_source_type(source_type: str) -> None:
    if source_type not in SOURCE_TYPES:
        raise HTTPException(status_code=400, detail=f"unsupported source_type: {source_type}")


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
        chunk_rows = build_kokkai_chunk_rows(conn, local_db_path, job_item_id)

    elif source_type == OPENDATA_SOURCE_TYPE:
        insert_opendata_contents_from_files(
            conn=conn,
            local_db_path=local_db_path,
            job_id=job_id,
            job_item_id=job_item_id,
            source_id=item.parent_source_id or "",
        )
        chunk_rows = build_opendata_chunk_rows(conn, job_id, job_item_id)

    elif source_type == UPLOAD_SOURCE_TYPE:
        insert_upload_contents_from_files(
            conn=conn,
            local_db_path=local_db_path,
            job_id=job_id,
            job_item_id=job_item_id,
            source_id=item.parent_source_id or "",
        )
        chunk_rows = build_upload_chunk_rows(conn, job_id, job_item_id)

    elif source_type == PUBLIC_URL_SOURCE_TYPE:
        chunk_rows = build_public_url_chunk_rows(conn, local_db_path, job_item_id)

    else:
        raise HTTPException(status_code=400, detail=f"unsupported source_type: {source_type}")

    insert_job_chunks(conn, job_id, job_item_id, source_type, chunk_rows)
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


def user_job_status_path(uid: str, job_id: str) -> str:
    return f"users/{uid}/job_status/{job_id}.json"


def local_job_status_json_path(uid: str, job_id: str) -> str:
    return f"/tmp/{uid}_job_status_{job_id}.json"


def upload_job_status_json(uid: str, job_id: str, data: dict[str, Any]) -> None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    local_path = local_job_status_json_path(uid, job_id)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    blob = bucket.blob(user_job_status_path(uid, job_id))
    blob.upload_from_filename(local_path, content_type="application/json")


def download_job_status_json(uid: str, job_id: str) -> Optional[dict[str, Any]]:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(user_job_status_path(uid, job_id))
    if not blob.exists():
        return None

    text = blob.download_as_text(encoding="utf-8")
    return json.loads(text)


def _count_one(conn, sql: str, params: tuple = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    if not row:
        return 0
    return int(row[0] or 0)


def build_job_status_dict(
    conn,
    uid: str,
    job_id: str,
    forced_status: Optional[str] = None,
    forced_phase: Optional[str] = None,
    forced_error_message: Optional[str] = None,
) -> dict[str, Any]:
    job_row = conn.execute(
        """
        SELECT
            job_id,
            source_type,
            source_name,
            request_type,
            status,
            selected_count,
            requested_at,
            started_at,
            finished_at,
            error_message
        FROM knowledge_jobs
        WHERE job_id = ?
        LIMIT 1
        """,
        (job_id,),
    ).fetchone()

    if not job_row:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")

    total_chunks = _count_one(
        conn,
        "SELECT COUNT(*) FROM knowledge_job_chunks WHERE job_id = ?",
        (job_id,),
    )
    done_chunks = _count_one(
        conn,
        "SELECT COUNT(*) FROM knowledge_job_chunks WHERE job_id = ? AND status = 'done'",
        (job_id,),
    )
    error_chunks = _count_one(
        conn,
        "SELECT COUNT(*) FROM knowledge_job_chunks WHERE job_id = ? AND status = 'error'",
        (job_id,),
    )
    qa_count = _count_one(
        conn,
        "SELECT COUNT(*) FROM knowledge_items WHERE job_id = ? AND knowledge_type = 'qa'",
        (job_id,),
    )
    plain_count = _count_one(
        conn,
        "SELECT COUNT(*) FROM knowledge_items WHERE job_id = ? AND knowledge_type = 'plain'",
        (job_id,),
    )

    remaining_chunks = max(total_chunks - done_chunks - error_chunks, 0)

    if forced_status:
        status = forced_status
    else:
        if total_chunks == 0:
            status = "new"
        elif remaining_chunks > 0:
            if done_chunks == 0 and error_chunks == 0:
                status = "new"
            else:
                status = "running"
        else:
            status = "partial_error" if error_chunks > 0 else "done"

    if forced_phase:
        phase = forced_phase
    else:
        if status == "new":
            phase = "created"
        elif status == "queued":
            phase = "queued"
        elif status == "running":
            phase = "execute_chunk"
        elif status == "done":
            phase = "done"
        elif status == "partial_error":
            phase = "done_with_error"
        else:
            phase = status

    started_at = job_row["started_at"]
    finished_at = job_row["finished_at"]

    now = now_iso()

    if status == "running" and not started_at:
        started_at = now

    if status in ("done", "partial_error") and not finished_at:
        finished_at = now

    error_message = forced_error_message
    if error_message is None:
        error_message = job_row["error_message"]

    if status == "partial_error" and not error_message:
        err_row = conn.execute(
            """
            SELECT error_message
            FROM knowledge_job_chunks
            WHERE job_id = ?
              AND status = 'error'
              AND error_message IS NOT NULL
            ORDER BY chunk_no DESC
            LIMIT 1
            """,
            (job_id,),
        ).fetchone()
        if err_row:
            error_message = err_row["error_message"]

    return {
        "job_id": job_row["job_id"],
        "uid": uid,
        "source_type": job_row["source_type"],
        "source_name": job_row["source_name"],
        "request_type": job_row["request_type"],
        "status": status,
        "phase": phase,
        "selected_count": int(job_row["selected_count"] or 0),
        "total_chunks": total_chunks,
        "done_chunks": done_chunks,
        "error_chunks": error_chunks,
        "qa_count": qa_count,
        "plain_count": plain_count,
        "requested_at": job_row["requested_at"],
        "started_at": started_at,
        "finished_at": finished_at,
        "updated_at": now,
        "error_message": error_message,
    }


def write_job_status_json(
    conn,
    uid: str,
    job_id: str,
    forced_status: Optional[str] = None,
    forced_phase: Optional[str] = None,
    forced_error_message: Optional[str] = None,
) -> dict[str, Any]:
    status_data = build_job_status_dict(
        conn=conn,
        uid=uid,
        job_id=job_id,
        forced_status=forced_status,
        forced_phase=forced_phase,
        forced_error_message=forced_error_message,
    )
    upload_job_status_json(uid, job_id, status_data)
    return status_data


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
    local_user_path = download_user_db_from_gcs(uid)

    job_id = new_id()
    requested_at = now_iso()

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

        write_job_status_json(
            conn=conn,
            uid=uid,
            job_id=job_id,
            forced_status="done" if body.preview_only else "new",
            forced_phase="created",
        )

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
def run_job(body: KnowledgeRunRequest, request: Request):
    from app.core.common import load_user_queue_config
    from google.cloud import tasks_v2
    import base64

    uid = get_uid_from_auth_header(request.headers.get("Authorization"))
    job_id = body.job_id

    queue = load_user_queue_config(uid)
    queue_full_name = queue["queue_full_name"]

    client = tasks_v2.CloudTasksClient()

    local_task_path = ensure_job_task_db(uid, job_id)
    conn = open_user_db(local_task_path)

    try:
        cur = conn.execute("""
            SELECT chunk_id, chunk_no
            FROM knowledge_job_chunks
            WHERE job_id = ?
            ORDER BY chunk_no
        """, (job_id,))
        rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail=f"chunks not found: {job_id}")

        conn.execute(
            """
            UPDATE knowledge_jobs
            SET status = ?,
                started_at = COALESCE(started_at, ?),
                error_message = NULL
            WHERE job_id = ?
            """,
            ("queued", now_iso(), job_id),
        )
        conn.commit()
        upload_job_task_db(uid, job_id, local_task_path)

        write_job_status_json(
            conn=conn,
            uid=uid,
            job_id=job_id,
            forced_status="queued",
            forced_phase="queued",
            forced_error_message=None,
        )

        for row in rows:
            payload = {
                "uid": uid,
                "job_id": job_id,
                "chunk_id": row["chunk_id"]
            }

            task = {
                "http_request": {
                    "http_method": tasks_v2.HttpMethod.POST,
                    "url": f"{CLOUD_RUN_BASE_URL}/knowledge/task/execute_chunk",
                    "headers": {"Content-Type": "application/json"},
                    "body": base64.b64encode(json.dumps(payload).encode()).decode(),
                }
            }

            client.create_task(
                parent=queue_full_name,
                task=task
            )

    finally:
        conn.close()

    return {
        "job_id": job_id,
        "status": "queued"
    }


@router.post("/task/execute_chunk")
def execute_chunk(body: dict):
    from app.routers.openai_llm_client import run_llm_json

    uid = body["uid"]
    job_id = body["job_id"]
    chunk_id = body["chunk_id"]

    local_task_path = ensure_job_task_db(uid, job_id)
    conn = open_user_db(local_task_path)

    try:
        row = conn.execute("""
            SELECT *
            FROM knowledge_job_chunks
            WHERE chunk_id = ?
        """, (chunk_id,)).fetchone()

        if not row:
            return {"status": "skip"}

        prompt = row["prompt"]
        prompt_type = row["prompt_type"]

        conn.execute(
            """
            UPDATE knowledge_jobs
            SET status = ?,
                started_at = COALESCE(started_at, ?)
            WHERE job_id = ?
            """,
            ("running", now_iso(), job_id),
        )
        conn.commit()

        write_job_status_json(
            conn=conn,
            uid=uid,
            job_id=job_id,
            forced_status="running",
            forced_phase="execute_chunk",
            forced_error_message=None,
        )

        try:
            result = run_llm_json(
                prompt,
                log_prefix=f"job={job_id} chunk={row['chunk_no']}"
            )

            if prompt_type == "qa":
                items = (
                    result.get("qa_list")
                    or result.get("items")
                    or result.get("qas")
                    or result.get("data")
                    or []
                )
            else:
                items = (
                    result.get("plain_list")
                    or result.get("items")
                    or result.get("qas")
                    or result.get("data")
                    or []
                )

            print(
                f"[TASK RESPONSE] job_id={job_id} chunk_no={row['chunk_no']} "
                f"prompt_type={prompt_type} item_count={len(items)}",
                flush=True
            )

            for i, item in enumerate(items):
                created_at = now_iso()

                if prompt_type == "qa":
                    conn.execute("""
                        INSERT INTO knowledge_items (
                            knowledge_id,
                            job_id,
                            job_item_id,
                            knowledge_type,
                            question,
                            answer,
                            sort_no,
                            status,
                            review_status,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        new_id(),
                        row["job_id"],
                        row["job_item_id"],
                        "qa",
                        item.get("question"),
                        item.get("answer"),
                        i,
                        "active",
                        "new",
                        created_at
                    ))

                elif prompt_type == "plain":
                    conn.execute("""
                        INSERT INTO knowledge_items (
                            knowledge_id,
                            job_id,
                            job_item_id,
                            knowledge_type,
                            content,
                            sort_no,
                            status,
                            review_status,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        new_id(),
                        row["job_id"],
                        row["job_item_id"],
                        "plain",
                        item.get("content"),
                        i,
                        "active",
                        "new",
                        created_at
                    ))

            conn.execute("""
                UPDATE knowledge_job_chunks
                SET status = 'done',
                    error_message = NULL
                WHERE chunk_id = ?
            """, (chunk_id,))

            conn.commit()

            done_chunks = _count_one(
                conn,
                "SELECT COUNT(*) FROM knowledge_job_chunks WHERE job_id = ? AND status = 'done'",
                (job_id,),
            )
            total_chunks = _count_one(
                conn,
                "SELECT COUNT(*) FROM knowledge_job_chunks WHERE job_id = ?",
                (job_id,),
            )
            error_chunks = _count_one(
                conn,
                "SELECT COUNT(*) FROM knowledge_job_chunks WHERE job_id = ? AND status = 'error'",
                (job_id,),
            )

            if error_chunks == 0 and done_chunks == total_chunks:
                conn.execute(
                    """
                    UPDATE knowledge_jobs
                    SET status = 'done',
                        finished_at = ?,
                        error_message = NULL
                    WHERE job_id = ?
                    """,
                    (now_iso(), job_id),
                )
                conn.commit()

            upload_job_task_db(uid, job_id, local_task_path)

            status_data = write_job_status_json(
                conn=conn,
                uid=uid,
                job_id=job_id,
                forced_status=None,
                forced_phase=None,
                forced_error_message=None,
            )

            return {
                "status": status_data["status"],
                "chunk_id": chunk_id,
                "item_count": len(items)
            }

        except Exception as e:
            error_message = str(e)[:1000]

            print(
                f"[TASK ERROR] job_id={job_id} chunk_no={row['chunk_no']} "
                f"chunk_id={chunk_id} error={error_message}",
                flush=True
            )

            conn.execute("""
                UPDATE knowledge_job_chunks
                SET status = 'error',
                    error_message = ?
                WHERE chunk_id = ?
            """, (error_message, chunk_id))

            conn.commit()
            upload_job_task_db(uid, job_id, local_task_path)

            status_data = write_job_status_json(
                conn=conn,
                uid=uid,
                job_id=job_id,
                forced_status=None,
                forced_phase=None,
                forced_error_message=error_message,
            )

            return {
                "status": status_data["status"],
                "chunk_id": chunk_id,
                "error_message": error_message
            }

    finally:
        conn.close()


@router.get("/status", response_model=KnowledgeJobStatusResponse)
def get_status(request: Request, job_id: str):
    uid = get_uid_from_auth_header(request.headers.get("Authorization"))

    data = download_job_status_json(uid, job_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"job status not found: {job_id}")

    return KnowledgeJobStatusResponse(
        job_id=data["job_id"],
        source_type=data.get("source_type"),
        source_name=data.get("source_name"),
        status=data["status"],
        phase=data.get("phase"),
        selected_count=int(data.get("selected_count") or 0),
        total_chunks=int(data.get("total_chunks") or 0),
        done_chunks=int(data.get("done_chunks") or 0),
        error_chunks=int(data.get("error_chunks") or 0),
        qa_count=int(data.get("qa_count") or 0),
        plain_count=int(data.get("plain_count") or 0),
        requested_at=data.get("requested_at"),
        started_at=data.get("started_at"),
        finished_at=data.get("finished_at"),
        updated_at=data.get("updated_at"),
        error_message=data.get("error_message"),
    )
