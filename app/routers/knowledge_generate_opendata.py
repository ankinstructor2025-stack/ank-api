from __future__ import annotations

import json
import os
import sqlite3
import logging
from typing import List, Optional, Any

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field
from google.cloud import storage

from .knowledge_generate_common import (
    fetch_job_row,
    fetch_next_new_job_item,
    get_uid_from_auth_header,
    load_chunk_config,
    load_template_text,
    local_user_db_path,
    new_id,
    now_iso,
    open_user_db,
    upload_local_db,
    user_db_path,
    replace_local_db_from_blob,
    try_read_status_payload,
    set_job_queued,
    set_job_running,
    set_job_progress,
    set_job_done,
    set_job_error,
)

from .openai_llm_client import run_chunked_llm_json
from .openai_chunking import ChunkConfig, build_chunks
from .openai_prompt_builder import build_opendata_prompt_text
from .content_splitter_pdf import split_pdf_records
from .content_detector import normalize_text
from app.core.common import enqueue_knowledge_job

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge/opendata", tags=["knowledge_opendata"])

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
SOURCE_TYPE = "opendata"

OPENDATA_QA_PROMPT_PATH = "template/opendata_qa_prompt.txt"
OPENDATA_PLAIN_PROMPT_PATH = "template/opendata_plain_prompt.txt"
OPENAI_CHUNK_CONFIG_PATH = "template/openai_chunk.json"
WORKER_SHARED_TOKEN = os.getenv("WORKER_SHARED_TOKEN", "")


class KnowledgeJobStatusResponse(BaseModel):
    updated_at: Optional[str] = None
    job_id: Optional[str] = None
    source_type: Optional[str] = None
    status: str = "idle"
    phase: Optional[str] = None
    message: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    row_count: int = 0
    knowledge_count: int = 0
    qa_current: int = 0
    qa_total: int = 0
    plain_current: int = 0
    plain_total: int = 0
    chunk_current: int = 0
    chunk_total: int = 0

def get_required_opendata_chunk_conf(chunk_config: dict, prompt_type: str) -> ChunkConfig:
    opendata_conf = chunk_config.get("opendata")
    if not isinstance(opendata_conf, dict):
        raise HTTPException(status_code=500, detail="openai_chunk.json: opendata section not found")

    conf = opendata_conf.get(prompt_type)
    if not isinstance(conf, dict):
        raise HTTPException(status_code=500, detail=f"openai_chunk.json: opendata.{prompt_type} section not found")

    missing = [key for key in ("max_chars", "max_items", "overlap_items") if key not in conf]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"openai_chunk.json: opendata.{prompt_type} missing keys: {', '.join(missing)}",
        )

    try:
        max_chars = int(conf["max_chars"])
        max_items = int(conf["max_items"])
        overlap_items = int(conf["overlap_items"])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"openai_chunk.json: invalid opendata.{prompt_type} values: {e}",
        )

    if max_chars <= 0:
        raise HTTPException(status_code=500, detail=f"openai_chunk.json: opendata.{prompt_type}.max_chars must be > 0")
    if max_items <= 0:
        raise HTTPException(status_code=500, detail=f"openai_chunk.json: opendata.{prompt_type}.max_items must be > 0")
    if overlap_items < 0:
        raise HTTPException(status_code=500, detail=f"openai_chunk.json: opendata.{prompt_type}.overlap_items must be >= 0")

    return ChunkConfig(max_chars=max_chars, max_items=max_items, overlap_items=overlap_items)










































LOCK_TTL_SECONDS = 60 * 60 * 6


def verify_worker_token(x_worker_token: str | None) -> None:
    if not WORKER_SHARED_TOKEN:
        return
    if x_worker_token != WORKER_SHARED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid worker token")


def fetch_opendata_file_rows(local_db_path: str, source_id: str) -> list[sqlite3.Row]:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT
                file_id,
                source_id,
                file_no,
                logical_name,
                original_name,
                source_url,
                gcs_path,
                ext,
                file_size,
                created_at
            FROM opendata_document_files
            WHERE source_id = ?
            ORDER BY file_no, file_id
            """,
            (source_id,),
        )
        return cur.fetchall()
    finally:
        conn.close()








def fetch_opendata_job_item_meta(conn: sqlite3.Connection, job_item_id: str) -> sqlite3.Row:
    cur = conn.execute(
        """
        SELECT
            ji.job_item_id,
            ji.parent_source_id,
            ji.parent_label,
            ji.parent_key1,
            ji.parent_key2,
            ji.row_count
        FROM knowledge_job_items ji
        WHERE ji.job_item_id = ?
        LIMIT 1
        """,
        (job_item_id,),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"knowledge_job_items not found: {job_item_id}")
    return row


def fetch_opendata_content_rows(conn: sqlite3.Connection, job_item_id: str) -> list[sqlite3.Row]:
    cur = conn.execute(
        """
        SELECT
            source_item_id,
            row_id,
            content_type,
            content_text,
            sort_no
        FROM knowledge_contents
        WHERE job_item_id = ?
        ORDER BY sort_no
        """,
        (job_item_id,),
    )
    return cur.fetchall()


def build_opendata_prompt_texts(
    conn: sqlite3.Connection,
    job_item_id: str,
    template_text: str,
    chunk_conf: ChunkConfig,
) -> list[str]:
    item = fetch_opendata_job_item_meta(conn, job_item_id)
    rows = fetch_opendata_content_rows(conn, job_item_id)

    chunks = build_chunks(
        rows,
        chunk_conf,
        allowed_content_types={"row"},
    )

    if not chunks:
        raise HTTPException(status_code=400, detail=f"knowledge_contents not found: {job_item_id}")

    prompt_template = template_text.strip()

    prompt_texts: list[str] = []
    for chunk in chunks:
        prompt_texts.append(
            build_opendata_prompt_text(
                job_item_id=job_item_id,
                prompt_template=prompt_template,
                chunk=chunk,
                parent_source_id=item["parent_source_id"],
                parent_key1=item["parent_key1"],
                parent_key2=item["parent_key2"],
                parent_label=item["parent_label"],
                row_count=item["row_count"],
            )
        )

    logger.info(
        "opendata prompt build: job_item_id=%s chunk_count=%s",
        job_item_id,
        len(prompt_texts),
    )
    return prompt_texts


def join_prompt_previews(prompt_texts: list[str]) -> str:
    blocks: list[str] = []
    for idx, prompt_text in enumerate(prompt_texts, start=1):
        blocks.append(f"===== CHUNK {idx} / {len(prompt_texts)} =====\n{prompt_text}")
    return "\n\n".join(blocks).strip()


def merge_qa_chunk_results(job_item_id: str, chunk_results: list[dict]) -> dict:
    merged: dict[str, Any] = {
        "job_item_id": job_item_id,
        "qa_list": [],
        "chunk_count": len(chunk_results),
    }

    for idx, result in enumerate(chunk_results, start=1):
        if not isinstance(result, dict):
            continue

        result_job_item_id = result.get("job_item_id")
        if result_job_item_id not in (None, "", job_item_id):
            raise Exception(f"qa job_item_id mismatch at chunk {idx}")

        qa_list = result.get("qa_list") or []
        if not isinstance(qa_list, list):
            raise Exception(f"qa_list is not list at chunk {idx}")

        merged["qa_list"].extend(qa_list)

    return merged


def merge_plain_chunk_results(job_item_id: str, chunk_results: list[dict]) -> dict:
    merged: dict[str, Any] = {
        "job_item_id": job_item_id,
        "plain_list": [],
        "chunk_count": len(chunk_results),
    }

    for idx, result in enumerate(chunk_results, start=1):
        if not isinstance(result, dict):
            continue

        result_job_item_id = result.get("job_item_id")
        if result_job_item_id not in (None, "", job_item_id):
            raise Exception(f"plain job_item_id mismatch at chunk {idx}")

        plain_list = result.get("plain_list") or []
        if not isinstance(plain_list, list):
            raise Exception(f"plain_list is not list at chunk {idx}")

        merged["plain_list"].extend(plain_list)

    return merged


def insert_qa_items_from_llm_result(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
    source_id: str,
    llm_result: dict,
) -> int:
    qa_list = llm_result.get("qa_list") or []
    if not isinstance(qa_list, list):
        raise Exception("qa_list is not list")

    inserted_count = 0
    now = now_iso()
    sort_no = 200000

    for qa in qa_list:
        if not isinstance(qa, dict):
            continue

        question_raw = normalize_text(qa.get("question"))
        answer_raw = normalize_text(qa.get("answer"))

        if not question_raw or not answer_raw:
            continue

        content_raw = f"[Q]\n{question_raw}\n\n[A]\n{answer_raw}"

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
                question_normalize,
                answer_normalize,
                content_normalize,
                question_vector,
                answer_vector,
                content_vector,
                summary,
                keywords,
                language,
                sort_no,
                status,
                review_status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id(),
                job_id,
                job_item_id,
                SOURCE_TYPE,
                source_id,
                None,
                None,
                "qa",
                None,
                question_raw,
                answer_raw,
                content_raw,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "ja",
                sort_no,
                "active",
                "new",
                now,
                now,
            ),
        )

        inserted_count += 1
        sort_no += 1

    logger.info(
        "insert_qa_items_from_llm_result: job_item_id=%s qa_count=%s",
        job_item_id,
        inserted_count,
    )
    return inserted_count


def insert_plain_items_from_llm_result(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
    source_id: str,
    llm_result: dict,
) -> int:
    plain_list = llm_result.get("plain_list") or []
    if not isinstance(plain_list, list):
        raise Exception("plain_list is not list")

    inserted_count = 0
    now = now_iso()
    sort_no = 300000

    for item in plain_list:
        if not isinstance(item, dict):
            continue

        content_raw = normalize_text(item.get("content"))
        if not content_raw:
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
                question_normalize,
                answer_normalize,
                content_normalize,
                question_vector,
                answer_vector,
                content_vector,
                summary,
                keywords,
                language,
                sort_no,
                status,
                review_status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id(),
                job_id,
                job_item_id,
                SOURCE_TYPE,
                source_id,
                None,
                None,
                "plain",
                None,
                None,
                None,
                content_raw,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "ja",
                sort_no,
                "active",
                "new",
                now,
                now,
            ),
        )

        inserted_count += 1
        sort_no += 1

    logger.info(
        "insert_plain_items_from_llm_result: job_item_id=%s plain_count=%s",
        job_item_id,
        inserted_count,
    )
    return inserted_count


def insert_opendata_contents(
    conn: sqlite3.Connection,
    bucket: storage.Bucket,
    job_id: str,
    job_item_id: str,
    source_id: str,
    file_rows: list[sqlite3.Row],
) -> int:
    inserted_count = 0
    now = now_iso()
    sort_no = 1

    for file_row in file_rows:
        ext = normalize_text(file_row["ext"]).lower()
        if ext != "pdf":
            continue

        gcs_path = file_row["gcs_path"]
        if not gcs_path:
            continue

        blob = bucket.blob(gcs_path)
        if not blob.exists():
            logger.warning("opendata file not found in gcs: %s", gcs_path)
            continue

        try:
            binary = blob.download_as_bytes()
            pages = split_pdf_records(binary)
        except Exception as e:
            logger.warning("pdf split failed: file_id=%s error=%s", file_row["file_id"], e)
            continue

        for page in pages:
            text = normalize_text(page.get("text"))
            if not text:
                continue

            page_no = int(page.get("page") or 0)
            source_item_id = f"{file_row['file_id']}#page={page_no}"
            row_id = f"{file_row['file_id']}:{page_no}"

            content_text = f"[FILE] {file_row['original_name'] or ''}\n[PAGE] {page_no}\n{text}"

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
                    SOURCE_TYPE,
                    source_id,
                    source_item_id,
                    row_id,
                    "row",
                    content_text,
                    sort_no,
                    now,
                    now,
                ),
            )
            inserted_count += 1
            sort_no += 1

    return inserted_count


def create_job_record(
    local_db_path: str,
    body: "KnowledgeJobCreateRequest",
    selected_count: int,
) -> tuple[str, str]:
    job_id = new_id()
    requested_at = now_iso()

    conn = open_user_db(local_db_path)
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
                SOURCE_TYPE,
                body.source_name or "オープンデータ",
                body.request_type,
                "done" if body.preview_only else "new",
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


def prepare_job_item(
    local_db_path: str,
    bucket: storage.Bucket,
    job_id: str,
    item: "KnowledgeTargetItem",
    requested_at: str,
    preview_only: bool,
) -> dict[str, Any]:
    source_id = item.parent_source_id or ""
    if not source_id:
        raise HTTPException(status_code=400, detail="parent_source_id is required")

    file_rows = fetch_opendata_file_rows(local_db_path, source_id)
    if not file_rows:
        raise HTTPException(status_code=400, detail=f"opendata_document_files not found: {source_id}")

    job_item_id = new_id()

    conn = open_user_db(local_db_path)
    try:
        conn.execute("BEGIN")

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
                SOURCE_TYPE,
                item.parent_source_id,
                item.parent_key1,
                item.parent_key2,
                item.parent_label,
                item.row_count,
                "done" if preview_only else "new",
                requested_at,
            ),
        )

        inserted_rows = insert_opendata_contents(
            conn=conn,
            bucket=bucket,
            job_id=job_id,
            job_item_id=job_item_id,
            source_id=source_id,
            file_rows=file_rows,
        )

        if inserted_rows <= 0:
            raise HTTPException(status_code=400, detail=f"pdf text not found: {source_id}")

        conn.execute(
            """
            UPDATE knowledge_job_items
            SET row_count = ?
            WHERE job_item_id = ?
            """,
            (inserted_rows, job_item_id),
        )

        conn.commit()

        return {
            "job_item_id": job_item_id,
            "source_id": source_id,
        }

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def build_prompts_for_existing_job_item(
    local_db_path: str,
    job_item_id: str,
    qa_chunk_conf: ChunkConfig,
    plain_chunk_conf: ChunkConfig,
    qa_template_text: str,
    plain_template_text: str,
) -> dict[str, Any]:
    conn = open_user_db(local_db_path)
    try:
        meta = fetch_opendata_job_item_meta(conn, job_item_id)

        qa_prompt_texts = build_opendata_prompt_texts(
            conn=conn,
            job_item_id=job_item_id,
            template_text=qa_template_text,
            chunk_conf=qa_chunk_conf,
        )

        plain_prompt_texts = build_opendata_prompt_texts(
            conn=conn,
            job_item_id=job_item_id,
            template_text=plain_template_text,
            chunk_conf=plain_chunk_conf,
        )

        return {
            "job_item_id": job_item_id,
            "source_id": meta["parent_source_id"],
            "parent_source_id": meta["parent_source_id"],
            "parent_label": meta["parent_label"],
            "qa_prompt_texts": qa_prompt_texts,
            "plain_prompt_texts": plain_prompt_texts,
            "qa_chunk_total": len(qa_prompt_texts),
            "plain_chunk_total": len(plain_prompt_texts),
        }
    finally:
        conn.close()


def mark_job_item_running(local_db_path: str, job_item_id: str) -> None:
    conn = open_user_db(local_db_path)
    try:
        conn.execute("BEGIN")
        conn.execute(
            """
            UPDATE knowledge_job_items
            SET status = 'running',
                started_at = COALESCE(started_at, ?),
                error_message = NULL
            WHERE job_item_id = ?
            """,
            (now_iso(), job_item_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def finalize_job_item_success(
    local_db_path: str,
    job_id: str,
    job_item_id: str,
    source_id: str,
    qa_llm_result: dict,
    plain_llm_result: dict,
    preview_only: bool,
) -> tuple[int, int]:
    conn = open_user_db(local_db_path)
    try:
        conn.execute("BEGIN")

        qa_count = 0
        plain_count = 0

        if not preview_only:
            qa_count = insert_qa_items_from_llm_result(
                conn=conn,
                job_id=job_id,
                job_item_id=job_item_id,
                source_id=source_id,
                llm_result=qa_llm_result,
            )

            plain_count = insert_plain_items_from_llm_result(
                conn=conn,
                job_id=job_id,
                job_item_id=job_item_id,
                source_id=source_id,
                llm_result=plain_llm_result,
            )

        finished_at = now_iso()
        item_status = "done"

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
                item_status,
                qa_count + plain_count,
                finished_at,
                job_item_id,
            ),
        )

        conn.commit()
        return qa_count, plain_count

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def finalize_job_item_error(
    local_db_path: str,
    job_item_id: str,
    error_message: str,
) -> None:
    conn = open_user_db(local_db_path)
    try:
        conn.execute("BEGIN")
        conn.execute(
            """
            UPDATE knowledge_job_items
            SET status = 'error',
                finished_at = ?,
                error_message = ?
            WHERE job_item_id = ?
            """,
            (
                now_iso(),
                error_message,
                job_item_id,
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_job_summary(
    local_db_path: str,
    job_id: str,
    requested_at: str,
    status: str,
    total_qa_count: int,
    total_plain_count: int,
    total_error_count: int,
    error_message: str | None = None,
) -> None:
    conn = open_user_db(local_db_path)
    try:
        conn.execute("BEGIN")
        conn.execute(
            """
            UPDATE knowledge_jobs
            SET status = ?,
                qa_count = ?,
                plain_count = ?,
                error_count = ?,
                error_message = ?,
                started_at = CASE
                    WHEN ? = 'running' THEN COALESCE(started_at, ?)
                    ELSE started_at
                END,
                finished_at = CASE
                    WHEN ? IN ('done', 'error') THEN ?
                    WHEN ? = 'running' THEN NULL
                    ELSE finished_at
                END
            WHERE job_id = ?
            """
            ,
            (
                status,
                total_qa_count,
                total_plain_count,
                total_error_count,
                error_message,
                status,
                requested_at,
                status,
                now_iso(),
                status,
                job_id,
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def process_opendata_job_item(
    local_db_path: str,
    job_id: str,
    job_item_id: str,
    qa_chunk_conf: ChunkConfig,
    plain_chunk_conf: ChunkConfig,
    qa_template_text: str,
    plain_template_text: str,
    preview_only: bool,
    progress_callback=None,
) -> dict[str, Any]:
    prepared = build_prompts_for_existing_job_item(
        local_db_path=local_db_path,
        job_item_id=job_item_id,
        qa_chunk_conf=qa_chunk_conf,
        plain_chunk_conf=plain_chunk_conf,
        qa_template_text=qa_template_text,
        plain_template_text=plain_template_text,
    )

    source_id = prepared["source_id"]
    qa_prompt_texts = prepared["qa_prompt_texts"]
    plain_prompt_texts = prepared["plain_prompt_texts"]
    total_qa_chunks = len(qa_prompt_texts)
    total_plain_chunks = len(plain_prompt_texts)

    if progress_callback:
        progress_callback(
            total_qa_chunks=total_qa_chunks,
            processed_qa_chunks=0,
            total_plain_chunks=total_plain_chunks,
            processed_plain_chunks=0,
            message="chunk prepared",
        )

    if preview_only:
        qa_llm_result = {"job_item_id": job_item_id, "qa_list": []}
        plain_llm_result = {"job_item_id": job_item_id, "plain_list": []}
        qa_llm_result_for_debug = None
        plain_llm_result_for_debug = None
    else:
        logger.info(
            "llm start: job_item_id=%s qa_chunks=%s plain_chunks=%s",
            job_item_id,
            total_qa_chunks,
            total_plain_chunks,
        )

        qa_chunk_results = []
        for idx, prompt_text in enumerate(qa_prompt_texts, start=1):
            result_list = run_chunked_llm_json([prompt_text], "LLM OPENDATA QA")
            if not result_list:
                raise Exception("empty qa chunk result")
            qa_chunk_results.append(result_list[0])
            if progress_callback:
                progress_callback(
                    total_qa_chunks=total_qa_chunks,
                    processed_qa_chunks=idx,
                    total_plain_chunks=total_plain_chunks,
                    processed_plain_chunks=0,
                    message=f"qa chunk {idx}/{total_qa_chunks}",
                )

        qa_llm_result = merge_qa_chunk_results(job_item_id, qa_chunk_results)
        qa_llm_result_for_debug = {
            "chunk_results": qa_chunk_results,
            "merged_result": qa_llm_result,
        }

        plain_chunk_results = []
        for idx, prompt_text in enumerate(plain_prompt_texts, start=1):
            result_list = run_chunked_llm_json([prompt_text], "LLM OPENDATA PLAIN")
            if not result_list:
                raise Exception("empty plain chunk result")
            plain_chunk_results.append(result_list[0])
            if progress_callback:
                progress_callback(
                    total_qa_chunks=total_qa_chunks,
                    processed_qa_chunks=total_qa_chunks,
                    total_plain_chunks=total_plain_chunks,
                    processed_plain_chunks=idx,
                    message=f"plain chunk {idx}/{total_plain_chunks}",
                )

        plain_llm_result = merge_plain_chunk_results(job_item_id, plain_chunk_results)
        plain_llm_result_for_debug = {
            "chunk_results": plain_chunk_results,
            "merged_result": plain_llm_result,
        }

    qa_count, plain_count = finalize_job_item_success(
        local_db_path=local_db_path,
        job_id=job_id,
        job_item_id=job_item_id,
        source_id=source_id,
        qa_llm_result=qa_llm_result,
        plain_llm_result=plain_llm_result,
        preview_only=preview_only,
    )

    return {
        "job_item_id": job_item_id,
        "parent_source_id": prepared["parent_source_id"],
        "parent_label": prepared["parent_label"],
        "qa_count": qa_count,
        "plain_count": plain_count,
        "status": "done",
        "qa_prompt_texts": qa_prompt_texts,
        "plain_prompt_texts": plain_prompt_texts,
        "qa_debug": qa_llm_result_for_debug,
        "plain_debug": plain_llm_result_for_debug,
    }


def run_opendata_job_background(uid: str, job_id: str) -> None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        logger.error("ank.db not found in background: %s", db_gcs_path)
        set_job_error(
            bucket,
            uid,
            job_id,
            f"ank.db not found: {db_gcs_path}",
            phase="extract_knowledge",
            message="ank.db not found",
        )
        return

    local_db_path = local_user_db_path(uid)

    try:
        replace_local_db_from_blob(db_blob, local_db_path)

        job_row = fetch_job_row(local_db_path, job_id)
        if not job_row:
            set_job_error(
                bucket,
                uid,
                job_id,
                f"knowledge_jobs not found: {job_id}",
                phase="extract_knowledge",
                message="knowledge_jobs not found",
            )
            return

        if job_row["source_type"] != SOURCE_TYPE:
            set_job_error(
                bucket,
                uid,
                job_id,
                "invalid source_type",
                phase="extract_knowledge",
                message="invalid source_type",
            )
            return

        qa_template_text = load_template_text(BUCKET_NAME, OPENDATA_QA_PROMPT_PATH)
        plain_template_text = load_template_text(BUCKET_NAME, OPENDATA_PLAIN_PROMPT_PATH)
        chunk_config = load_chunk_config(BUCKET_NAME, OPENAI_CHUNK_CONFIG_PATH)
        qa_chunk_conf = get_required_opendata_chunk_conf(chunk_config, "qa")
        plain_chunk_conf = get_required_opendata_chunk_conf(chunk_config, "plain")

        requested_at = job_row["requested_at"] or now_iso()
        total_qa_count = int(job_row["qa_count"] or 0)
        total_plain_count = int(job_row["plain_count"] or 0)
        total_error_count = int(job_row["error_count"] or 0)

        job_items = []
        conn = open_user_db(local_db_path)
        try:
            cur = conn.execute(
                """
                SELECT ji.parent_source_id, ji.parent_label, ji.row_count
                FROM knowledge_job_items ji
                WHERE ji.job_id = ?
                ORDER BY ji.created_at, ji.job_item_id
                LIMIT 1
                """,
                (job_id,),
            )
            first_item = cur.fetchone()
        finally:
            conn.close()

        dataset_id = first_item["parent_source_id"] if first_item else None
        dataset_name = first_item["parent_label"] if first_item else None
        row_count = int(first_item["row_count"] or 0) if first_item else 0

        update_job_summary(
            local_db_path=local_db_path,
            job_id=job_id,
            requested_at=requested_at,
            status="running",
            total_qa_count=total_qa_count,
            total_plain_count=total_plain_count,
            total_error_count=total_error_count,
            error_message=None,
        )
        upload_local_db(db_blob, local_db_path)

        set_job_running(
            bucket,
            uid,
            job_id,
            SOURCE_TYPE,
            phase="extract_knowledge",
            message="background started",
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            row_count=row_count,
            qa_total=0,
            plain_total=0,
            chunk_total=0,
        )

        while True:
            row = fetch_next_new_job_item(local_db_path, job_id)
            if not row:
                break

            current_job_item_id = row["job_item_id"]
            current_parent_label = row["parent_label"]
            current_parent_source_id = row["parent_source_id"]
            current_row_count = int(row["row_count"] or 0)
            mark_job_item_running(local_db_path, current_job_item_id)

            def progress_callback(*, total_qa_chunks=0, processed_qa_chunks=0, total_plain_chunks=0, processed_plain_chunks=0, message=None):
                set_job_progress(
                    bucket,
                    uid,
                    phase="extract_knowledge",
                    message=message,
                    knowledge_count=total_qa_count + total_plain_count,
                    qa_current=int(processed_qa_chunks or 0),
                    qa_total=int(total_qa_chunks or 0),
                    plain_current=int(processed_plain_chunks or 0),
                    plain_total=int(total_plain_chunks or 0),
                    chunk_current=int(processed_qa_chunks or 0) + int(processed_plain_chunks or 0),
                    chunk_total=int(total_qa_chunks or 0) + int(total_plain_chunks or 0),
                )

            try:
                progress_callback(message="item running")

                result = process_opendata_job_item(
                    local_db_path=local_db_path,
                    job_id=job_id,
                    job_item_id=current_job_item_id,
                    qa_chunk_conf=qa_chunk_conf,
                    plain_chunk_conf=plain_chunk_conf,
                    qa_template_text=qa_template_text,
                    plain_template_text=plain_template_text,
                    preview_only=False,
                    progress_callback=progress_callback,
                )

                total_qa_count += int(result["qa_count"] or 0)
                total_plain_count += int(result["plain_count"] or 0)

                update_job_summary(
                    local_db_path=local_db_path,
                    job_id=job_id,
                    requested_at=requested_at,
                    status="running",
                    total_qa_count=total_qa_count,
                    total_plain_count=total_plain_count,
                    total_error_count=total_error_count,
                    error_message=None,
                )

                set_job_progress(
                    bucket,
                    uid,
                    phase="extract_knowledge",
                    message="item done",
                    knowledge_count=total_qa_count + total_plain_count,
                )

            except Exception as e:
                logger.exception(
                    "run_opendata_job_background item failed: job_id=%s job_item_id=%s",
                    job_id,
                    current_job_item_id,
                )

                try:
                    finalize_job_item_error(
                        local_db_path=local_db_path,
                        job_item_id=current_job_item_id,
                        error_message=str(e),
                    )
                except Exception:
                    logger.exception("failed to update error state: job_item_id=%s", current_job_item_id)

                total_error_count += 1

                try:
                    update_job_summary(
                        local_db_path=local_db_path,
                        job_id=job_id,
                        requested_at=requested_at,
                        status="error",
                        total_qa_count=total_qa_count,
                        total_plain_count=total_plain_count,
                        total_error_count=total_error_count,
                        error_message=str(e),
                    )
                except Exception:
                    logger.exception("failed to update job summary error: job_id=%s", job_id)

                set_job_error(
                    bucket,
                    uid,
                    job_id,
                    str(e),
                    phase="extract_knowledge",
                    message="item failed",
                )
                return

        update_job_summary(
            local_db_path=local_db_path,
            job_id=job_id,
            requested_at=requested_at,
            status="done",
            total_qa_count=total_qa_count,
            total_plain_count=total_plain_count,
            total_error_count=total_error_count,
            error_message=None,
        )
        upload_local_db(db_blob, local_db_path)
        set_job_done(
            bucket,
            uid,
            job_id,
            phase="extract_knowledge",
            message="completed",
            knowledge_count=total_qa_count + total_plain_count,
        )

    except Exception as e:
        logger.exception("run_opendata_job_background failed: job_id=%s", job_id)
        set_job_error(
            bucket,
            uid,
            job_id,
            str(e),
            phase="extract_knowledge",
            message="background failed",
        )


class KnowledgeTargetItem(BaseModel):
    source_type: Optional[str] = Field(default=SOURCE_TYPE, description="opendata")
    parent_source_id: Optional[str] = None
    parent_key1: Optional[str] = None
    parent_key2: Optional[str] = None
    parent_label: Optional[str] = None
    row_count: int = 0


class KnowledgeJobCreateRequest(BaseModel):
    source_type: Optional[str] = Field(default=SOURCE_TYPE, description="opendata")
    source_name: Optional[str] = None
    request_type: str = "extract_knowledge"
    items: List[KnowledgeTargetItem]
    preview_only: bool = False


class PromptPreviewItem(BaseModel):
    job_item_id: str
    parent_source_id: Optional[str] = None
    parent_label: Optional[str] = None
    prompt_type: str
    prompt_text: str


class KnowledgeDebugItem(BaseModel):
    job_item_id: str
    parent_label: Optional[str] = None
    status: str
    qa_count: int = 0
    plain_count: int = 0
    error_message: Optional[str] = None
    llm_result: Optional[Any] = None


class KnowledgeJobCreateResponse(BaseModel):
    job_id: str
    selected_count: int
    created_item_count: int
    status: str
    prompt_previews: List[PromptPreviewItem] = []
    debug_items: List[KnowledgeDebugItem] = []


class KnowledgeRunRequest(BaseModel):
    job_id: str


class WorkerRunRequest(BaseModel):
    uid: str
    job_id: str
    source_type: Optional[str] = None


def validate_request(body: KnowledgeJobCreateRequest) -> None:
    if body.source_type not in (None, "", SOURCE_TYPE):
        raise HTTPException(status_code=400, detail=f"source_type must be '{SOURCE_TYPE}'")
    if not body.items:
        raise HTTPException(status_code=400, detail="items is empty")

    for item in body.items:
        if item.source_type not in (None, "", SOURCE_TYPE):
            raise HTTPException(status_code=400, detail=f"item.source_type must be '{SOURCE_TYPE}'")


@router.post("/job", response_model=KnowledgeJobCreateResponse)
def create_opendata_job(
    body: KnowledgeJobCreateRequest,
    authorization: str | None = Header(default=None),
):
    validate_request(body)

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

    local_db_path = local_user_db_path(uid)
    replace_local_db_from_blob(db_blob, local_db_path)

    try:
        qa_template_text = load_template_text(BUCKET_NAME, OPENDATA_QA_PROMPT_PATH)
        plain_template_text = load_template_text(BUCKET_NAME, OPENDATA_PLAIN_PROMPT_PATH)
        chunk_config = load_chunk_config(BUCKET_NAME, OPENAI_CHUNK_CONFIG_PATH)
        qa_chunk_conf = get_required_opendata_chunk_conf(chunk_config, "qa")
        plain_chunk_conf = get_required_opendata_chunk_conf(chunk_config, "plain")

        unique_items: List[KnowledgeTargetItem] = []
        seen_keys = set()

        for item in body.items:
            key = (
                SOURCE_TYPE,
                item.parent_source_id or "",
                item.parent_key1 or "",
                item.parent_key2 or "",
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_items.append(item)

        selected_count = len(unique_items)
        job_id, requested_at = create_job_record(local_db_path, body, selected_count)

        prompt_previews: List[PromptPreviewItem] = []
        debug_items: List[KnowledgeDebugItem] = []
        created_item_count = 0

        for item in unique_items:
            prepared = prepare_job_item(
                local_db_path=local_db_path,
                bucket=bucket,
                job_id=job_id,
                item=item,
                requested_at=requested_at,
                preview_only=body.preview_only,
            )

            prompts = build_prompts_for_existing_job_item(
                local_db_path=local_db_path,
                job_item_id=prepared["job_item_id"],
                qa_chunk_conf=qa_chunk_conf,
                plain_chunk_conf=plain_chunk_conf,
                qa_template_text=qa_template_text,
                plain_template_text=plain_template_text,
            )

            prompt_previews.append(
                PromptPreviewItem(
                    job_item_id=prepared["job_item_id"],
                    parent_source_id=item.parent_source_id,
                    parent_label=item.parent_label,
                    prompt_type="qa",
                    prompt_text=join_prompt_previews(prompts["qa_prompt_texts"]),
                )
            )
            prompt_previews.append(
                PromptPreviewItem(
                    job_item_id=prepared["job_item_id"],
                    parent_source_id=item.parent_source_id,
                    parent_label=item.parent_label,
                    prompt_type="plain",
                    prompt_text=join_prompt_previews(prompts["plain_prompt_texts"]),
                )
            )

            debug_items.append(
                KnowledgeDebugItem(
                    job_item_id=prepared["job_item_id"],
                    parent_label=item.parent_label,
                    status="done" if body.preview_only else "new",
                    qa_count=0,
                    plain_count=0,
                    error_message=None,
                    llm_result=None,
                )
            )
            created_item_count += 1

        initial_status = "done" if body.preview_only else "new"
        update_job_summary(
            local_db_path=local_db_path,
            job_id=job_id,
            requested_at=requested_at,
            status=initial_status,
            total_qa_count=0,
            total_plain_count=0,
            total_error_count=0,
            error_message=None,
        )

        upload_local_db(db_blob, local_db_path)

        return KnowledgeJobCreateResponse(
            job_id=job_id,
            selected_count=selected_count,
            created_item_count=created_item_count,
            status=initial_status,
            prompt_previews=prompt_previews,
            debug_items=debug_items,
        )

    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail=f"duplicate or integrity error: {e}")

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("create_opendata_job failed")
        raise HTTPException(status_code=500, detail=f"create_opendata_job failed: {type(e).__name__}: {e}")


@router.post("/run", response_model=KnowledgeJobCreateResponse)
def run_opendata_job(
    body: KnowledgeRunRequest,
    authorization: str | None = Header(default=None),
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

    current_status = try_read_status_payload(bucket, uid)
    current_status_name = (current_status.get("status") or "").strip().lower()
    if current_status_name in ("queued", "running"):
        running_job_id = current_status.get("job_id")
        if running_job_id == body.job_id:
            return KnowledgeJobCreateResponse(
                job_id=body.job_id,
                selected_count=0,
                created_item_count=0,
                status=current_status_name or "queued",
                prompt_previews=[],
                debug_items=[],
            )
        raise HTTPException(
            status_code=409,
            detail=f"別のジョブが実行中です。完了後に再実行してください。 running_job_id={running_job_id or ''}",
        )

    local_db_path = local_user_db_path(uid)
    replace_local_db_from_blob(db_blob, local_db_path)

    try:
        job_row = fetch_job_row(local_db_path, body.job_id)
        if not job_row:
            raise HTTPException(status_code=404, detail=f"knowledge_jobs not found: {body.job_id}")
        if job_row["source_type"] != SOURCE_TYPE:
            raise HTTPException(status_code=400, detail=f"job source_type must be '{SOURCE_TYPE}'")

        if job_row["status"] == "done":
            set_job_done(
                bucket,
                uid,
                body.job_id,
                phase="extract_knowledge",
                message="already done",
                knowledge_count=int(job_row["qa_count"] or 0) + int(job_row["plain_count"] or 0),
            )
            return KnowledgeJobCreateResponse(
                job_id=body.job_id,
                selected_count=0,
                created_item_count=0,
                status="done",
                prompt_previews=[],
                debug_items=[],
            )

        requested_at = job_row["requested_at"] or now_iso()
        total_qa_count = int(job_row["qa_count"] or 0)
        total_plain_count = int(job_row["plain_count"] or 0)
        total_error_count = int(job_row["error_count"] or 0)

        conn = open_user_db(local_db_path)
        try:
            cur = conn.execute(
                """
                SELECT ji.parent_source_id, ji.parent_label, ji.row_count
                FROM knowledge_job_items ji
                WHERE ji.job_id = ?
                ORDER BY ji.created_at, ji.job_item_id
                LIMIT 1
                """,
                (body.job_id,),
            )
            first_item = cur.fetchone()
        finally:
            conn.close()

        dataset_id = first_item["parent_source_id"] if first_item else None
        dataset_name = first_item["parent_label"] if first_item else None
        row_count = int(first_item["row_count"] or 0) if first_item else 0

        update_job_summary(
            local_db_path=local_db_path,
            job_id=body.job_id,
            requested_at=requested_at,
            status="queued",
            total_qa_count=total_qa_count,
            total_plain_count=total_plain_count,
            total_error_count=total_error_count,
            error_message=None,
        )
        upload_local_db(db_blob, local_db_path)

        set_job_queued(
            bucket,
            uid,
            body.job_id,
            SOURCE_TYPE,
            phase="extract_knowledge",
            message="job queued",
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            row_count=row_count,
            qa_total=0,
            plain_total=0,
            chunk_total=0,
        )

        try:
            enqueue_knowledge_job(
                source_type=SOURCE_TYPE,
                uid=uid,
                job_id=body.job_id,
            )
        except Exception as e:
            logger.exception("enqueue opendata job failed: job_id=%s", body.job_id)
            update_job_summary(
                local_db_path=local_db_path,
                job_id=body.job_id,
                requested_at=requested_at,
                status="error",
                total_qa_count=total_qa_count,
                total_plain_count=total_plain_count,
                total_error_count=total_error_count + 1,
                error_message=str(e),
            )
            upload_local_db(db_blob, local_db_path)
            set_job_error(
                bucket,
                uid,
                body.job_id,
                str(e),
                phase="extract_knowledge",
                message="enqueue failed",
            )
            raise HTTPException(status_code=500, detail=f"enqueue failed: {type(e).__name__}: {e}")

        return KnowledgeJobCreateResponse(
            job_id=body.job_id,
            selected_count=0,
            created_item_count=0,
            status="queued",
            prompt_previews=[],
            debug_items=[],
        )

    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail=f"duplicate or integrity error: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("run_opendata_job failed")
        raise HTTPException(status_code=500, detail=f"run_opendata_job failed: {type(e).__name__}: {e}")


@router.post("/_worker/run")
def run_opendata_job_worker(
    body: WorkerRunRequest,
    x_worker_token: str | None = Header(default=None),
):
    verify_worker_token(x_worker_token)

    if body.source_type not in (None, "", SOURCE_TYPE):
        raise HTTPException(status_code=400, detail=f"source_type must be '{SOURCE_TYPE}'")

    if not (body.uid or "").strip():
        raise HTTPException(status_code=400, detail="uid is required")
    if not (body.job_id or "").strip():
        raise HTTPException(status_code=400, detail="job_id is required")

    run_opendata_job_background(body.uid, body.job_id)
    return {"ok": True, "job_id": body.job_id, "status": "done"}


@router.get("/status", response_model=KnowledgeJobStatusResponse)
def get_opendata_job_status(
    job_id: str = Query(...),
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    try:
        payload = try_read_status_payload(bucket, uid)
        if payload.get("job_id") == job_id:
            return KnowledgeJobStatusResponse(**payload)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("failed to read knowledge_generate.json: %s", e)

    raise HTTPException(status_code=404, detail=f"status not found: {job_id}")
