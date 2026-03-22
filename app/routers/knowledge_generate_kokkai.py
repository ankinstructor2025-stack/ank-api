from __future__ import annotations

import os
import sqlite3
import logging
from typing import List, Optional, Any

from fastapi import APIRouter, Header, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from google.cloud import storage


from .openai_llm_client import run_chunked_llm_json
from .openai_chunking import ChunkConfig, build_chunks
from .openai_prompt_builder import build_kokkai_prompt_text
from .knowledge_generate_common import (
    now_iso,
    new_id,
    user_db_path,
    local_user_db_path,
    build_status_payload_from_db,
    build_source_status_payload_from_db,
    get_uid_from_auth_header,
    normalize_text,
    extract_row_text,
    load_template_text,
    get_generation_status_source,
    update_generation_status_source,
    load_chunk_config,
    open_user_db,
    upload_local_db,
    build_lock_key,
    try_acquire_job_lock,
    release_job_lock,
    get_running_lock_job_id,
    fetch_job_row,
    fetch_next_new_job_item,
    replace_local_db_from_blob,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge/kokkai", tags=["knowledge_kokkai"])

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
SOURCE_TYPE = "kokkai"

KOKKAI_QA_PROMPT_PATH = "template/kokkai_qa_prompt.txt"
KOKKAI_PLAIN_PROMPT_PATH = "template/kokkai_plain_prompt.txt"
OPENAI_CHUNK_CONFIG_PATH = "template/openai_chunk.json"

def get_kokkai_chunk_conf(chunk_config: dict, prompt_type: str) -> ChunkConfig:
    kokkai_conf = chunk_config.get("kokkai")
    if not isinstance(kokkai_conf, dict):
        raise HTTPException(status_code=500, detail="openai_chunk.json: kokkai section not found")

    conf = kokkai_conf.get(prompt_type)
    if not isinstance(conf, dict):
        raise HTTPException(status_code=500, detail=f"openai_chunk.json: kokkai.{prompt_type} section not found")

    missing = [key for key in ("max_chars", "max_items", "overlap_items") if key not in conf]
    if missing:
        raise HTTPException(status_code=500, detail=f"openai_chunk.json: kokkai.{prompt_type} missing keys: {', '.join(missing)}")

    try:
        max_chars = int(conf["max_chars"])
        max_items = int(conf["max_items"])
        overlap_items = int(conf["overlap_items"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"openai_chunk.json: invalid kokkai.{prompt_type} values: {e}")

    if max_chars <= 0:
        raise HTTPException(status_code=500, detail=f"openai_chunk.json: kokkai.{prompt_type}.max_chars must be > 0")
    if max_items <= 0:
        raise HTTPException(status_code=500, detail=f"openai_chunk.json: kokkai.{prompt_type}.max_items must be > 0")
    if overlap_items < 0:
        raise HTTPException(status_code=500, detail=f"openai_chunk.json: kokkai.{prompt_type}.overlap_items must be >= 0")

    return ChunkConfig(max_chars=max_chars, max_items=max_items, overlap_items=overlap_items)


def fetch_kokkai_job_item_meta(conn: sqlite3.Connection, job_item_id: str) -> sqlite3.Row:
    cur = conn.execute(
        """
        SELECT
            ji.job_item_id,
            ji.parent_source_id,
            ji.parent_label,
            d.name_of_house,
            d.name_of_meeting,
            d.logical_name
        FROM knowledge_job_items ji
        LEFT JOIN kokkai_documents d
          ON d.issue_id = ji.parent_source_id
        WHERE ji.job_item_id = ?
        LIMIT 1
        """,
        (job_item_id,),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"knowledge_job_items not found: {job_item_id}")
    return row


def fetch_kokkai_content_rows(conn: sqlite3.Connection, job_item_id: str) -> list[sqlite3.Row]:
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


def build_kokkai_prompt_texts(
    conn: sqlite3.Connection,
    job_item_id: str,
    template_path: str,
    default_template: str,
    chunk_conf: ChunkConfig,
) -> list[str]:
    item = fetch_kokkai_job_item_meta(conn, job_item_id)
    rows = fetch_kokkai_content_rows(conn, job_item_id)

    chunks = build_chunks(
        rows,
        chunk_conf,
        allowed_content_types={"row"},
    )

    if not chunks:
        raise HTTPException(status_code=400, detail=f"knowledge_contents not found: {job_item_id}")

    prompt_template = load_template_text(template_path, default_template)

    prompt_texts: list[str] = []
    for chunk in chunks:
        prompt_texts.append(
            build_kokkai_prompt_text(
                job_item_id=job_item_id,
                prompt_template=prompt_template,
                chunk=chunk,
                name_of_house=item["name_of_house"],
                name_of_meeting=item["name_of_meeting"],
                logical_name=item["logical_name"],
                parent_label=item["parent_label"],
            )
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


def insert_kokkai_contents(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
    issue_id: str,
    source_rows: list[sqlite3.Row],
) -> int:
    inserted_count = 0
    now = now_iso()

    for idx, row in enumerate(source_rows, start=1):
        speech_text = extract_row_text(row["speech"])
        if not speech_text:
            continue

        source_item_id = row["speech_id"]
        row_id = f"{row['issue_id']}:{row['speech_id']}"

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
                issue_id,
                source_item_id,
                row_id,
                "row",
                speech_text,
                idx,
                now,
                now,
            ),
        )
        inserted_count += 1

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
        conn.execute("BEGIN IMMEDIATE")
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
                body.source_name or "国会議事録",
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
    job_id: str,
    item: "KnowledgeTargetItem",
    requested_at: str,
    preview_only: bool,
) -> dict[str, Any]:
    issue_id = item.parent_source_id or ""
    if not issue_id:
        raise HTTPException(status_code=400, detail="parent_source_id is required")

    source_rows = fetch_kokkai_source_rows(local_db_path, issue_id)
    if not source_rows:
        raise HTTPException(status_code=400, detail=f"kokkai_document_rows not found: {issue_id}")

    job_item_id = new_id()

    conn = open_user_db(local_db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")

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

        inserted_rows = insert_kokkai_contents(
            conn=conn,
            job_id=job_id,
            job_item_id=job_item_id,
            issue_id=issue_id,
            source_rows=source_rows,
        )

        if inserted_rows <= 0:
            raise HTTPException(status_code=400, detail=f"speech not found: {issue_id}")

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
            "source_id": issue_id,
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
) -> dict[str, Any]:
    conn = open_user_db(local_db_path)
    try:
        meta = fetch_kokkai_job_item_meta(conn, job_item_id)

        qa_prompt_texts = build_kokkai_prompt_texts(
            conn=conn,
            job_item_id=job_item_id,
            template_path=KOKKAI_QA_PROMPT_PATH,
            default_template=DEFAULT_KOKKAI_QA_PROMPT,
            chunk_conf=qa_chunk_conf,
        )

        plain_prompt_texts = build_kokkai_prompt_texts(
            conn=conn,
            job_item_id=job_item_id,
            template_path=KOKKAI_PLAIN_PROMPT_PATH,
            default_template=DEFAULT_KOKKAI_PLAIN_PROMPT,
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
        conn.execute("BEGIN IMMEDIATE")
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
        conn.execute("BEGIN IMMEDIATE")

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
        conn.execute("BEGIN IMMEDIATE")
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
        conn.execute("BEGIN IMMEDIATE")
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
            """,
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


def build_source_status_payload(
    local_db_path: str,
    job_id: str,
    *,
    phase: str | None = None,
    current_item_id: str | None = None,
    current_label: str | None = None,
    message: str | None = None,
    error_message: str | None = None,
    total_qa_chunks: int = 0,
    processed_qa_chunks: int = 0,
    total_plain_chunks: int = 0,
    processed_plain_chunks: int = 0,
) -> dict[str, Any]:
    payload = build_status_payload_from_db(local_db_path, job_id)
    items = payload.get("items") or []
    done_count = sum(1 for item in items if item.get("status") == "done")
    error_count = sum(1 for item in items if item.get("status") == "error")
    waiting_count = sum(1 for item in items if item.get("status") in ("new", "running"))
    payload.update({
        "phase": phase,
        "done_count": done_count,
        "error_count": error_count,
        "waiting_count": waiting_count,
        "current_item_id": current_item_id,
        "current_label": current_label,
        "message": message,
        "error_message": error_message if error_message is not None else payload.get("error_message"),
        "total_qa_chunks": int(total_qa_chunks or 0),
        "processed_qa_chunks": int(processed_qa_chunks or 0),
        "total_plain_chunks": int(total_plain_chunks or 0),
        "processed_plain_chunks": int(processed_plain_chunks or 0),
        "total_chunks": int(total_qa_chunks or 0) + int(total_plain_chunks or 0),
        "processed_chunks": int(processed_qa_chunks or 0) + int(processed_plain_chunks or 0),
    })
    return payload


def process_kokkai_job_item(
    local_db_path: str,
    job_id: str,
    job_item_id: str,
    qa_chunk_conf: ChunkConfig,
    plain_chunk_conf: ChunkConfig,
    preview_only: bool,
    progress_callback=None,
) -> dict[str, Any]:
    prepared = build_prompts_for_existing_job_item(
        local_db_path=local_db_path,
        job_item_id=job_item_id,
        qa_chunk_conf=qa_chunk_conf,
        plain_chunk_conf=plain_chunk_conf,
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
            result_list = run_chunked_llm_json([prompt_text], "LLM KOKKAI QA")
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
            result_list = run_chunked_llm_json([prompt_text], "LLM KOKKAI PLAIN")
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


def run_kokkai_job_background(uid: str, job_id: str) -> None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        logger.error("ank.db not found in background: %s", db_gcs_path)
        update_generation_status_source(BUCKET_NAME, uid, SOURCE_TYPE, {
            "job_id": job_id,
            "status": "error",
            "phase": "extract_knowledge",
            "message": "ank.db not found",
            "error_message": f"ank.db not found: {db_gcs_path}",
            "finished_at": now_iso(),
        })
        return

    local_db_path = local_user_db_path(uid)

    try:
        replace_local_db_from_blob(db_blob, local_db_path)

        job_row = fetch_job_row(local_db_path, job_id)
        if not job_row:
            update_generation_status_source(BUCKET_NAME, uid, SOURCE_TYPE, {
                "job_id": job_id,
                "status": "error",
                "phase": "extract_knowledge",
                "message": "knowledge_jobs not found",
                "error_message": f"knowledge_jobs not found: {job_id}",
                "finished_at": now_iso(),
            })
            return

        if job_row["source_type"] != SOURCE_TYPE:
            update_generation_status_source(BUCKET_NAME, uid, SOURCE_TYPE, {
                "job_id": job_id,
                "status": "error",
                "phase": "extract_knowledge",
                "message": "invalid source_type",
                "error_message": "invalid source_type",
                "finished_at": now_iso(),
            })
            return

        if job_row["status"] == "done":
            update_generation_status_source(BUCKET_NAME, uid, SOURCE_TYPE, build_source_status_payload(local_db_path, job_id, phase="extract_knowledge", message="already done"))
            return

        qa_template_text = load_template_text(BUCKET_NAME, KOKKAI_QA_PROMPT_PATH)
        plain_template_text = load_template_text(BUCKET_NAME, KOKKAI_PLAIN_PROMPT_PATH)
        chunk_config = load_chunk_config(BUCKET_NAME, OPENAI_CHUNK_CONFIG_PATH)
        qa_chunk_conf = get_kokkai_chunk_conf(chunk_config, "qa")
        plain_chunk_conf = get_kokkai_chunk_conf(chunk_config, "plain")

        requested_at = job_row["requested_at"] or now_iso()
        total_qa_count = int(job_row["qa_count"] or 0)
        total_plain_count = int(job_row["plain_count"] or 0)
        total_error_count = int(job_row["error_count"] or 0)

        update_generation_status_source(BUCKET_NAME, uid, SOURCE_TYPE, build_source_status_payload(local_db_path, job_id, phase="extract_knowledge", message="background started"))

        while True:
            row = fetch_next_new_job_item(local_db_path, job_id)
            if not row:
                break

            current_job_item_id = row["job_item_id"]
            current_parent_label = row["parent_label"]
            mark_job_item_running(local_db_path, current_job_item_id)

            def progress_callback(*, total_qa_chunks=0, processed_qa_chunks=0, total_plain_chunks=0, processed_plain_chunks=0, message=None):
                update_generation_status_source(
                    BUCKET_NAME,
                    uid,
                    SOURCE_TYPE,
                    build_source_status_payload(
                        local_db_path,
                        job_id,
                        phase="extract_knowledge",
                        current_item_id=current_job_item_id,
                        current_label=current_parent_label,
                        message=message,
                        total_qa_chunks=total_qa_chunks,
                        processed_qa_chunks=processed_qa_chunks,
                        total_plain_chunks=total_plain_chunks,
                        processed_plain_chunks=processed_plain_chunks,
                    ),
                )

            try:
                progress_callback(message="item running")
                result = process_kokkai_job_item(
                    local_db_path=local_db_path,
                    job_id=job_id,
                    job_item_id=current_job_item_id,
                    qa_chunk_conf=qa_chunk_conf,
                    plain_chunk_conf=plain_chunk_conf,
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
                update_generation_status_source(BUCKET_NAME, uid, SOURCE_TYPE, build_source_status_payload_from_db(local_db_path, job_id, phase="extract_knowledge", current_item_id=current_job_item_id, current_label=result.get("parent_label"), message="item done", chunk_total=len(result.get("qa_prompt_texts") or []) + len(result.get("plain_prompt_texts") or []), chunk_done=len(result.get("qa_prompt_texts") or []) + len(result.get("plain_prompt_texts") or [])))
                progress_callback(message="item done")

            except Exception as e:
                logger.exception("run_kokkai_job_background item failed: job_id=%s job_item_id=%s", job_id, current_job_item_id)
                try:
                    finalize_job_item_error(local_db_path=local_db_path, job_item_id=current_job_item_id, error_message=str(e))
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

                payload = build_source_status_payload(local_db_path, job_id, phase="extract_knowledge", current_item_id=current_job_item_id, current_label=current_parent_label, message="item failed", error_message=str(e))
                payload["status"] = "error"
                payload["finished_at"] = now_iso()
                update_generation_status_source(BUCKET_NAME, uid, SOURCE_TYPE, payload)
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
        final_payload = build_source_status_payload(local_db_path, job_id, phase="extract_knowledge", message="completed")
        final_payload["status"] = "done"
        update_generation_status_source(BUCKET_NAME, uid, SOURCE_TYPE, final_payload)

    except Exception as e:
        logger.exception("run_kokkai_job_background failed: job_id=%s", job_id)
        payload = None
        try:
            payload = get_generation_status_source(BUCKET_NAME, uid, SOURCE_TYPE)
        except Exception:
            payload = {}
        payload = dict(payload or {})
        payload.update({
            "job_id": job_id,
            "status": "error",
            "phase": "extract_knowledge",
            "message": "background failed",
            "error_message": str(e),
            "finished_at": now_iso(),
        })
        update_generation_status_source(BUCKET_NAME, uid, SOURCE_TYPE, payload)


@router.post("/run", response_model=KnowledgeJobCreateResponse)
def run_kokkai_job(
    body: KnowledgeRunRequest,
    background_tasks: BackgroundTasks,
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

    current_source = get_generation_status_source(BUCKET_NAME, uid, SOURCE_TYPE)
    if current_source.get("status") == "running":
        running_job_id = current_source.get("job_id")
        if running_job_id == body.job_id:
            return KnowledgeJobCreateResponse(
                job_id=body.job_id,
                selected_count=int(current_source.get("selected_count") or 0),
                created_item_count=len(current_source.get("items") or []),
                status="running",
                prompt_previews=[],
                debug_items=[],
            )
        raise HTTPException(status_code=409, detail=f"別のジョブが実行中です。完了後に再実行してください。 running_job_id={running_job_id or ''}")

    local_db_path = local_user_db_path(uid)
    replace_local_db_from_blob(db_blob, local_db_path)

    try:
        job_row = fetch_job_row(local_db_path, body.job_id)
        if not job_row:
            raise HTTPException(status_code=404, detail=f"knowledge_jobs not found: {body.job_id}")
        if job_row["source_type"] != SOURCE_TYPE:
            raise HTTPException(status_code=400, detail=f"job source_type must be '{SOURCE_TYPE}'")

        if job_row["status"] == "done":
            payload = build_status_payload_from_db(local_db_path, body.job_id)
            done_payload = build_source_status_payload(local_db_path, body.job_id, phase="extract_knowledge", message="already done")
            done_payload["status"] = "done"
            update_generation_status_source(BUCKET_NAME, uid, SOURCE_TYPE, done_payload)
            return KnowledgeJobCreateResponse(
                job_id=body.job_id,
                selected_count=int(payload["selected_count"] or 0),
                created_item_count=len(payload["items"]),
                status="done",
                prompt_previews=[],
                debug_items=[],
            )

        requested_at = job_row["requested_at"] or now_iso()
        total_qa_count = int(job_row["qa_count"] or 0)
        total_plain_count = int(job_row["plain_count"] or 0)
        total_error_count = int(job_row["error_count"] or 0)

        update_job_summary(
            local_db_path=local_db_path,
            job_id=body.job_id,
            requested_at=requested_at,
            status="running",
            total_qa_count=total_qa_count,
            total_plain_count=total_plain_count,
            total_error_count=total_error_count,
            error_message=None,
        )

        running_payload = build_source_status_payload(local_db_path, body.job_id, phase="extract_knowledge", message="job started")
        running_payload["status"] = "running"
        update_generation_status_source(BUCKET_NAME, uid, SOURCE_TYPE, running_payload)

        payload = build_status_payload_from_db(local_db_path, body.job_id)
        background_tasks.add_task(run_kokkai_job_background, uid, body.job_id)

        return KnowledgeJobCreateResponse(
            job_id=body.job_id,
            selected_count=int(payload["selected_count"] or 0),
            created_item_count=len(payload["items"]),
            status="running",
            prompt_previews=[],
            debug_items=[],
        )

    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail=f"duplicate or integrity error: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("run_kokkai_job failed")
        raise HTTPException(status_code=500, detail=f"run_kokkai_job failed: {type(e).__name__}: {e}")


@router.get("/status", response_model=KnowledgeJobStatusResponse)
def get_kokkai_job_status(
    job_id: str = Query(...),
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)

    try:
        source_payload = get_generation_status_source(BUCKET_NAME, uid, SOURCE_TYPE)
        if source_payload.get("job_id") == job_id and source_payload.get("status") not in (None, "idle"):
            return KnowledgeJobStatusResponse(**source_payload)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("failed to read knowledge_generate.json: %s", e)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(status_code=400, detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}")

    local_db_path = local_user_db_path(uid)
    replace_local_db_from_blob(db_blob, local_db_path)
    payload = build_status_payload_from_db(local_db_path, job_id)
    return KnowledgeJobStatusResponse(**payload)
