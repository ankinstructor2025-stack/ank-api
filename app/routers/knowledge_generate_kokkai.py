from __future__ import annotations

import json
import os
import sqlite3
import uuid
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Optional, Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth

from llm_client import run_chunked_llm_json
from chunking import ChunkConfig, build_chunks
from prompt_builder import build_kokkai_prompt_text


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge/kokkai", tags=["knowledge_kokkai"])

JST = ZoneInfo("Asia/Tokyo")
BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
KOKKAI_QA_PROMPT_PATH = "template/kokkai_qa_prompt.txt"
KOKKAI_PLAIN_PROMPT_PATH = "template/kokkai_plain_prompt.txt"
OPENAI_CHUNK_CONFIG_PATH = "template/openai_chunk.json"
SOURCE_TYPE = "kokkai"


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


def extract_speaker(content_obj: dict) -> str:
    return normalize_text(content_obj.get("speaker"))


def load_template_text(path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)

    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"{path} not found")

    return blob.download_as_bytes().decode("utf-8")


def load_chunk_config() -> dict:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(OPENAI_CHUNK_CONFIG_PATH)

    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"{OPENAI_CHUNK_CONFIG_PATH} not found")

    try:
        obj = json.loads(blob.download_as_bytes().decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("chunk config root is not object")
        return obj
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to parse {OPENAI_CHUNK_CONFIG_PATH}: {e}")


def get_kokkai_chunk_conf(chunk_config: dict, prompt_type: str) -> ChunkConfig:
    conf = ((chunk_config.get("kokkai") or {}).get(prompt_type) or {})
    max_chars = int(conf.get("max_chars") or 12000)
    max_items = int(conf.get("max_items") or 80)
    overlap_items = int(conf.get("overlap_items") or 5)

    if max_chars <= 0:
        max_chars = 12000
    if max_items <= 0:
        max_items = 80
    if overlap_items < 0:
        overlap_items = 0

    return ChunkConfig(
        max_chars=max_chars,
        max_items=max_items,
        overlap_items=overlap_items,
    )


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
          ON d.source_id = ji.parent_source_id
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
    chunk_conf: ChunkConfig,
) -> list[str]:
    item = fetch_kokkai_job_item_meta(conn, job_item_id)
    rows = fetch_kokkai_content_rows(conn, job_item_id)

    chunks = build_chunks(
        rows,
        chunk_conf,
        allowed_content_types={"speech"},
    )

    if not chunks:
        raise HTTPException(status_code=400, detail=f"knowledge_contents not found: {job_item_id}")

    prompt_template = load_template_text(template_path).strip()

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
                SOURCE_TYPE,
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


class KnowledgeTargetItem(BaseModel):
    source_type: Optional[str] = Field(default=SOURCE_TYPE, description="kokkai")
    parent_source_id: Optional[str] = None
    parent_key1: Optional[str] = None
    parent_key2: Optional[str] = None
    parent_label: Optional[str] = None
    row_count: int = 0


class KnowledgeJobCreateRequest(BaseModel):
    source_type: Optional[str] = Field(default=SOURCE_TYPE, description="kokkai")
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


def validate_kokkai_request(body: KnowledgeJobCreateRequest) -> None:
    if body.source_type not in (None, "", SOURCE_TYPE):
        raise HTTPException(status_code=400, detail=f"source_type must be '{SOURCE_TYPE}'")
    if not body.items:
        raise HTTPException(status_code=400, detail="items is empty")

    for item in body.items:
        if item.source_type not in (None, "", SOURCE_TYPE):
            raise HTTPException(status_code=400, detail=f"item.source_type must be '{SOURCE_TYPE}'")


@router.post("/job", response_model=KnowledgeJobCreateResponse)
def create_kokkai_job(
    body: KnowledgeJobCreateRequest,
    authorization: str | None = Header(default=None),
):
    validate_kokkai_request(body)

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

    local_db_path = f"/tmp/ank_{uid}_knowledge.db"
    db_blob.download_to_filename(local_db_path)

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row

    try:
        chunk_config = load_chunk_config()
        qa_chunk_conf = get_kokkai_chunk_conf(chunk_config, "qa")
        plain_chunk_conf = get_kokkai_chunk_conf(chunk_config, "plain")

        job_id = new_id()
        requested_at = now_iso()

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
                body.source_name,
                body.request_type,
                "preview" if body.preview_only else "running",
                selected_count,
                requested_at,
            ),
        )

        created_item_count = 0
        total_plain_count = 0
        total_qa_count = 0
        total_error_count = 0
        prompt_previews: List[PromptPreviewItem] = []
        debug_items: List[KnowledgeDebugItem] = []

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
                    SOURCE_TYPE,
                    item.parent_source_id,
                    item.parent_key1,
                    item.parent_key2,
                    item.parent_label,
                    item.row_count,
                    "preview" if body.preview_only else "running",
                    requested_at,
                    requested_at,
                ),
            )

            try:
                source_id = item.parent_source_id or ""
                if not source_id:
                    raise HTTPException(status_code=400, detail="parent_source_id is required")

                qa_count = 0
                plain_count = 0
                qa_llm_result_for_debug: Optional[dict] = None
                plain_llm_result_for_debug: Optional[dict] = None

                insert_kokkai_contents(
                    conn=conn,
                    job_id=job_id,
                    job_item_id=job_item_id,
                    source_id=source_id,
                )

                qa_prompt_texts = build_kokkai_prompt_texts(
                    conn=conn,
                    job_item_id=job_item_id,
                    template_path=KOKKAI_QA_PROMPT_PATH,
                    chunk_conf=qa_chunk_conf,
                )

                plain_prompt_texts = build_kokkai_prompt_texts(
                    conn=conn,
                    job_item_id=job_item_id,
                    template_path=KOKKAI_PLAIN_PROMPT_PATH,
                    chunk_conf=plain_chunk_conf,
                )

                prompt_previews.append(
                    PromptPreviewItem(
                        job_item_id=job_item_id,
                        parent_source_id=item.parent_source_id,
                        parent_label=item.parent_label,
                        prompt_type="qa",
                        prompt_text=join_prompt_previews(qa_prompt_texts),
                    )
                )

                prompt_previews.append(
                    PromptPreviewItem(
                        job_item_id=job_item_id,
                        parent_source_id=item.parent_source_id,
                        parent_label=item.parent_label,
                        prompt_type="plain",
                        prompt_text=join_prompt_previews(plain_prompt_texts),
                    )
                )

                if not body.preview_only:
                    qa_chunk_results = run_chunked_llm_json(
                        qa_prompt_texts,
                        "LLM KOKKAI QA",
                    )
                    qa_llm_result = merge_qa_chunk_results(job_item_id, qa_chunk_results)
                    qa_llm_result_for_debug = {
                        "chunk_results": qa_chunk_results,
                        "merged_result": qa_llm_result,
                    }

                    qa_count = insert_qa_items_from_llm_result(
                        conn=conn,
                        job_id=job_id,
                        job_item_id=job_item_id,
                        source_id=source_id,
                        llm_result=qa_llm_result,
                    )

                    plain_chunk_results = run_chunked_llm_json(
                        plain_prompt_texts,
                        "LLM KOKKAI PLAIN",
                    )
                    plain_llm_result = merge_plain_chunk_results(job_item_id, plain_chunk_results)
                    plain_llm_result_for_debug = {
                        "chunk_results": plain_chunk_results,
                        "merged_result": plain_llm_result,
                    }

                    plain_count = insert_plain_items_from_llm_result(
                        conn=conn,
                        job_id=job_id,
                        job_item_id=job_item_id,
                        source_id=source_id,
                        llm_result=plain_llm_result,
                    )

                finished_at = now_iso()
                item_status = "preview" if body.preview_only else "ready"

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

                debug_items.append(
                    KnowledgeDebugItem(
                        job_item_id=job_item_id,
                        parent_label=item.parent_label,
                        status=item_status,
                        qa_count=qa_count,
                        plain_count=plain_count,
                        error_message=None,
                        llm_result={
                            "qa": qa_llm_result_for_debug,
                            "plain": plain_llm_result_for_debug,
                        },
                    )
                )

                created_item_count += 1
                total_qa_count += qa_count
                total_plain_count += plain_count

            except Exception as e:
                logger.exception("kokkai job item failed: job_id=%s job_item_id=%s", job_id, job_item_id)

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

                debug_items.append(
                    KnowledgeDebugItem(
                        job_item_id=job_item_id,
                        parent_label=item.parent_label,
                        status="error",
                        qa_count=0,
                        plain_count=0,
                        error_message=str(e),
                        llm_result=None,
                    )
                )

                raise

        finished_at = now_iso()
        final_status = (
            "partial_error"
            if total_error_count > 0
            else ("preview" if body.preview_only else "ready")
        )

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
            prompt_previews=prompt_previews,
            debug_items=debug_items,
        )

    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise HTTPException(status_code=409, detail=f"duplicate or integrity error: {e}")

    except Exception as e:
        conn.rollback()
        logger.exception("create_kokkai_job failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()
