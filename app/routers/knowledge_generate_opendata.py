from __future__ import annotations

import json
import os
import sqlite3
import uuid
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Optional, Any

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth

from .llm_client import run_chunked_llm_json
from .chunking import ChunkConfig, build_chunks
from .prompt_builder import build_opendata_prompt_text


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge/opendata", tags=["knowledge_opendata"])

JST = ZoneInfo("Asia/Tokyo")
BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
SOURCE_TYPE = "opendata"

OPENDATA_QA_PROMPT_PATH = "template/opendata_qa_prompt.txt"
OPENDATA_PLAIN_PROMPT_PATH = "template/opendata_plain_prompt.txt"
OPENAI_CHUNK_CONFIG_PATH = "template/openai_chunk.json"


DEFAULT_OPENDATA_QA_PROMPT = """あなたは、オープンデータから検索に使えるQAを抽出するアシスタントです。

入力として、同一データセットに属する複数の行データや説明文が与えられます。
内容を読み取り、利用価値のあるQAを抽出してください。

目的は、チャットボットや検索システムで再利用できるナレッジを作ることです。
そのため、表面的な言い換えではなく、意味のある質問と回答の組を作成してください。

出力は必ずJSONオブジェクトで返してください。
形式:
{
  "job_item_id": "...",
  "qa_list": [
    {
      "question": "...",
      "answer": "..."
    }
  ]
}

注意:
- 根拠が弱いものは作らない
- 回答は入力に含まれる情報だけを使う
- 推測で補わない
- 同じ意味のQAを重複して作らない
"""

DEFAULT_OPENDATA_PLAIN_PROMPT = """あなたは、オープンデータから検索に使える説明文を抽出するアシスタントです。

入力として、同一データセットに属する複数の行データや説明文が与えられます。
内容を読み取り、検索や要約に使える平文ナレッジを抽出してください。

出力は必ずJSONオブジェクトで返してください。
形式:
{
  "job_item_id": "...",
  "plain_list": [
    {
      "content": "..."
    }
  ]
}

注意:
- 重要な定義、制度概要、項目説明、集計の意味などを優先する
- 行データの断片をそのまま大量に返さない
- 推測で補わない
- 同じ意味の説明文を重複して作らない
"""


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


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def load_json_safe(text: str) -> dict | list | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def flatten_json_like(value: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []

    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            lines.extend(flatten_json_like(v, key))
        return lines

    if isinstance(value, list):
        for idx, item in enumerate(value):
            key = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            lines.extend(flatten_json_like(item, key))
        return lines

    text = normalize_text(str(value) if value is not None else "")
    if not text:
        return []

    if prefix:
        return [f"{prefix}: {text}"]
    return [text]


def extract_row_text(content_raw: str | None) -> str:
    src = (content_raw or "").strip()
    if not src:
        return ""

    parsed = load_json_safe(src)
    if parsed is None:
        return normalize_text(src)

    lines = flatten_json_like(parsed)
    if not lines:
        return normalize_text(src)

    return "\n".join(lines)


def load_template_text(path: str, default_text: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)

    if not blob.exists():
        return default_text.strip()

    return blob.download_as_bytes().decode("utf-8").strip()


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


def get_opendata_chunk_conf(chunk_config: dict, prompt_type: str) -> ChunkConfig:
    conf = ((chunk_config.get("opendata") or {}).get(prompt_type) or {})
    max_chars = int(conf.get("max_chars") or 10000)
    max_items = int(conf.get("max_items") or 60)
    overlap_items = int(conf.get("overlap_items") or 3)

    if prompt_type == "plain":
        if not conf.get("max_chars"):
            max_chars = 12000
        if not conf.get("max_items"):
            max_items = 100
        if not conf.get("overlap_items") and conf.get("overlap_items") != 0:
            overlap_items = 5

    if max_chars <= 0:
        max_chars = 10000 if prompt_type == "qa" else 12000
    if max_items <= 0:
        max_items = 60 if prompt_type == "qa" else 100
    if overlap_items < 0:
        overlap_items = 0

    return ChunkConfig(
        max_chars=max_chars,
        max_items=max_items,
        overlap_items=overlap_items,
    )


def open_user_db(local_db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(local_db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def fetch_opendata_source_rows(local_db_path: str, source_id: str) -> list[sqlite3.Row]:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT
                row_id,
                row_index,
                source_item_id,
                content
            FROM row_data
            WHERE source_type = 'opendata'
              AND file_id = ?
            ORDER BY row_index
            """,
            (source_id,),
        )
        return cur.fetchall()
    finally:
        conn.close()


def fetch_job_row(local_db_path: str, job_id: str) -> sqlite3.Row | None:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
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
            WHERE job_id = ?
            LIMIT 1
            """,
            (job_id,),
        )
        return cur.fetchone()
    finally:
        conn.close()


def fetch_job_items(local_db_path: str, job_id: str) -> list[sqlite3.Row]:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT
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
            FROM knowledge_job_items
            WHERE job_id = ?
            ORDER BY created_at, job_item_id
            """,
            (job_id,),
        )
        return cur.fetchall()
    finally:
        conn.close()


def fetch_next_queued_job_item(local_db_path: str, job_id: str) -> sqlite3.Row | None:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT
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
            FROM knowledge_job_items
            WHERE job_id = ?
              AND status = 'queued'
            ORDER BY created_at, job_item_id
            LIMIT 1
            """,
            (job_id,),
        )
        return cur.fetchone()
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
    template_path: str,
    default_template: str,
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

    prompt_template = load_template_text(template_path, default_template)

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
        "opendata prompt build: job_item_id=%s chunk_count=%s template=%s",
        job_item_id,
        len(prompt_texts),
        template_path,
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
    job_id: str,
    job_item_id: str,
    source_id: str,
    source_rows: list[sqlite3.Row],
) -> int:
    inserted_count = 0
    now = now_iso()

    for idx, row in enumerate(source_rows, start=1):
        content_text = extract_row_text(row["content"])
        if not content_text:
            continue

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
                row["source_item_id"],
                row["row_id"],
                "row",
                content_text,
                idx,
                now,
                now,
            ),
        )
        inserted_count += 1

    return inserted_count


def prepare_job_item_and_prompts(
    local_db_path: str,
    job_id: str,
    item: "KnowledgeTargetItem",
    requested_at: str,
    qa_chunk_conf: ChunkConfig,
    plain_chunk_conf: ChunkConfig,
    preview_only: bool,
) -> dict[str, Any]:
    source_id = item.parent_source_id or ""
    if not source_id:
        raise HTTPException(status_code=400, detail="parent_source_id is required")

    source_rows = fetch_opendata_source_rows(local_db_path, source_id)
    if not source_rows:
        raise HTTPException(status_code=400, detail=f"row_data not found: {source_id}")

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
                "preview" if preview_only else "queued",
                requested_at,
                requested_at,
            ),
        )

        insert_opendata_contents(
            conn=conn,
            job_id=job_id,
            job_item_id=job_item_id,
            source_id=source_id,
            source_rows=source_rows,
        )

        qa_prompt_texts = build_opendata_prompt_texts(
            conn=conn,
            job_item_id=job_item_id,
            template_path=OPENDATA_QA_PROMPT_PATH,
            default_template=DEFAULT_OPENDATA_QA_PROMPT,
            chunk_conf=qa_chunk_conf,
        )

        plain_prompt_texts = build_opendata_prompt_texts(
            conn=conn,
            job_item_id=job_item_id,
            template_path=OPENDATA_PLAIN_PROMPT_PATH,
            default_template=DEFAULT_OPENDATA_PLAIN_PROMPT,
            chunk_conf=plain_chunk_conf,
        )

        conn.commit()

        return {
            "job_item_id": job_item_id,
            "source_id": source_id,
            "qa_prompt_texts": qa_prompt_texts,
            "plain_prompt_texts": plain_prompt_texts,
        }

    except Exception:
        conn.rollback()
        raise
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
        item_status = "preview" if preview_only else "ready"

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
                started_at = COALESCE(started_at, ?),
                finished_at = ?
            WHERE job_id = ?
            """,
            (
                status,
                total_qa_count,
                total_plain_count,
                total_error_count,
                requested_at,
                now_iso(),
                job_id,
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


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
                "preview" if body.preview_only else "queued",
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


def process_opendata_job_item(
    local_db_path: str,
    job_id: str,
    requested_at: str,
    row: sqlite3.Row,
    qa_chunk_conf: ChunkConfig,
    plain_chunk_conf: ChunkConfig,
    preview_only: bool,
) -> dict[str, Any]:
    item = KnowledgeTargetItem(
        source_type=row["source_type"],
        parent_source_id=row["parent_source_id"],
        parent_key1=row["parent_key1"],
        parent_key2=row["parent_key2"],
        parent_label=row["parent_label"],
        row_count=row["row_count"] or 0,
    )

    prepared = prepare_job_item_and_prompts(
        local_db_path=local_db_path,
        job_id=job_id,
        item=item,
        requested_at=requested_at,
        qa_chunk_conf=qa_chunk_conf,
        plain_chunk_conf=plain_chunk_conf,
        preview_only=preview_only,
    )

    job_item_id = prepared["job_item_id"]
    source_id = prepared["source_id"]
    qa_prompt_texts = prepared["qa_prompt_texts"]
    plain_prompt_texts = prepared["plain_prompt_texts"]

    if preview_only:
        qa_llm_result = {"job_item_id": job_item_id, "qa_list": []}
        plain_llm_result = {"job_item_id": job_item_id, "plain_list": []}
        qa_llm_result_for_debug = None
        plain_llm_result_for_debug = None
    else:
        mark_job_item_running(local_db_path, job_item_id)

        qa_chunk_results = run_chunked_llm_json(
            qa_prompt_texts,
            "LLM OPENDATA QA",
        )
        qa_llm_result = merge_qa_chunk_results(job_item_id, qa_chunk_results)
        qa_llm_result_for_debug = {
            "chunk_results": qa_chunk_results,
            "merged_result": qa_llm_result,
        }

        plain_chunk_results = run_chunked_llm_json(
            plain_prompt_texts,
            "LLM OPENDATA PLAIN",
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
        "parent_source_id": item.parent_source_id,
        "parent_label": item.parent_label,
        "qa_count": qa_count,
        "plain_count": plain_count,
        "status": "preview" if preview_only else "ready",
        "qa_prompt_texts": qa_prompt_texts,
        "plain_prompt_texts": plain_prompt_texts,
        "qa_debug": qa_llm_result_for_debug,
        "plain_debug": plain_llm_result_for_debug,
    }


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


class KnowledgeJobStatusItem(BaseModel):
    job_item_id: str
    parent_source_id: Optional[str] = None
    parent_label: Optional[str] = None
    status: str
    knowledge_count: int = 0
    error_message: Optional[str] = None
    row_count: int = 0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class KnowledgeJobStatusResponse(BaseModel):
    job_id: str
    status: str
    selected_count: int = 0
    qa_count: int = 0
    plain_count: int = 0
    error_count: int = 0
    requested_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None
    items: List[KnowledgeJobStatusItem] = []


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

    local_db_path = f"/tmp/ank_{uid}_knowledge_opendata.db"
    db_blob.download_to_filename(local_db_path)

    try:
        chunk_config = load_chunk_config()
        qa_chunk_conf = get_opendata_chunk_conf(chunk_config, "qa")
        plain_chunk_conf = get_opendata_chunk_conf(chunk_config, "plain")

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
            prepared = prepare_job_item_and_prompts(
                local_db_path=local_db_path,
                job_id=job_id,
                item=item,
                requested_at=requested_at,
                qa_chunk_conf=qa_chunk_conf,
                plain_chunk_conf=plain_chunk_conf,
                preview_only=body.preview_only,
            )

            prompt_previews.append(
                PromptPreviewItem(
                    job_item_id=prepared["job_item_id"],
                    parent_source_id=item.parent_source_id,
                    parent_label=item.parent_label,
                    prompt_type="qa",
                    prompt_text=join_prompt_previews(prepared["qa_prompt_texts"]),
                )
            )
            prompt_previews.append(
                PromptPreviewItem(
                    job_item_id=prepared["job_item_id"],
                    parent_source_id=item.parent_source_id,
                    parent_label=item.parent_label,
                    prompt_type="plain",
                    prompt_text=join_prompt_previews(prepared["plain_prompt_texts"]),
                )
            )

            debug_items.append(
                KnowledgeDebugItem(
                    job_item_id=prepared["job_item_id"],
                    parent_label=item.parent_label,
                    status="preview" if body.preview_only else "queued",
                    qa_count=0,
                    plain_count=0,
                    error_message=None,
                    llm_result=None,
                )
            )
            created_item_count += 1

        initial_status = "preview" if body.preview_only else "queued"
        update_job_summary(
            local_db_path=local_db_path,
            job_id=job_id,
            requested_at=requested_at,
            status=initial_status,
            total_qa_count=0,
            total_plain_count=0,
            total_error_count=0,
        )

        db_blob.upload_from_filename(local_db_path)

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

    local_db_path = f"/tmp/ank_{uid}_knowledge_opendata_run.db"
    db_blob.download_to_filename(local_db_path)

    try:
        job_row = fetch_job_row(local_db_path, body.job_id)
        if not job_row:
            raise HTTPException(status_code=404, detail=f"knowledge_jobs not found: {body.job_id}")
        if job_row["source_type"] != SOURCE_TYPE:
            raise HTTPException(status_code=400, detail=f"job source_type must be '{SOURCE_TYPE}'")

        if job_row["status"] == "ready":
            items = fetch_job_items(local_db_path, body.job_id)
            return KnowledgeJobCreateResponse(
                job_id=body.job_id,
                selected_count=job_row["selected_count"] or 0,
                created_item_count=len(items),
                status="ready",
                prompt_previews=[],
                debug_items=[],
            )

        if job_row["status"] == "preview":
            raise HTTPException(status_code=400, detail="preview job cannot be run")

        chunk_config = load_chunk_config()
        qa_chunk_conf = get_opendata_chunk_conf(chunk_config, "qa")
        plain_chunk_conf = get_opendata_chunk_conf(chunk_config, "plain")

        requested_at = job_row["requested_at"] or now_iso()
        selected_count = job_row["selected_count"] or 0

        prompt_previews: List[PromptPreviewItem] = []
        debug_items: List[KnowledgeDebugItem] = []
        created_item_count = 0
        total_qa_count = int(job_row["qa_count"] or 0)
        total_plain_count = int(job_row["plain_count"] or 0)
        total_error_count = int(job_row["error_count"] or 0)

        conn = open_user_db(local_db_path)
        try:
            conn.execute("BEGIN")
            conn.execute(
                """
                UPDATE knowledge_jobs
                SET status = 'running',
                    started_at = COALESCE(started_at, ?),
                    error_message = NULL
                WHERE job_id = ?
                """,
                (now_iso(), body.job_id),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        while True:
            row = fetch_next_queued_job_item(local_db_path, body.job_id)
            if not row:
                break

            job_item_id = None
            try:
                result = process_opendata_job_item(
                    local_db_path=local_db_path,
                    job_id=body.job_id,
                    requested_at=requested_at,
                    row=row,
                    qa_chunk_conf=qa_chunk_conf,
                    plain_chunk_conf=plain_chunk_conf,
                    preview_only=False,
                )

                job_item_id = result["job_item_id"]

                prompt_previews.append(
                    PromptPreviewItem(
                        job_item_id=job_item_id,
                        parent_source_id=result["parent_source_id"],
                        parent_label=result["parent_label"],
                        prompt_type="qa",
                        prompt_text=join_prompt_previews(result["qa_prompt_texts"]),
                    )
                )
                prompt_previews.append(
                    PromptPreviewItem(
                        job_item_id=job_item_id,
                        parent_source_id=result["parent_source_id"],
                        parent_label=result["parent_label"],
                        prompt_type="plain",
                        prompt_text=join_prompt_previews(result["plain_prompt_texts"]),
                    )
                )

                debug_items.append(
                    KnowledgeDebugItem(
                        job_item_id=job_item_id,
                        parent_label=result["parent_label"],
                        status=result["status"],
                        qa_count=result["qa_count"],
                        plain_count=result["plain_count"],
                        error_message=None,
                        llm_result={
                            "qa": result["qa_debug"],
                            "plain": result["plain_debug"],
                        },
                    )
                )

                created_item_count += 1
                total_qa_count += result["qa_count"]
                total_plain_count += result["plain_count"]

            except Exception as e:
                logger.exception("run_opendata_job item failed: job_id=%s job_item_id=%s", body.job_id, job_item_id)

                if job_item_id:
                    try:
                        finalize_job_item_error(
                            local_db_path=local_db_path,
                            job_item_id=job_item_id,
                            error_message=str(e),
                        )
                    except Exception:
                        logger.exception("failed to update error state: job_item_id=%s", job_item_id)

                debug_items.append(
                    KnowledgeDebugItem(
                        job_item_id=job_item_id or "",
                        parent_label=row["parent_label"],
                        status="error",
                        qa_count=0,
                        plain_count=0,
                        error_message=str(e),
                        llm_result=None,
                    )
                )
                total_error_count += 1
                break

        final_status = "partial_error" if total_error_count > 0 else "ready"

        update_job_summary(
            local_db_path=local_db_path,
            job_id=body.job_id,
            requested_at=requested_at,
            status=final_status,
            total_qa_count=total_qa_count,
            total_plain_count=total_plain_count,
            total_error_count=total_error_count,
        )

        db_blob.upload_from_filename(local_db_path)

        return KnowledgeJobCreateResponse(
            job_id=body.job_id,
            selected_count=selected_count,
            created_item_count=created_item_count,
            status=final_status,
            prompt_previews=prompt_previews,
            debug_items=debug_items,
        )

    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail=f"duplicate or integrity error: {e}")

    except Exception as e:
        logger.exception("run_opendata_job failed")
        raise HTTPException(status_code=500, detail=f"run_opendata_job failed: {type(e).__name__}: {e}")


@router.get("/status", response_model=KnowledgeJobStatusResponse)
def get_opendata_job_status(
    job_id: str = Query(...),
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

    local_db_path = f"/tmp/ank_{uid}_knowledge_opendata_status.db"
    db_blob.download_to_filename(local_db_path)

    try:
        job_row = fetch_job_row(local_db_path, job_id)
        if not job_row:
            raise HTTPException(status_code=404, detail=f"knowledge_jobs not found: {job_id}")
        if job_row["source_type"] != SOURCE_TYPE:
            raise HTTPException(status_code=400, detail=f"job source_type must be '{SOURCE_TYPE}'")

        item_rows = fetch_job_items(local_db_path, job_id)
        items = [
            KnowledgeJobStatusItem(
                job_item_id=row["job_item_id"],
                parent_source_id=row["parent_source_id"],
                parent_label=row["parent_label"],
                status=row["status"] or "",
                knowledge_count=int(row["knowledge_count"] or 0),
                error_message=row["error_message"],
                row_count=int(row["row_count"] or 0),
                started_at=row["started_at"],
                finished_at=row["finished_at"],
            )
            for row in item_rows
        ]

        return KnowledgeJobStatusResponse(
            job_id=job_row["job_id"],
            status=job_row["status"] or "",
            selected_count=int(job_row["selected_count"] or 0),
            qa_count=int(job_row["qa_count"] or 0),
            plain_count=int(job_row["plain_count"] or 0),
            error_count=int(job_row["error_count"] or 0),
            requested_at=job_row["requested_at"],
            started_at=job_row["started_at"],
            finished_at=job_row["finished_at"],
            error_message=job_row["error_message"],
            items=items,
        )

    except Exception as e:
        logger.exception("get_opendata_job_status failed")
        raise HTTPException(status_code=500, detail=f"get_opendata_job_status failed: {type(e).__name__}: {e}")
