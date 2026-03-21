from __future__ import annotations

import json
import os
import sqlite3
import uuid
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Optional, Any

from fastapi import APIRouter, Header, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth

from .openai_llm_client import run_chunked_llm_json
from .openai_chunking import ChunkConfig, build_chunks
from .openai_prompt_builder import build_kokkai_prompt_text


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge/kokkai", tags=["knowledge_kokkai"])

JST = ZoneInfo("Asia/Tokyo")
BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
SOURCE_TYPE = "kokkai"

KOKKAI_QA_PROMPT_PATH = "template/kokkai_qa_prompt.txt"
KOKKAI_PLAIN_PROMPT_PATH = "template/kokkai_plain_prompt.txt"
OPENAI_CHUNK_CONFIG_PATH = "template/openai_chunk.json"

DEFAULT_KOKKAI_QA_PROMPT = """あなたは、国会ぎじろくから検索に使えるQAを抽出するアシスタントです。

入力として、同一の会議に属する複数の発言が与えられます。
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

DEFAULT_KOKKAI_PLAIN_PROMPT = """あなたは、国会ぎじろくから検索に使える説明文を抽出するアシスタントです。

入力として、同一の会議に属する複数の発言が与えられます。
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
- 重要な定義、制度概要、方針、手順、要点、注意事項を優先する
- 発言の断片をそのまま大量に返さない
- 推測で補わない
- 同じ意味の説明文を重複して作らない
"""


def now_iso() -> str:
    return datetime.now(tz=JST).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


def local_user_db_path(uid: str) -> str:
    return f"/tmp/ank_{uid}.db"


def row_to_status_item(row: sqlite3.Row) -> dict[str, Any]:
    qa_chunk_total = int(row["qa_chunk_total"] or 0)
    qa_chunk_done = int(row["qa_chunk_done"] or 0)
    plain_chunk_total = int(row["plain_chunk_total"] or 0)
    plain_chunk_done = int(row["plain_chunk_done"] or 0)

    return {
        "job_item_id": row["job_item_id"],
        "parent_source_id": row["parent_source_id"],
        "parent_label": row["parent_label"],
        "status": row["status"] or "",
        "knowledge_count": int(row["knowledge_count"] or 0),
        "error_message": row["error_message"],
        "row_count": int(row["row_count"] or 0),
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "qa_chunk_total": qa_chunk_total,
        "qa_chunk_done": qa_chunk_done,
        "plain_chunk_total": plain_chunk_total,
        "plain_chunk_done": plain_chunk_done,
        "chunk_total": qa_chunk_total + plain_chunk_total,
        "chunk_done": qa_chunk_done + plain_chunk_done,
    }


def build_status_payload_from_db(local_db_path: str, job_id: str) -> dict[str, Any]:
    job_row = fetch_job_row(local_db_path, job_id)
    if not job_row:
        raise HTTPException(status_code=404, detail=f"knowledge_jobs not found: {job_id}")

    item_rows = fetch_job_items(local_db_path, job_id)

    total_qa_chunks = sum(int(row["qa_chunk_total"] or 0) for row in item_rows)
    processed_qa_chunks = sum(int(row["qa_chunk_done"] or 0) for row in item_rows)
    total_plain_chunks = sum(int(row["plain_chunk_total"] or 0) for row in item_rows)
    processed_plain_chunks = sum(int(row["plain_chunk_done"] or 0) for row in item_rows)

    return {
        "job_id": job_row["job_id"],
        "status": job_row["status"] or "",
        "selected_count": int(job_row["selected_count"] or 0),
        "qa_count": int(job_row["qa_count"] or 0),
        "plain_count": int(job_row["plain_count"] or 0),
        "error_count": int(job_row["error_count"] or 0),
        "requested_at": job_row["requested_at"],
        "started_at": job_row["started_at"],
        "finished_at": job_row["finished_at"],
        "error_message": job_row["error_message"],
        "total_qa_chunks": total_qa_chunks,
        "processed_qa_chunks": processed_qa_chunks,
        "total_plain_chunks": total_plain_chunks,
        "processed_plain_chunks": processed_plain_chunks,
        "total_chunks": total_qa_chunks + total_plain_chunks,
        "processed_chunks": processed_qa_chunks + processed_plain_chunks,
        "items": [row_to_status_item(row) for row in item_rows],
    }


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


def load_json_safe(text: str) -> dict | list | None:
    try:
        return json.loads(text)
    except Exception:
        return None


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


def open_user_db(local_db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(local_db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def upload_local_db(db_blob: storage.Blob, local_db_path: str) -> None:
    if not os.path.exists(local_db_path):
        raise FileNotFoundError(f"local db not found: {local_db_path}")

    snapshot_path = f"{local_db_path}.upload.sqlite"

    src_conn = None
    dst_conn = None

    try:
        src_conn = sqlite3.connect(local_db_path, timeout=30)
        src_conn.row_factory = sqlite3.Row
        src_conn.execute("PRAGMA busy_timeout = 30000")

        journal_mode = src_conn.execute("PRAGMA journal_mode").fetchone()[0]
        logger.info("upload_local_db journal_mode=%s path=%s", journal_mode, local_db_path)

        if str(journal_mode).lower() == "wal":
            src_conn.execute("PRAGMA wal_checkpoint(FULL)")
            logger.info("upload_local_db wal checkpoint done: %s", local_db_path)

        dst_conn = sqlite3.connect(snapshot_path, timeout=30)
        src_conn.backup(dst_conn)
        dst_conn.commit()

    finally:
        if dst_conn is not None:
            dst_conn.close()
        if src_conn is not None:
            src_conn.close()

    db_blob.upload_from_filename(snapshot_path)
    logger.info("db uploaded to gcs: %s", db_blob.name)

    try:
        os.remove(snapshot_path)
    except Exception:
        logger.warning("failed to remove snapshot file: %s", snapshot_path)


LOCK_TTL_SECONDS = 60 * 60 * 6


def ensure_job_locks_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_locks (
            lock_key TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            locked_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
        """
    )


def build_lock_key(uid: str) -> str:
    return f"{uid}:{SOURCE_TYPE}"


def try_acquire_job_lock(local_db_path: str, lock_key: str, job_id: str) -> bool:
    conn = open_user_db(local_db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        ensure_job_locks_table(conn)
        now = now_iso()
        expires_at = datetime.fromisoformat(now).timestamp() + LOCK_TTL_SECONDS
        expires_iso = datetime.fromtimestamp(expires_at, tz=JST).isoformat()
        conn.execute("DELETE FROM job_locks WHERE expires_at < ?", (now,))
        try:
            conn.execute(
                "INSERT INTO job_locks (lock_key, job_id, locked_at, expires_at) VALUES (?, ?, ?, ?)",
                (lock_key, job_id, now, expires_iso),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            conn.rollback()
            return False
    finally:
        conn.close()


def release_job_lock(local_db_path: str, lock_key: str, job_id: str | None = None) -> None:
    conn = open_user_db(local_db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        ensure_job_locks_table(conn)
        if job_id:
            conn.execute("DELETE FROM job_locks WHERE lock_key = ? AND job_id = ?", (lock_key, job_id))
        else:
            conn.execute("DELETE FROM job_locks WHERE lock_key = ?", (lock_key,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_running_lock_job_id(local_db_path: str, lock_key: str) -> str | None:
    conn = open_user_db(local_db_path)
    try:
        ensure_job_locks_table(conn)
        now = now_iso()
        conn.execute("DELETE FROM job_locks WHERE expires_at < ?", (now,))
        conn.commit()
        cur = conn.execute("SELECT job_id FROM job_locks WHERE lock_key = ? LIMIT 1", (lock_key,))
        row = cur.fetchone()
        return row["job_id"] if row else None
    finally:
        conn.close()


def fetch_kokkai_source_rows(local_db_path: str, issue_id: str) -> list[sqlite3.Row]:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT
                issue_id,
                speech_id,
                status,
                speaker,
                speech,
                created_at,
                updated_at
            FROM kokkai_document_rows
            WHERE issue_id = ?
            ORDER BY COALESCE(speech_order, 0), speech_id
            """,
            (issue_id,),
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
                finished_at,
                qa_chunk_total,
                qa_chunk_done,
                plain_chunk_total,
                plain_chunk_done
            FROM knowledge_job_items
            WHERE job_id = ?
            ORDER BY created_at, job_item_id
            """,
            (job_id,),
        )
        return cur.fetchall()
    finally:
        conn.close()


def fetch_next_new_job_item(local_db_path: str, job_id: str) -> sqlite3.Row | None:
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
              AND status = 'new'
            ORDER BY created_at, job_item_id
            LIMIT 1
            """,
            (job_id,),
        )
        return cur.fetchone()
    finally:
        conn.close()


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
        speech_text = normalize_text(row["speech"])
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


def process_kokkai_job_item(
    local_db_path: str,
    job_id: str,
    job_item_id: str,
    qa_chunk_conf: ChunkConfig,
    plain_chunk_conf: ChunkConfig,
    preview_only: bool,
    db_blob: storage.Blob,
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

    update_job_item_chunk_totals(
        local_db_path=local_db_path,
        job_item_id=job_item_id,
        qa_chunk_total=len(qa_prompt_texts),
        plain_chunk_total=len(plain_prompt_texts),
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
            len(qa_prompt_texts),
            len(plain_prompt_texts),
        )

        qa_chunk_results = []
        for prompt_text in qa_prompt_texts:
            result_list = run_chunked_llm_json([prompt_text], "LLM KOKKAI QA")
            if not result_list:
                raise Exception("empty qa chunk result")
            qa_chunk_results.append(result_list[0])
            increment_job_item_chunk_done(local_db_path, job_item_id, "qa", db_blob)

        qa_llm_result = merge_qa_chunk_results(job_item_id, qa_chunk_results)
        qa_llm_result_for_debug = {
            "chunk_results": qa_chunk_results,
            "merged_result": qa_llm_result,
        }

        plain_chunk_results = []
        for prompt_text in plain_prompt_texts:
            result_list = run_chunked_llm_json([prompt_text], "LLM KOKKAI PLAIN")
            if not result_list:
                raise Exception("empty plain chunk result")
            plain_chunk_results.append(result_list[0])
            increment_job_item_chunk_done(local_db_path, job_item_id, "plain", db_blob)

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
        return

    local_db_path = local_user_db_path(uid)
    lock_key = build_lock_key(uid)

    try:
        if not os.path.exists(local_db_path):
            db_blob.download_to_filename(local_db_path)

        job_row = fetch_job_row(local_db_path, job_id)
        if not job_row:
            logger.error("knowledge_jobs not found in background: %s", job_id)
            return

        if job_row["source_type"] != SOURCE_TYPE:
            update_job_summary(
                local_db_path=local_db_path,
                job_id=job_id,
                requested_at=(job_row["requested_at"] or now_iso()),
                status="error",
                total_qa_count=0,
                total_plain_count=0,
                total_error_count=1,
                error_message="invalid source_type",
            )
            upload_local_db(db_blob, local_db_path)
            logger.error("job source_type mismatch: job_id=%s source_type=%s", job_id, job_row["source_type"])
            return

        if job_row["status"] == "done":
            upload_local_db(db_blob, local_db_path)
            return

        chunk_config = load_chunk_config()
        qa_chunk_conf = get_kokkai_chunk_conf(chunk_config, "qa")
        plain_chunk_conf = get_kokkai_chunk_conf(chunk_config, "plain")

        requested_at = job_row["requested_at"] or now_iso()
        total_qa_count = int(job_row["qa_count"] or 0)
        total_plain_count = int(job_row["plain_count"] or 0)
        total_error_count = int(job_row["error_count"] or 0)

        logger.info("run_kokkai_job_background start: job_id=%s", job_id)

        while True:
            row = fetch_next_new_job_item(local_db_path, job_id)
            if not row:
                logger.info("no new items: job_id=%s", job_id)
                break

            current_job_item_id = row["job_item_id"]
            logger.info("processing item in background: job_id=%s job_item_id=%s", job_id, current_job_item_id)

            try:
                mark_job_item_running(local_db_path, current_job_item_id)

                result = process_kokkai_job_item(
                    local_db_path=local_db_path,
                    job_id=job_id,
                    job_item_id=current_job_item_id,
                    qa_chunk_conf=qa_chunk_conf,
                    plain_chunk_conf=plain_chunk_conf,
                    preview_only=False,
                    db_blob=db_blob,
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

            except Exception as e:
                logger.exception("run_kokkai_job_background item failed: job_id=%s job_item_id=%s", job_id, current_job_item_id)
                try:
                    finalize_job_item_error(
                        local_db_path=local_db_path,
                        job_item_id=current_job_item_id,
                        error_message=str(e),
                    )
                except Exception:
                    logger.exception("failed to update error state: job_item_id=%s", current_job_item_id)

                total_error_count += 1
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
                upload_local_db(db_blob, local_db_path)
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

        logger.info("run_kokkai_job_background finished: job_id=%s status=done", job_id)

    except Exception as e:
        logger.exception("run_kokkai_job_background failed: job_id=%s", job_id)

        try:
            job_row = fetch_job_row(local_db_path, job_id)
            requested_at = (job_row["requested_at"] if job_row else None) or now_iso()
            total_qa_count = int(job_row["qa_count"] or 0) if job_row else 0
            total_plain_count = int(job_row["plain_count"] or 0) if job_row else 0
            total_error_count = int(job_row["error_count"] or 0) if job_row else 0

            update_job_summary(
                local_db_path=local_db_path,
                job_id=job_id,
                requested_at=requested_at,
                status="error",
                total_qa_count=total_qa_count,
                total_plain_count=total_plain_count,
                total_error_count=total_error_count + 1,
                error_message=f"{type(e).__name__}: {e}",
            )
            upload_local_db(db_blob, local_db_path)
        except Exception:
            logger.exception("failed to update job error state in background: job_id=%s", job_id)
    finally:
        try:
            release_job_lock(local_db_path, lock_key, job_id)
        except Exception:
            logger.exception("failed to release job lock: job_id=%s", job_id)


def update_job_item_chunk_totals(
    local_db_path: str,
    job_item_id: str,
    qa_chunk_total: int,
    plain_chunk_total: int,
) -> None:
    conn = open_user_db(local_db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            UPDATE knowledge_job_items
            SET qa_chunk_total = ?,
                qa_chunk_done = COALESCE(qa_chunk_done, 0),
                plain_chunk_total = ?,
                plain_chunk_done = COALESCE(plain_chunk_done, 0)
            WHERE job_item_id = ?
            """,
            (qa_chunk_total, plain_chunk_total, job_item_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def increment_job_item_chunk_done(
    local_db_path: str,
    job_item_id: str,
    prompt_type: str,
    db_blob: storage.Blob,
) -> None:
    if prompt_type not in ("qa", "plain"):
        raise ValueError(f"invalid prompt_type: {prompt_type}")

    column = "qa_chunk_done" if prompt_type == "qa" else "plain_chunk_done"

    conn = open_user_db(local_db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            f"""
            UPDATE knowledge_job_items
            SET {column} = COALESCE({column}, 0) + 1
            WHERE job_item_id = ?
            """,
            (job_item_id,),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    upload_local_db(db_blob, local_db_path)

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
    qa_chunk_total: int = 0
    qa_chunk_done: int = 0
    plain_chunk_total: int = 0
    plain_chunk_done: int = 0
    chunk_total: int = 0
    chunk_done: int = 0


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
    total_qa_chunks: int = 0
    processed_qa_chunks: int = 0
    total_plain_chunks: int = 0
    processed_plain_chunks: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
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
def create_kokkai_job(
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
    db_blob.download_to_filename(local_db_path)

    try:
        chunk_config = load_chunk_config()
        qa_chunk_conf = get_kokkai_chunk_conf(chunk_config, "qa")
        plain_chunk_conf = get_kokkai_chunk_conf(chunk_config, "plain")

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
        logger.exception("create_kokkai_job failed")
        raise HTTPException(status_code=500, detail=f"create_kokkai_job failed: {type(e).__name__}: {e}")


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

    local_db_path = local_user_db_path(uid)
    db_blob.download_to_filename(local_db_path)

    try:
        job_row = fetch_job_row(local_db_path, body.job_id)
        if not job_row:
            raise HTTPException(status_code=404, detail=f"knowledge_jobs not found: {body.job_id}")
        if job_row["source_type"] != SOURCE_TYPE:
            raise HTTPException(status_code=400, detail=f"job source_type must be '{SOURCE_TYPE}'")

        if job_row["status"] == "done":
            payload = build_status_payload_from_db(local_db_path, body.job_id)
            return KnowledgeJobCreateResponse(
                job_id=body.job_id,
                selected_count=int(payload["selected_count"] or 0),
                created_item_count=len(payload["items"]),
                status="done",
                prompt_previews=[],
                debug_items=[],
            )

        if job_row["status"] == "running":
            payload = build_status_payload_from_db(local_db_path, body.job_id)
            return KnowledgeJobCreateResponse(
                job_id=body.job_id,
                selected_count=int(payload["selected_count"] or 0),
                created_item_count=len(payload["items"]),
                status="running",
                prompt_previews=[],
                debug_items=[],
            )

        lock_key = build_lock_key(uid)
        other_running_job_id = get_running_lock_job_id(local_db_path, lock_key)
        if other_running_job_id and other_running_job_id != body.job_id:
            raise HTTPException(
                status_code=409,
                detail=f"別のジョブが実行中です。完了後に再実行してください。 running_job_id={other_running_job_id}",
            )
        if not other_running_job_id:
            acquired = try_acquire_job_lock(local_db_path, lock_key, body.job_id)
            if not acquired:
                other_running_job_id = get_running_lock_job_id(local_db_path, lock_key)
                raise HTTPException(
                    status_code=409,
                    detail=f"別のジョブが実行中です。完了後に再実行してください。 running_job_id={other_running_job_id or ''}",
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

        upload_local_db(db_blob, local_db_path)

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

    try:
        if os.path.exists(local_db_path):
            os.remove(local_db_path)

        wal_path = f"{local_db_path}-wal"
        shm_path = f"{local_db_path}-shm"

        if os.path.exists(wal_path):
            os.remove(wal_path)
        if os.path.exists(shm_path):
            os.remove(shm_path)
    except Exception as e:
        logger.warning("failed to clear local db cache: %s", e)

    db_blob.download_to_filename(local_db_path)

    try:
        payload = build_status_payload_from_db(local_db_path, job_id)
        return KnowledgeJobStatusResponse(**payload)

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("get_kokkai_job_status failed")
        raise HTTPException(
            status_code=500,
            detail=f"get_kokkai_job_status failed: {type(e).__name__}: {e}"
        )
