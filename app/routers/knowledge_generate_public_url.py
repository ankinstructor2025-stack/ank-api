from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any, List, Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query
from google.cloud import storage
from pydantic import BaseModel, Field

from .content_splitter_text import build_record, clean_text, split_paragraph_blocks, split_text_records
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
    replace_local_db_from_blob,
    set_job_done,
    set_job_error,
    set_job_progress,
    set_job_running,
    try_read_status_payload,
    upload_local_db,
    user_db_path,
)
from .openai_chunking import ChunkConfig, build_chunks
from .openai_llm_client import run_chunked_llm_json
from .openai_prompt_builder import build_upload_prompt_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge/url", tags=["knowledge_url"])

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
SOURCE_TYPE = "public_url"

URL_QA_PROMPT_PATH = "template/public_url_qa_prompt.txt"
URL_PLAIN_PROMPT_PATH = "template/public_url_plain_prompt.txt"
OPENAI_CHUNK_CONFIG_PATH = "template/openai_chunk.json"

DEFAULT_URL_QA_PROMPT = """あなたは、公開URLから抽出された本文テキストを読み取り、検索に使えるQAを作成するアシスタントです。

入力として、同じURLルートに属する複数ページの本文ブロックが与えられます。
ページの見出し、説明、注意事項、手順、制度説明、定義、条件などを読み取り、利用価値のあるQAを抽出してください。

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
- 回答は入力テキストだけを根拠にする
- 推測や補完をしない
- 単なる言い換えを大量に作らない
- 同じ意味のQAを重複して作らない
- ページ固有の条件や対象者がある場合は質問文か回答文に残す
"""

DEFAULT_URL_PLAIN_PROMPT = """あなたは、公開URLから抽出された本文テキストを読み取り、検索に使える平文ナレッジを作成するアシスタントです。

入力として、同じURLルートに属する複数ページの本文ブロックが与えられます。
ページの見出し、説明、注意事項、手順、制度説明、定義、条件などを読み取り、検索に使える説明文を抽出してください。

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
- 重要な定義、ルール、手順、要点、注意事項を優先する
- 入力断片をそのまま大量に返さない
- 推測で補わない
- 同じ意味の説明文を重複して作らない
"""


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


class KnowledgeTargetItem(BaseModel):
    source_type: Optional[str] = Field(default=SOURCE_TYPE, description="public_url")
    parent_source_id: Optional[str] = None
    parent_key1: Optional[str] = None
    parent_key2: Optional[str] = None
    parent_label: Optional[str] = None
    row_count: int = 0


class KnowledgeJobCreateRequest(BaseModel):
    source_type: Optional[str] = Field(default=SOURCE_TYPE, description="public_url")
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


def load_template_text_or_default(path: str, default_text: str) -> str:
    try:
        return load_template_text(BUCKET_NAME, path)
    except HTTPException as e:
        if e.status_code == 404:
            logger.warning("template not found. use default template: %s", path)
            return default_text.strip()
        raise


def get_required_url_chunk_conf(chunk_config: dict[str, Any], prompt_type: str) -> ChunkConfig:
    for section_name in ("public_url", "upload"):
        section = chunk_config.get(section_name)
        if not isinstance(section, dict):
            continue
        conf = section.get(prompt_type)
        if not isinstance(conf, dict):
            continue
        try:
            max_chars = int(conf["max_chars"])
            max_items = int(conf["max_items"])
            overlap_items = int(conf.get("overlap_items", 0))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"openai_chunk.json: invalid {section_name}.{prompt_type} values: {e}")
        if max_chars <= 0 or max_items <= 0 or overlap_items < 0:
            raise HTTPException(status_code=500, detail=f"openai_chunk.json: invalid {section_name}.{prompt_type} values")
        return ChunkConfig(max_chars=max_chars, max_items=max_items, overlap_items=overlap_items)

    if prompt_type == "qa":
        return ChunkConfig(max_chars=12000, max_items=80, overlap_items=5)
    return ChunkConfig(max_chars=15000, max_items=120, overlap_items=5)


def validate_request(body: KnowledgeJobCreateRequest) -> None:
    if body.source_type not in (None, "", SOURCE_TYPE):
        raise HTTPException(status_code=400, detail=f"source_type must be '{SOURCE_TYPE}'")
    if not body.items:
        raise HTTPException(status_code=400, detail="items is empty")
    for idx, item in enumerate(body.items):
        if item.source_type not in (None, "", SOURCE_TYPE):
            raise HTTPException(status_code=400, detail=f"items[{idx}].source_type must be '{SOURCE_TYPE}'")
        if not (item.parent_source_id or "").strip():
            raise HTTPException(status_code=400, detail=f"items[{idx}].parent_source_id is required")


def fetch_url_page_rows(local_db_path: str, root_id: str) -> list[sqlite3.Row]:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT
                p.page_id,
                p.parent_page_id,
                p.page_url,
                p.depth,
                p.status,
                p.title,
                p.score,
                p.decision,
                p.is_usable,
                c.content_id,
                c.content_text
            FROM url_pages p
            JOIN url_page_contents c
              ON c.page_id = p.page_id
            WHERE p.root_id = ?
              AND COALESCE(c.content_text, '') <> ''
              AND p.status = 'done'
              AND COALESCE(p.is_usable, 1) = 1
            ORDER BY p.depth, p.created_at, p.page_id
            """,
            (root_id,),
        )
        return cur.fetchall()
    finally:
        conn.close()


def fetch_url_root_meta(local_db_path: str, root_id: str) -> sqlite3.Row | None:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT root_id, source_type, root_url, created_at
            FROM url_roots
            WHERE root_id = ?
            LIMIT 1
            """,
            (root_id,),
        )
        return cur.fetchone()
    finally:
        conn.close()


def build_page_records(page_row: sqlite3.Row, max_rows_per_page: int = 200) -> list[dict[str, Any]]:
    page_url = str(page_row["page_url"] or "")
    title = str(page_row["title"] or "").strip()
    depth = int(page_row["depth"] or 0)
    page_id = str(page_row["page_id"])
    content_text = clean_text(str(page_row["content_text"] or ""))
    if not content_text:
        return []

    raw_records = split_text_records(content_text.encode("utf-8"), max_rows=max_rows_per_page)
    if not raw_records:
        raw_records = split_paragraph_blocks(content_text, max_rows=max_rows_per_page)
    if not raw_records:
        raw_records = [build_record(record_type="paragraph", index=0, title=title[:80], text=content_text)]

    records: list[dict[str, Any]] = []
    for idx, rec in enumerate(raw_records, start=1):
        block_text = clean_text(str(rec.get("text") or ""))
        if not block_text:
            continue
        record_title = str(rec.get("title") or "").strip()
        record_type = str(rec.get("record_type") or "paragraph").strip() or "paragraph"
        decorated = (
            f"[PAGE_TITLE] {title}\n"
            f"[PAGE_URL] {page_url}\n"
            f"[DEPTH] {depth}\n"
            f"[BLOCK_TYPE] {record_type}\n"
            f"[BLOCK_TITLE] {record_title}\n"
            f"{block_text}"
        ).strip()
        records.append(
            {
                "page_id": page_id,
                "page_url": page_url,
                "record_no": idx,
                "text": decorated,
            }
        )
    return records


def fetch_url_job_item_meta(conn: sqlite3.Connection, job_item_id: str) -> sqlite3.Row:
    cur = conn.execute(
        """
        SELECT
            ji.job_item_id,
            ji.parent_source_id,
            ji.parent_key1,
            ji.parent_key2,
            ji.parent_label,
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


def fetch_url_content_rows(conn: sqlite3.Connection, job_item_id: str) -> list[sqlite3.Row]:
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


def build_url_prompt_texts(
    conn: sqlite3.Connection,
    job_item_id: str,
    template_text: str,
    chunk_conf: ChunkConfig,
) -> list[str]:
    item = fetch_url_job_item_meta(conn, job_item_id)
    rows = fetch_url_content_rows(conn, job_item_id)
    chunks = build_chunks(rows, chunk_conf, allowed_content_types={"row"})
    if not chunks:
        raise HTTPException(status_code=400, detail=f"knowledge_contents not found: {job_item_id}")

    prompt_texts: list[str] = []
    for chunk in chunks:
        prompt_texts.append(
            build_upload_prompt_text(
                job_item_id=job_item_id,
                prompt_template=template_text.strip(),
                chunk=chunk,
                file_name=item["parent_key2"] or item["parent_label"] or "public_url",
                file_type="public_url",
                source_label=item["parent_label"] or item["parent_key2"] or item["parent_source_id"],
                row_count=item["row_count"],
            )
        )
    return prompt_texts


def join_prompt_previews(prompt_texts: list[str]) -> str:
    blocks: list[str] = []
    for idx, prompt_text in enumerate(prompt_texts, start=1):
        blocks.append(f"===== CHUNK {idx} / {len(prompt_texts)} =====\n{prompt_text}")
    return "\n\n".join(blocks).strip()


def merge_qa_chunk_results(job_item_id: str, chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {"job_item_id": job_item_id, "qa_list": [], "chunk_count": len(chunk_results)}
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


def merge_plain_chunk_results(job_item_id: str, chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {"job_item_id": job_item_id, "plain_list": [], "chunk_count": len(chunk_results)}
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
    llm_result: dict[str, Any],
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
        question_raw = " ".join(str(qa.get("question") or "").split()).strip()
        answer_raw = " ".join(str(qa.get("answer") or "").split()).strip()
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
    return inserted_count


def insert_plain_items_from_llm_result(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
    source_id: str,
    llm_result: dict[str, Any],
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
        content_raw = " ".join(str(item.get("content") or "").split()).strip()
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
    return inserted_count


def insert_url_contents(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
    source_id: str,
    page_rows: list[sqlite3.Row],
) -> int:
    inserted_count = 0
    now = now_iso()
    sort_no = 1
    for page_row in page_rows:
        for rec in build_page_records(page_row):
            row_id = f"{rec['page_id']}:{rec['record_no']}"
            source_item_id = f"{rec['page_url']}#block={rec['record_no']}"
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
                    rec["text"],
                    sort_no,
                    now,
                    now,
                ),
            )
            inserted_count += 1
            sort_no += 1
    return inserted_count


def create_job_record(local_db_path: str, body: KnowledgeJobCreateRequest, selected_count: int) -> tuple[str, str]:
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
                body.source_name or "公開URL",
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
    item: KnowledgeTargetItem,
    requested_at: str,
    preview_only: bool,
) -> dict[str, Any]:
    source_id = item.parent_source_id or ""
    if not source_id:
        raise HTTPException(status_code=400, detail="parent_source_id is required")

    page_rows = fetch_url_page_rows(local_db_path, source_id)
    if not page_rows:
        raise HTTPException(status_code=400, detail=f"public_url content not found: {source_id}")

    root_meta = fetch_url_root_meta(local_db_path, source_id)
    if not root_meta:
        raise HTTPException(status_code=400, detail=f"url_roots not found: {source_id}")

    parent_key1 = item.parent_key1 or str(root_meta["source_type"] or "")
    parent_key2 = item.parent_key2 or str(root_meta["root_url"] or "")
    parent_label = item.parent_label or parent_key2 or parent_key1 or source_id

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
                source_id,
                parent_key1,
                parent_key2,
                parent_label,
                item.row_count,
                "done" if preview_only else "new",
                requested_at,
            ),
        )

        inserted_rows = insert_url_contents(conn, job_id, job_item_id, source_id, page_rows)
        if inserted_rows <= 0:
            raise HTTPException(status_code=400, detail=f"public_url text not found: {source_id}")

        conn.execute(
            """
            UPDATE knowledge_job_items
            SET row_count = ?
            WHERE job_item_id = ?
            """,
            (inserted_rows, job_item_id),
        )
        conn.commit()
        return {"job_item_id": job_item_id, "source_id": source_id, "parent_label": parent_label}
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
        meta = fetch_url_job_item_meta(conn, job_item_id)
        qa_prompt_texts = build_url_prompt_texts(conn, job_item_id, qa_template_text, qa_chunk_conf)
        plain_prompt_texts = build_url_prompt_texts(conn, job_item_id, plain_template_text, plain_chunk_conf)
        return {
            "job_item_id": job_item_id,
            "source_id": meta["parent_source_id"],
            "parent_source_id": meta["parent_source_id"],
            "parent_label": meta["parent_label"],
            "qa_prompt_texts": qa_prompt_texts,
            "plain_prompt_texts": plain_prompt_texts,
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


def update_job_summary(
    local_db_path: str,
    job_id: str,
    requested_at: str,
    status: str,
    total_qa_count: int,
    total_plain_count: int,
    total_error_count: int,
    error_message: str | None,
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
                    WHEN ? IN ('running', 'done', 'error') THEN COALESCE(started_at, ?)
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


def finalize_job_item_success(
    local_db_path: str,
    job_id: str,
    job_item_id: str,
    source_id: str,
    qa_llm_result: dict[str, Any],
    plain_llm_result: dict[str, Any],
    preview_only: bool,
) -> tuple[int, int]:
    conn = open_user_db(local_db_path)
    try:
        conn.execute("BEGIN")
        qa_count = 0
        plain_count = 0
        if not preview_only:
            qa_count = insert_qa_items_from_llm_result(conn, job_id, job_item_id, source_id, qa_llm_result)
            plain_count = insert_plain_items_from_llm_result(conn, job_id, job_item_id, source_id, plain_llm_result)
        conn.execute(
            """
            UPDATE knowledge_job_items
            SET status = 'done',
                knowledge_count = ?,
                error_message = NULL,
                started_at = COALESCE(started_at, ?),
                finished_at = ?
            WHERE job_item_id = ?
            """,
            (qa_count + plain_count, now_iso(), now_iso(), job_item_id),
        )
        conn.commit()
        return qa_count, plain_count
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def finalize_job_item_error(local_db_path: str, job_item_id: str, error_message: str) -> None:
    conn = open_user_db(local_db_path)
    try:
        conn.execute("BEGIN")
        conn.execute(
            """
            UPDATE knowledge_job_items
            SET status = 'error',
                error_message = ?,
                started_at = COALESCE(started_at, ?),
                finished_at = ?
            WHERE job_item_id = ?
            """,
            (error_message, now_iso(), now_iso(), job_item_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def process_url_job_item(
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
        progress_callback(total_qa_chunks=total_qa_chunks, processed_qa_chunks=0, total_plain_chunks=total_plain_chunks, processed_plain_chunks=0, message="chunk prepared")

    if preview_only:
        qa_llm_result = {"job_item_id": job_item_id, "qa_list": []}
        plain_llm_result = {"job_item_id": job_item_id, "plain_list": []}
        qa_llm_result_for_debug = None
        plain_llm_result_for_debug = None
    else:
        qa_chunk_results: list[dict[str, Any]] = []
        for idx, prompt_text in enumerate(qa_prompt_texts, start=1):
            result_list = run_chunked_llm_json([prompt_text], "LLM PUBLIC URL QA")
            if not result_list:
                raise Exception("empty qa chunk result")
            qa_chunk_results.append(result_list[0])
            if progress_callback:
                progress_callback(total_qa_chunks=total_qa_chunks, processed_qa_chunks=idx, total_plain_chunks=total_plain_chunks, processed_plain_chunks=0, message=f"qa chunk {idx}/{total_qa_chunks}")

        qa_llm_result = merge_qa_chunk_results(job_item_id, qa_chunk_results)
        qa_llm_result_for_debug = {"chunk_results": qa_chunk_results, "merged_result": qa_llm_result}

        plain_chunk_results: list[dict[str, Any]] = []
        for idx, prompt_text in enumerate(plain_prompt_texts, start=1):
            result_list = run_chunked_llm_json([prompt_text], "LLM PUBLIC URL PLAIN")
            if not result_list:
                raise Exception("empty plain chunk result")
            plain_chunk_results.append(result_list[0])
            if progress_callback:
                progress_callback(total_qa_chunks=total_qa_chunks, processed_qa_chunks=total_qa_chunks, total_plain_chunks=total_plain_chunks, processed_plain_chunks=idx, message=f"plain chunk {idx}/{total_plain_chunks}")

        plain_llm_result = merge_plain_chunk_results(job_item_id, plain_chunk_results)
        plain_llm_result_for_debug = {"chunk_results": plain_chunk_results, "merged_result": plain_llm_result}

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
        "qa_count": qa_count,
        "plain_count": plain_count,
        "status": "done",
        "qa_prompt_texts": qa_prompt_texts,
        "plain_prompt_texts": plain_prompt_texts,
        "qa_debug": qa_llm_result_for_debug,
        "plain_debug": plain_llm_result_for_debug,
    }


def run_url_job_background(uid: str, job_id: str) -> None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        set_job_error(bucket, uid, job_id, f"ank.db not found: {db_gcs_path}", phase="extract_knowledge", message="ank.db not found")
        return

    local_db_path = local_user_db_path(uid)

    try:
        replace_local_db_from_blob(db_blob, local_db_path)

        job_row = fetch_job_row(local_db_path, job_id)
        if not job_row:
            set_job_error(bucket, uid, job_id, f"knowledge_jobs not found: {job_id}", phase="extract_knowledge", message="knowledge_jobs not found")
            return
        if job_row["source_type"] != SOURCE_TYPE:
            set_job_error(bucket, uid, job_id, "invalid source_type", phase="extract_knowledge", message="invalid source_type")
            return

        qa_template_text = load_template_text_or_default(URL_QA_PROMPT_PATH, DEFAULT_URL_QA_PROMPT)
        plain_template_text = load_template_text_or_default(URL_PLAIN_PROMPT_PATH, DEFAULT_URL_PLAIN_PROMPT)
        chunk_config = load_chunk_config(BUCKET_NAME, OPENAI_CHUNK_CONFIG_PATH)
        qa_chunk_conf = get_required_url_chunk_conf(chunk_config, "qa")
        plain_chunk_conf = get_required_url_chunk_conf(chunk_config, "plain")

        requested_at = job_row["requested_at"] or now_iso()
        total_qa_count = int(job_row["qa_count"] or 0)
        total_plain_count = int(job_row["plain_count"] or 0)
        total_error_count = int(job_row["error_count"] or 0)

        conn = open_user_db(local_db_path)
        try:
            cur = conn.execute(
                """
                SELECT parent_source_id, parent_label, row_count
                FROM knowledge_job_items
                WHERE job_id = ?
                ORDER BY created_at, job_item_id
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
                result = process_url_job_item(
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

                update_job_summary(local_db_path, job_id, requested_at, "running", total_qa_count, total_plain_count, total_error_count, None)
                upload_local_db(db_blob, local_db_path)
                set_job_progress(bucket, uid, phase="extract_knowledge", message="item done", knowledge_count=total_qa_count + total_plain_count)
            except Exception as e:
                logger.exception("process_url_job_item failed: job_item_id=%s", current_job_item_id)
                try:
                    finalize_job_item_error(local_db_path, current_job_item_id, str(e))
                except Exception:
                    logger.exception("failed to update error state: job_item_id=%s", current_job_item_id)

                total_error_count += 1
                try:
                    update_job_summary(local_db_path, job_id, requested_at, "error", total_qa_count, total_plain_count, total_error_count, str(e))
                    upload_local_db(db_blob, local_db_path)
                except Exception:
                    logger.exception("failed to update job summary error: job_id=%s", job_id)

                set_job_error(bucket, uid, job_id, str(e), phase="extract_knowledge", message="item failed")
                return

        update_job_summary(local_db_path, job_id, requested_at, "done", total_qa_count, total_plain_count, total_error_count, None)
        upload_local_db(db_blob, local_db_path)
        set_job_done(bucket, uid, job_id, phase="extract_knowledge", message="completed", knowledge_count=total_qa_count + total_plain_count)
    except Exception as e:
        logger.exception("run_url_job_background failed: job_id=%s", job_id)
        set_job_error(bucket, uid, job_id, str(e), phase="extract_knowledge", message="background failed")


@router.post("/jobs", response_model=KnowledgeJobCreateResponse)
def create_url_job(
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
        raise HTTPException(status_code=400, detail=f"ank.db not found: {db_gcs_path}")

    local_db_path = local_user_db_path(uid)
    replace_local_db_from_blob(db_blob, local_db_path)

    selected_count = len(body.items)
    job_id, requested_at = create_job_record(local_db_path, body, selected_count)

    chunk_config = load_chunk_config(BUCKET_NAME, OPENAI_CHUNK_CONFIG_PATH)
    qa_chunk_conf = get_required_url_chunk_conf(chunk_config, "qa")
    plain_chunk_conf = get_required_url_chunk_conf(chunk_config, "plain")
    qa_template_text = load_template_text_or_default(URL_QA_PROMPT_PATH, DEFAULT_URL_QA_PROMPT)
    plain_template_text = load_template_text_or_default(URL_PLAIN_PROMPT_PATH, DEFAULT_URL_PLAIN_PROMPT)

    created_item_count = 0
    prompt_previews: list[PromptPreviewItem] = []
    debug_items: list[KnowledgeDebugItem] = []

    for item in body.items:
        prepared = prepare_job_item(local_db_path, job_id, item, requested_at, body.preview_only)
        created_item_count += 1

        if body.preview_only:
            result = process_url_job_item(
                local_db_path=local_db_path,
                job_id=job_id,
                job_item_id=prepared["job_item_id"],
                qa_chunk_conf=qa_chunk_conf,
                plain_chunk_conf=plain_chunk_conf,
                qa_template_text=qa_template_text,
                plain_template_text=plain_template_text,
                preview_only=True,
            )
            prompt_previews.append(
                PromptPreviewItem(
                    job_item_id=prepared["job_item_id"],
                    parent_source_id=prepared["source_id"],
                    parent_label=prepared.get("parent_label"),
                    prompt_type="qa",
                    prompt_text=join_prompt_previews(result["qa_prompt_texts"]),
                )
            )
            prompt_previews.append(
                PromptPreviewItem(
                    job_item_id=prepared["job_item_id"],
                    parent_source_id=prepared["source_id"],
                    parent_label=prepared.get("parent_label"),
                    prompt_type="plain",
                    prompt_text=join_prompt_previews(result["plain_prompt_texts"]),
                )
            )
            debug_items.append(
                KnowledgeDebugItem(
                    job_item_id=prepared["job_item_id"],
                    parent_label=prepared.get("parent_label"),
                    status="done",
                    qa_count=0,
                    plain_count=0,
                    llm_result={"qa_debug": result["qa_debug"], "plain_debug": result["plain_debug"]},
                )
            )

    upload_local_db(db_blob, local_db_path)

    return KnowledgeJobCreateResponse(
        job_id=job_id,
        selected_count=selected_count,
        created_item_count=created_item_count,
        status="done" if body.preview_only else "new",
        prompt_previews=prompt_previews,
        debug_items=debug_items,
    )


@router.post("/run")
def run_url_job(
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
        raise HTTPException(status_code=400, detail=f"ank.db not found: {db_gcs_path}")

    local_db_path = local_user_db_path(uid)
    replace_local_db_from_blob(db_blob, local_db_path)

    job_row = fetch_job_row(local_db_path, body.job_id)
    if not job_row:
        raise HTTPException(status_code=404, detail=f"knowledge_jobs not found: {body.job_id}")
    if job_row["source_type"] != SOURCE_TYPE:
        raise HTTPException(status_code=400, detail="invalid source_type")

    background_tasks.add_task(run_url_job_background, uid, body.job_id)
    return {"job_id": body.job_id, "status": "running", "message": "background started"}


@router.get("/status", response_model=KnowledgeJobStatusResponse)
def get_url_job_status(
    job_id: str = Query(...),
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    status_payload = try_read_status_payload(bucket, uid)
    if status_payload and status_payload.get("job_id") == job_id:
        return KnowledgeJobStatusResponse(**status_payload)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(status_code=400, detail=f"ank.db not found: {db_gcs_path}")

    local_db_path = local_user_db_path(uid)
    replace_local_db_from_blob(db_blob, local_db_path)

    job_row = fetch_job_row(local_db_path, job_id)
    if not job_row:
        raise HTTPException(status_code=404, detail=f"knowledge_jobs not found: {job_id}")

    return KnowledgeJobStatusResponse(
        job_id=job_id,
        source_type=job_row["source_type"],
        status=job_row["status"] or "idle",
        error_message=job_row["error_message"],
        started_at=job_row["started_at"],
        finished_at=job_row["finished_at"],
        knowledge_count=int(job_row["qa_count"] or 0) + int(job_row["plain_count"] or 0),
    )
