from __future__ import annotations

import os
import sqlite3
from typing import Any

from .knowledge_generate_common import (
    load_chunk_config,
    load_template_text,
    now_iso,
    open_user_db,
)

SOURCE_TYPE = "opendata"

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
OPENDATA_QA_PROMPT_PATH = "template/opendata_qa_prompt.txt"
OPENDATA_PLAIN_PROMPT_PATH = "template/opendata_plain_prompt.txt"
OPENAI_CHUNK_CONFIG_PATH = "template/openai_chunk.json"

from app.core.common import normalize_text

def get_required_opendata_chunk_conf(chunk_config: dict[str, Any], prompt_type: str) -> dict[str, Any]:
    opendata_conf = chunk_config.get("opendata", {})
    return opendata_conf.get(prompt_type, {})


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


def fetch_opendata_job_item_meta(conn: sqlite3.Connection, job_item_id: str) -> sqlite3.Row | None:
    cur = conn.execute(
        """
        SELECT
            job_item_id,
            parent_source_id,
            parent_label,
            parent_key1,
            parent_key2,
            row_count
        FROM knowledge_job_items
        WHERE job_item_id = ?
        LIMIT 1
        """,
        (job_item_id,),
    )
    return cur.fetchone()


def insert_opendata_contents(conn: sqlite3.Connection, job_id: str, job_item_id: str, source_rows: list[sqlite3.Row]) -> int:
    conn.execute("DELETE FROM knowledge_contents WHERE job_item_id = ?", (job_item_id,))
    count = 0
    for idx, row in enumerate(source_rows, start=1):
        text = normalize_text(row["logical_name"]) or normalize_text(row["original_name"])
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
                row["source_id"],
                row["file_id"],
                row["file_id"],
                "row",
                text,
                idx,
                now_iso(),
                now_iso(),
            ),
        )
        count += 1
    return count


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


def build_opendata_prompt_texts(conn: sqlite3.Connection, job_item_id: str, prompt_type: str) -> list[str]:
    rows = fetch_opendata_content_rows(conn, job_item_id)
    if not rows:
        return []

    content_lines = [row["content_text"] or "" for row in rows]
    header = f"[SOURCE_TYPE={SOURCE_TYPE}][JOB_ITEM_ID={job_item_id}][PROMPT_TYPE={prompt_type}]"
    body = "\n\n".join(x for x in content_lines if x).strip()
    if not body:
        return []
    return [f"{header}\n\n{body}"]


def build_opendata_chunk_rows(conn: sqlite3.Connection, job_item_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    qa_prompts = build_opendata_prompt_texts(conn, job_item_id, "qa")
    for idx, prompt in enumerate(qa_prompts, start=1):
        rows.append(
            {
                "chunk_no": idx,
                "prompt_type": "qa",
                "prompt": prompt,
                "row_count": 0,
                "status": "new",
            }
        )

    plain_prompts = build_opendata_prompt_texts(conn, job_item_id, "plain")
    base_no = len(rows)
    for idx, prompt in enumerate(plain_prompts, start=1):
        rows.append(
            {
                "chunk_no": base_no + idx,
                "prompt_type": "plain",
                "prompt": prompt,
                "row_count": 0,
                "status": "new",
            }
        )

    return rows
