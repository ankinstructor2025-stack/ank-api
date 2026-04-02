from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from .knowledge_generate_common import (
    load_chunk_config,
    load_template_text,
    now_iso,
    open_user_db,
)

SOURCE_TYPE = "upload"

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
UPLOAD_QA_PROMPT_PATH = "template/upload_qa_prompt.txt"
UPLOAD_PLAIN_PROMPT_PATH = "template/upload_plain_prompt.txt"
OPENAI_CHUNK_CONFIG_PATH = "template/openai_chunk.json"

from app.core.common import normalize_text

def normalize_upload_chunk_ext(ext: str | None) -> str:
    ext_norm = normalize_text(ext).lower()
    if ext_norm in {"txt", "text", "md", "markdown", "log", "tsv", "html", "htm"}:
        return "txt"
    return ext_norm


def get_required_upload_chunk_conf(chunk_config: dict[str, Any], ext: str, prompt_type: str) -> dict[str, Any]:
    upload_conf = chunk_config.get("file_upload", {})
    ext_conf = upload_conf.get(normalize_upload_chunk_ext(ext), {})
    return ext_conf.get(prompt_type, {})


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

    text = normalize_text(value)
    if not text:
        return []

    return [f"{prefix}: {text}" if prefix else text]


def extract_structured_text(content: Any) -> str | None:
    if content is None:
        return None

    parsed = content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except Exception:
            return normalize_text(content) or None

    if isinstance(parsed, (dict, list)):
        lines = flatten_json_like(parsed)
        merged = "\n".join(lines).strip()
        return merged or None

    return normalize_text(parsed) or None


def fetch_upload_file_row(local_db_path: str, file_id: str) -> sqlite3.Row | None:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT
                file_id,
                file_name,
                ext,
                created_at
            FROM upload_files
            WHERE file_id = ?
            LIMIT 1
            """,
            (file_id,),
        )
        return cur.fetchone()
    finally:
        conn.close()


def fetch_upload_job_item_meta(conn: sqlite3.Connection, job_item_id: str) -> sqlite3.Row | None:
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


def insert_upload_contents(conn: sqlite3.Connection, job_id: str, job_item_id: str, file_row: sqlite3.Row | None) -> int:
    conn.execute("DELETE FROM knowledge_contents WHERE job_item_id = ?", (job_item_id,))
    if not file_row:
        return 0

    text = normalize_text(file_row["file_name"])
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
            file_row["file_id"],
            file_row["file_id"],
            file_row["file_id"],
            "row",
            text,
            1,
            now_iso(),
            now_iso(),
        ),
    )
    return 1


def fetch_upload_content_rows(conn: sqlite3.Connection, job_item_id: str) -> list[sqlite3.Row]:
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


def build_upload_prompt_texts(conn: sqlite3.Connection, job_item_id: str, prompt_type: str) -> list[str]:
    rows = fetch_upload_content_rows(conn, job_item_id)
    if not rows:
        return []

    content_lines = [row["content_text"] or "" for row in rows]
    header = f"[SOURCE_TYPE={SOURCE_TYPE}][JOB_ITEM_ID={job_item_id}][PROMPT_TYPE={prompt_type}]"
    body = "\n\n".join(x for x in content_lines if x).strip()
    if not body:
        return []
    return [f"{header}\n\n{body}"]


def build_upload_chunk_rows(conn: sqlite3.Connection, job_item_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    qa_prompts = build_upload_prompt_texts(conn, job_item_id, "qa")
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

    plain_prompts = build_upload_prompt_texts(conn, job_item_id, "plain")
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
