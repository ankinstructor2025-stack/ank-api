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

SOURCE_TYPE = "public_url"

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
URL_QA_PROMPT_PATH = "template/public_url_qa_prompt.txt"
URL_PLAIN_PROMPT_PATH = "template/public_url_plain_prompt.txt"
OPENAI_CHUNK_CONFIG_PATH = "template/openai_chunk.json"

DEFAULT_URL_QA_PROMPT = ""
DEFAULT_URL_PLAIN_PROMPT = ""

from app.core.common import normalize_text

def load_template_text_or_default(path: str, default_text: str) -> str:
    text = load_template_text(BUCKET_NAME, path)
    return text if text else default_text


def get_required_url_chunk_conf(chunk_config: dict[str, Any], prompt_type: str) -> dict[str, Any]:
    for section_name in ("public_url", "upload"):
        section = chunk_config.get(section_name, {})
        conf = section.get(prompt_type)
        if isinstance(conf, dict):
            return conf
    return {}


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


def fetch_url_job_item_meta(conn: sqlite3.Connection, job_item_id: str) -> sqlite3.Row | None:
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


def insert_url_contents(conn: sqlite3.Connection, job_id: str, job_item_id: str, page_rows: list[sqlite3.Row]) -> int:
    conn.execute("DELETE FROM knowledge_contents WHERE job_item_id = ?", (job_item_id,))
    count = 0
    for idx, row in enumerate(page_rows, start=1):
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
                row["page_id"],
                row["page_id"],
                row["page_id"],
                "row",
                normalize_text(row["content_text"]),
                idx,
                now_iso(),
                now_iso(),
            ),
        )
        count += 1
    return count


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


def build_url_prompt_texts(conn: sqlite3.Connection, job_item_id: str, prompt_type: str) -> list[str]:
    rows = fetch_url_content_rows(conn, job_item_id)
    if not rows:
        return []

    content_lines = [row["content_text"] or "" for row in rows]
    header = f"[SOURCE_TYPE={SOURCE_TYPE}][JOB_ITEM_ID={job_item_id}][PROMPT_TYPE={prompt_type}]"
    body = "\n\n".join(x for x in content_lines if x).strip()
    if not body:
        return []
    return [f"{header}\n\n{body}"]


def build_public_url_chunk_rows(conn: sqlite3.Connection, job_item_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    qa_prompts = build_url_prompt_texts(conn, job_item_id, "qa")
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

    plain_prompts = build_url_prompt_texts(conn, job_item_id, "plain")
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
