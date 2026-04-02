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

SOURCE_TYPE = "kokkai"

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
KOKKAI_QA_PROMPT_PATH = "template/kokkai_qa_prompt.txt"
KOKKAI_PLAIN_PROMPT_PATH = "template/kokkai_plain_prompt.txt"
OPENAI_CHUNK_CONFIG_PATH = "template/openai_chunk.json"

DEFAULT_KOKKAI_QA_PROMPT = ""
DEFAULT_KOKKAI_PLAIN_PROMPT = ""


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    merged = "\n".join(line for line in lines if line)
    return merged.strip()


def load_template_text_with_default(template_path: str, default_template: str) -> str:
    text = load_template_text(BUCKET_NAME, template_path)
    return text if text else default_template


def get_required_kokkai_chunk_conf(chunk_config: dict[str, Any], prompt_type: str) -> dict[str, Any]:
    kokkai_conf = chunk_config.get("kokkai", {})
    return kokkai_conf.get(prompt_type, {})


def fetch_kokkai_source_rows(local_db_path: str, issue_id: str) -> list[sqlite3.Row]:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT
                issue_id,
                speech_id,
                speech_order,
                speaker,
                speech
            FROM kokkai_document_rows
            WHERE issue_id = ?
            ORDER BY speech_order, speech_id
            """,
            (issue_id,),
        )
        return cur.fetchall()
    finally:
        conn.close()


def fetch_kokkai_job_item_meta(conn: sqlite3.Connection, job_item_id: str) -> sqlite3.Row | None:
    cur = conn.execute(
        """
        SELECT
            ji.job_item_id,
            ji.parent_source_id,
            ji.parent_label,
            ji.parent_key1,
            ji.parent_key2,
            ji.row_count,
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
    return cur.fetchone()


def insert_kokkai_contents(conn: sqlite3.Connection, job_id: str, job_item_id: str, source_rows: list[sqlite3.Row]) -> int:
    conn.execute("DELETE FROM knowledge_contents WHERE job_item_id = ?", (job_item_id,))
    count = 0
    for idx, row in enumerate(source_rows, start=1):
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
                row["issue_id"],
                row["speech_id"],
                row["speech_id"],
                "row",
                normalize_text(row["speech"]),
                idx,
                now_iso(),
                now_iso(),
            ),
        )
        count += 1
    return count


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
    prompt_type: str,
) -> list[str]:
    rows = fetch_kokkai_content_rows(conn, job_item_id)
    if not rows:
        return []

    # 第一段階の仮実装: 1件に全部まとめる
    content_lines = []
    for row in rows:
        content_lines.append(row["content_text"] or "")

    header = f"[SOURCE_TYPE={SOURCE_TYPE}][JOB_ITEM_ID={job_item_id}][PROMPT_TYPE={prompt_type}]"
    body = "\n\n".join(x for x in content_lines if x).strip()
    if not body:
        return []
    return [f"{header}\n\n{body}"]


def build_kokkai_chunk_rows(
    conn: sqlite3.Connection,
    job_item_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    qa_prompts = build_kokkai_prompt_texts(conn, job_item_id, "qa")
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

    plain_prompts = build_kokkai_prompt_texts(conn, job_item_id, "plain")
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
