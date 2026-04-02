from __future__ import annotations

import os
import sqlite3
from typing import Any

from .knowledge_generate_common import (
    load_chunk_config,
    load_template_text,
    PROMPT_TEMPLATE_PATHS,
)

from app.core.common import normalize_text

SOURCE_TYPE = "opendata"

# =========================
# chunk config取得
# =========================
def get_required_opendata_chunk_conf(
    chunk_config: dict[str, Any],
    prompt_type: str
) -> dict[str, Any]:
    opendata_conf = chunk_config.get("opendata", {})
    return opendata_conf.get(prompt_type, {})


# =========================
# content取得
# =========================
def fetch_opendata_content_rows(
    conn: sqlite3.Connection,
    job_item_id: str
) -> list[sqlite3.Row]:

    cur = conn.execute(
        """
        SELECT
            content_text,
            sort_no
        FROM knowledge_contents
        WHERE job_item_id = ?
        ORDER BY sort_no
        """,
        (job_item_id,),
    )
    return cur.fetchall()


# =========================
# chunk生成
# =========================
def build_opendata_chunk_rows(
    conn: sqlite3.Connection,
    job_item_id: str
) -> list[dict[str, Any]]:

    # ===== chunk設定読み込み =====
    chunk_config = load_chunk_config()

    qa_conf = get_required_opendata_chunk_conf(chunk_config, "qa")
    plain_conf = get_required_opendata_chunk_conf(chunk_config, "plain")

    qa_size = qa_conf.get("chunk_size", 1)
    qa_overlap = qa_conf.get("overlap", 0)

    plain_size = plain_conf.get("chunk_size", 1)
    plain_overlap = plain_conf.get("overlap", 0)

    # ===== テンプレート読み込み（ここ重要）=====
    qa_template = load_template_text(
        PROMPT_TEMPLATE_PATHS["opendata"]["qa"]
    )

    plain_template = load_template_text(
        PROMPT_TEMPLATE_PATHS["opendata"]["plain"]
    )

    if not qa_template:
        raise RuntimeError("QA prompt template is empty")

    if not plain_template:
        raise RuntimeError("PLAIN prompt template is empty")

    # ===== データ取得 =====
    rows = fetch_opendata_content_rows(conn, job_item_id)
    if not rows:
        return []

    contents = [r["content_text"] for r in rows if r["content_text"]]

    chunks: list[dict[str, Any]] = []
    chunk_no = 0

    # =========================
    # chunk分割関数
    # =========================
    def create_chunks(data: list[str], size: int, overlap: int):
        i = 0
        result = []
        while i < len(data):
            block = data[i:i + size]
            if not block:
                break
            result.append(block)
            i += max(1, size - overlap)
        return result

    # =========================
    # QA
    # =========================
    qa_blocks = create_chunks(contents, qa_size, qa_overlap)

    for block in qa_blocks:
        body = "\n\n".join(block)
        prompt = f"{qa_template}\n\n---\n\n{body}"

        chunks.append({
            "chunk_no": chunk_no,
            "prompt_type": "qa",
            "prompt": prompt,
            "status": "new",
        })
        chunk_no += 1

    # =========================
    # PLAIN
    # =========================
    plain_blocks = create_chunks(contents, plain_size, plain_overlap)

    for block in plain_blocks:
        body = "\n\n".join(block)
        prompt = f"{plain_template}\n\n---\n\n{body}"

        chunks.append({
            "chunk_no": chunk_no,
            "prompt_type": "plain",
            "prompt": prompt,
            "status": "new",
        })
        chunk_no += 1

    return chunks
