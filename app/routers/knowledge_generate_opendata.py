from __future__ import annotations

import sqlite3
from typing import Any

from .knowledge_generate_common import (
    load_chunk_config,
    load_template_text,
    PROMPT_TEMPLATE_PATHS,
)
from .openai_chunking import ChunkConfig, build_chunks
from .openai_prompt_builder import build_opendata_prompt_text

SOURCE_TYPE = "opendata"


# =========================
# chunk config取得（エラー前提）
# =========================
def get_required_opendata_chunk_conf(
    chunk_config: dict[str, Any],
    prompt_type: str,
) -> dict[str, Any]:

    if not chunk_config:
        raise RuntimeError("openai_chunk.json is empty or not loaded")

    opendata_conf = chunk_config.get("opendata")
    if not opendata_conf:
        raise RuntimeError("openai_chunk.json: 'opendata' section missing")

    conf = opendata_conf.get(prompt_type)
    if not conf:
        raise RuntimeError(f"openai_chunk.json: opendata.{prompt_type} missing")

    return conf


def to_chunk_config(conf: dict[str, Any]) -> ChunkConfig:
    required_keys = ["max_chars", "max_items", "overlap_items"]

    for key in required_keys:
        if key not in conf:
            raise RuntimeError(f"openai_chunk.json missing key: {key}")

    return ChunkConfig(
        max_chars=int(conf["max_chars"]),
        max_items=int(conf["max_items"]),
        overlap_items=int(conf["overlap_items"]),
    )


# =========================
# content取得
# =========================
def fetch_opendata_content_rows(
    conn: sqlite3.Connection,
    job_item_id: str,
) -> list[sqlite3.Row]:
    cur = conn.execute(
        """
        SELECT
            row_id,
            source_item_id,
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


# =========================
# chunk生成
# =========================
def build_opendata_chunk_rows(
    conn: sqlite3.Connection,
    job_item_id: str,
) -> list[dict[str, Any]]:

    chunk_config = load_chunk_config()
    if not chunk_config:
        raise RuntimeError("openai_chunk.json could not be loaded from GCS")

    qa_conf = get_required_opendata_chunk_conf(chunk_config, "qa")
    plain_conf = get_required_opendata_chunk_conf(chunk_config, "plain")

    qa_chunk_config = to_chunk_config(qa_conf)
    plain_chunk_config = to_chunk_config(plain_conf)

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

    rows = fetch_opendata_content_rows(conn, job_item_id)
    if not rows:
        raise RuntimeError("knowledge_contents is empty for job_item_id")

    row_count = len(rows)
    chunk_rows: list[dict[str, Any]] = []

    # =========================
    # QA chunk
    # =========================
    qa_chunks = build_chunks(rows, qa_chunk_config)

    for chunk in qa_chunks:
        prompt = build_opendata_prompt_text(
            job_item_id=job_item_id,
            prompt_template=qa_template,
            chunk=chunk,
            parent_source_id=job_item_id,
            parent_key1=SOURCE_TYPE,
            parent_key2="json",
            parent_label=f"{SOURCE_TYPE}:{job_item_id}",
            row_count=row_count,
        )

        chunk_rows.append(
            {
                "job_item_id": job_item_id,
                "source_type": SOURCE_TYPE,
                "chunk_no": chunk.chunk_no,
                "prompt_type": "qa",
                "prompt": prompt,
                "status": "new",
            }
        )

    # =========================
    # PLAIN chunk
    # =========================
    plain_chunks = build_chunks(rows, plain_chunk_config)

    for chunk in plain_chunks:
        prompt = build_opendata_prompt_text(
            job_item_id=job_item_id,
            prompt_template=plain_template,
            chunk=chunk,
            parent_source_id=job_item_id,
            parent_key1=SOURCE_TYPE,
            parent_key2="json",
            parent_label=f"{SOURCE_TYPE}:{job_item_id}",
            row_count=row_count,
        )

        chunk_rows.append(
            {
                "job_item_id": job_item_id,
                "source_type": SOURCE_TYPE,
                "chunk_no": chunk.chunk_no,
                "prompt_type": "plain",
                "prompt": prompt,
                "status": "new",
            }
        )

    if not chunk_rows:
        raise RuntimeError("chunk generation result is empty")

    return chunk_rows


# =========================
# CHUNK登録
# =========================
def insert_opendata_chunks(
    conn: sqlite3.Connection,
    job_item_id: str,
) -> int:

    chunk_rows = build_opendata_chunk_rows(conn, job_item_id)

    conn.execute(
        """
        DELETE FROM CHUNK
        WHERE job_item_id = ?
          AND source_type = ?
        """,
        (job_item_id, SOURCE_TYPE),
    )

    for row in chunk_rows:
        conn.execute(
            """
            INSERT INTO CHUNK (
                job_item_id,
                source_type,
                chunk_no,
                prompt_type,
                prompt,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["job_item_id"],
                row["source_type"],
                row["chunk_no"],
                row["prompt_type"],
                row["prompt"],
                row["status"],
            ),
        )

    conn.commit()
    return len(chunk_rows)
