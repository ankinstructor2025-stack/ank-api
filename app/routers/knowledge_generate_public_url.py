from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from .knowledge_generate_common import (
    PROMPT_TEMPLATE_PATHS,
    load_chunk_config,
    load_template_text,
    open_user_db,
)
from .openai_chunking import ChunkConfig, build_chunks, render_chunk_text


SOURCE_TYPE = "public_url"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    merged = "\n".join(line for line in lines if line)
    return merged.strip()


def get_required_url_chunk_conf(
    chunk_config: dict[str, Any],
    prompt_type: str,
) -> dict[str, Any]:
    if not chunk_config:
        raise RuntimeError("openai_chunk.json is empty or not loaded")

    public_url_conf = chunk_config.get("public_url")
    if not public_url_conf:
        raise RuntimeError("openai_chunk.json: 'public_url' section missing")

    conf = public_url_conf.get(prompt_type)
    if not conf:
        raise RuntimeError(f"openai_chunk.json: public_url.{prompt_type} missing")

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


def prompt_reserved_chars(prompt_template: str) -> int:
    template = (prompt_template or "").strip()
    if not template:
        raise RuntimeError("prompt template is empty")
    return len(template) + 2


def to_prompt_safe_chunk_config(
    base_config: ChunkConfig,
    prompt_template: str,
) -> ChunkConfig:
    reserved = prompt_reserved_chars(prompt_template)
    usable_chars = int(base_config.max_chars) - reserved

    if usable_chars <= 0:
        raise RuntimeError(
            f"max_chars is too small for prompt template. "
            f"max_chars={base_config.max_chars}, reserved={reserved}"
        )

    return ChunkConfig(
        max_chars=usable_chars,
        max_items=int(base_config.max_items),
        overlap_items=int(base_config.overlap_items),
    )


def fetch_url_job_item_meta(
    task_conn: sqlite3.Connection,
    job_item_id: str,
) -> sqlite3.Row | None:
    cur = task_conn.execute(
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


def fetch_url_root_meta(
    local_db_path: str,
    root_id: str,
) -> sqlite3.Row | None:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT
                root_id,
                source_type,
                root_url,
                created_at
            FROM url_roots
            WHERE root_id = ?
            LIMIT 1
            """,
            (root_id,),
        )
        return cur.fetchone()
    finally:
        conn.close()


def fetch_url_page_rows(
    local_db_path: str,
    root_id: str,
) -> list[sqlite3.Row]:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT
                p.page_id,
                p.root_id,
                p.parent_page_id,
                p.page_url,
                p.depth,
                p.status,
                p.child_count,
                p.title,
                p.content_type,
                p.http_status,
                p.fetched_at,
                p.text_length,
                p.link_count,
                p.short_line_ratio,
                p.score,
                p.decision,
                p.decision_reason,
                p.is_usable,
                p.created_at,
                c.content_id,
                c.content_text,
                c.created_at AS content_created_at
            FROM url_pages p
            JOIN url_page_contents c
              ON c.page_id = p.page_id
            WHERE p.root_id = ?
              AND COALESCE(c.content_text, '') <> ''
              AND COALESCE(p.is_usable, 0) = 1
            ORDER BY p.depth, p.created_at, p.page_id, c.content_id
            """,
            (root_id,),
        )
        return cur.fetchall()
    finally:
        conn.close()


def page_row_to_chunk_text(row: sqlite3.Row) -> str:
    title = normalize_text(row["title"])
    page_url = normalize_text(row["page_url"])
    content_text = normalize_text(row["content_text"])

    if not content_text:
        return ""

    lines: list[str] = []

    if title:
        lines.append(f"タイトル: {title}")

    if page_url:
        lines.append(f"URL: {page_url}")

    lines.append("本文:")
    lines.append(content_text)

    return "\n".join(lines).strip()


def build_public_url_chunk_inputs(
    task_conn: sqlite3.Connection,
    local_db_path: str,
    job_item_id: str,
) -> list[dict[str, Any]]:
    meta = fetch_url_job_item_meta(task_conn, job_item_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"knowledge_job_items not found: {job_item_id}")

    root_id = normalize_text(meta["parent_source_id"])
    if not root_id:
        raise HTTPException(status_code=400, detail=f"parent_source_id is empty: {job_item_id}")

    root_meta = fetch_url_root_meta(local_db_path, root_id)
    if not root_meta:
        raise HTTPException(status_code=404, detail=f"url_roots not found: {root_id}")

    rows = fetch_url_page_rows(local_db_path, root_id)
    if not rows:
        raise HTTPException(status_code=400, detail=f"url_page_contents is empty: {root_id}")

    chunk_inputs: list[dict[str, Any]] = []
    sort_no = 1

    for row in rows:
        text = page_row_to_chunk_text(row)
        if not text:
            continue

        page_id = normalize_text(row["page_id"]) or f"page_{sort_no}"
        content_id = normalize_text(row["content_id"]) or page_id
        row_id = f"{page_id}:{content_id}"

        chunk_inputs.append(
            {
                "sort_no": sort_no,
                "content_text": text,
                "source_item_id": page_id,
                "content_type": normalize_text(row["content_type"]) or "html",
                "row_id": row_id,
            }
        )
        sort_no += 1

    if not chunk_inputs:
        raise HTTPException(status_code=400, detail=f"chunk inputs are empty: {root_id}")

    return chunk_inputs


def build_prompt_text(
    prompt_template: str,
    chunk,
    *,
    max_prompt_chars: int | None = None,
) -> str:
    template = (prompt_template or "").strip()
    if not template:
        raise RuntimeError("prompt template is empty")

    chunk_text = render_chunk_text(chunk, include_source_item_id=False).strip()
    if not chunk_text:
        raise RuntimeError("chunk_text is empty")

    prompt = f"{template}\n\n{chunk_text}".strip()

    if max_prompt_chars is not None and len(prompt) > int(max_prompt_chars):
        raise RuntimeError(
            f"prompt exceeds max_chars after template merge: "
            f"len={len(prompt)}, max_chars={max_prompt_chars}, chunk_no={chunk.chunk_no}"
        )

    return prompt


def build_public_url_chunk_rows(
    task_conn: sqlite3.Connection,
    local_db_path: str,
    job_item_id: str,
) -> list[dict[str, Any]]:
    chunk_config = load_chunk_config()
    if not chunk_config:
        raise HTTPException(status_code=500, detail="openai_chunk.json could not be loaded from GCS")

    qa_conf = get_required_url_chunk_conf(chunk_config, "qa")
    plain_conf = get_required_url_chunk_conf(chunk_config, "plain")

    qa_chunk_config_raw = to_chunk_config(qa_conf)
    plain_chunk_config_raw = to_chunk_config(plain_conf)

    qa_template = load_template_text(PROMPT_TEMPLATE_PATHS["public_url"]["qa"])
    plain_template = load_template_text(PROMPT_TEMPLATE_PATHS["public_url"]["plain"])

    if not qa_template:
        raise HTTPException(status_code=500, detail="public_url QA prompt template is empty")

    if not plain_template:
        raise HTTPException(status_code=500, detail="public_url PLAIN prompt template is empty")

    qa_chunk_config = to_prompt_safe_chunk_config(qa_chunk_config_raw, qa_template)
    plain_chunk_config = to_prompt_safe_chunk_config(plain_chunk_config_raw, plain_template)

    chunk_inputs = build_public_url_chunk_inputs(task_conn, local_db_path, job_item_id)
    created_at = utc_now_iso()

    chunk_rows: list[dict[str, Any]] = []

    qa_chunks = build_chunks(chunk_inputs, qa_chunk_config)
    plain_chunks = build_chunks(chunk_inputs, plain_chunk_config)

    chunk_no = 1
    max_len = max(len(qa_chunks), len(plain_chunks))

    for i in range(max_len):
        if i < len(qa_chunks):
            qa_chunk = qa_chunks[i]
            qa_prompt = build_prompt_text(
                qa_template,
                qa_chunk,
                max_prompt_chars=qa_chunk_config_raw.max_chars,
            )
            chunk_rows.append(
                {
                    "chunk_no": chunk_no,
                    "prompt_type": "qa",
                    "prompt": qa_prompt,
                    "row_count": qa_chunk.item_count,
                    "status": "new",
                    "created_at": created_at,
                }
            )
            chunk_no += 1

        if i < len(plain_chunks):
            plain_chunk = plain_chunks[i]
            plain_prompt = build_prompt_text(
                plain_template,
                plain_chunk,
                max_prompt_chars=plain_chunk_config_raw.max_chars,
            )
            chunk_rows.append(
                {
                    "chunk_no": chunk_no,
                    "prompt_type": "plain",
                    "prompt": plain_prompt,
                    "row_count": plain_chunk.item_count,
                    "status": "new",
                    "created_at": created_at,
                }
            )
            chunk_no += 1

    if not chunk_rows:
        raise HTTPException(status_code=400, detail=f"chunk generation result is empty: {job_item_id}")

    return chunk_rows
