from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from .knowledge_generate_common import (
    PROMPT_TEMPLATE_PATHS,
    load_chunk_config,
    load_template_text,
    now_iso,
    open_user_db,
)
from .openai_chunking import ChunkConfig, build_chunks, render_chunk_text


SOURCE_TYPE = "kokkai"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    merged = "\n".join(line for line in lines if line)
    return merged.strip()


def get_required_kokkai_chunk_conf(
    chunk_config: dict[str, Any],
    prompt_type: str,
) -> dict[str, Any]:
    if not chunk_config:
        raise RuntimeError("openai_chunk.json is empty or not loaded")

    kokkai_conf = chunk_config.get("kokkai")
    if not kokkai_conf:
        raise RuntimeError("openai_chunk.json: 'kokkai' section missing")

    conf = kokkai_conf.get(prompt_type)
    if not conf:
        raise RuntimeError(f"openai_chunk.json: kokkai.{prompt_type} missing")

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


def fetch_kokkai_speech_rows_for_chunk(
    conn: sqlite3.Connection,
    issue_id: str,
) -> list[sqlite3.Row]:
    cur = conn.execute(
        """
        SELECT
            issue_id,
            speech_id,
            speech_order,
            status,
            speaker,
            speech,
            created_at,
            updated_at
        FROM kokkai_document_rows
        WHERE issue_id = ?
        ORDER BY speech_order, speech_id
        """,
        (issue_id,),
    )
    return cur.fetchall()


def speech_row_to_chunk_text(row: sqlite3.Row) -> str:
    speaker = normalize_text(row["speaker"])
    speech = normalize_text(row["speech"])

    if not speech:
        return ""

    if speaker:
        return f"発言者: {speaker}\n本文: {speech}"

    return speech


def build_kokkai_chunk_inputs(
    conn: sqlite3.Connection,
    job_item_id: str,
) -> list[dict[str, Any]]:
    meta = fetch_kokkai_job_item_meta(conn, job_item_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"knowledge_job_items not found: {job_item_id}")

    issue_id = normalize_text(meta["parent_source_id"])
    if not issue_id:
        raise HTTPException(status_code=400, detail=f"parent_source_id is empty: {job_item_id}")

    rows = fetch_kokkai_speech_rows_for_chunk(conn, issue_id)
    if not rows:
        raise HTTPException(status_code=400, detail=f"kokkai_document_rows is empty: {issue_id}")

    chunk_inputs: list[dict[str, Any]] = []
    sort_no = 1

    for row in rows:
        text = speech_row_to_chunk_text(row)
        if not text:
            continue

        source_item_id = normalize_text(row["speech_id"]) or f"speech_{sort_no}"
        row_id = f'{issue_id}:{source_item_id}'

        chunk_inputs.append(
            {
                "sort_no": sort_no,
                "content_text": text,
                "source_item_id": source_item_id,
                "content_type": "speech",
                "row_id": row_id,
            }
        )
        sort_no += 1

    if not chunk_inputs:
        raise HTTPException(status_code=400, detail=f"chunk inputs are empty: {issue_id}")

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


def build_kokkai_chunk_rows(
    conn: sqlite3.Connection,
    job_item_id: str,
) -> list[dict[str, Any]]:
    chunk_config = load_chunk_config()
    if not chunk_config:
        raise HTTPException(status_code=500, detail="openai_chunk.json could not be loaded from GCS")

    qa_conf = get_required_kokkai_chunk_conf(chunk_config, "qa")
    plain_conf = get_required_kokkai_chunk_conf(chunk_config, "plain")

    qa_chunk_config_raw = to_chunk_config(qa_conf)
    plain_chunk_config_raw = to_chunk_config(plain_conf)

    qa_template = load_template_text(PROMPT_TEMPLATE_PATHS["kokkai"]["qa"])
    plain_template = load_template_text(PROMPT_TEMPLATE_PATHS["kokkai"]["plain"])

    if not qa_template:
        raise HTTPException(status_code=500, detail="kokkai QA prompt template is empty")

    if not plain_template:
        raise HTTPException(status_code=500, detail="kokkai PLAIN prompt template is empty")

    qa_chunk_config = to_prompt_safe_chunk_config(qa_chunk_config_raw, qa_template)
    plain_chunk_config = to_prompt_safe_chunk_config(plain_chunk_config_raw, plain_template)

    chunk_inputs = build_kokkai_chunk_inputs(conn, job_item_id)
    created_at = utc_now_iso()

    chunk_rows: list[dict[str, Any]] = []

    qa_chunks = build_chunks(chunk_inputs, qa_chunk_config)
    for chunk in qa_chunks:
        prompt = build_prompt_text(
            qa_template,
            chunk,
            max_prompt_chars=qa_chunk_config_raw.max_chars,
        )
        chunk_rows.append(
            {
                "chunk_no": chunk.chunk_no,
                "prompt_type": "qa",
                "prompt": prompt,
                "row_count": chunk.item_count,
                "status": "new",
                "created_at": created_at,
            }
        )

    plain_chunks = build_chunks(chunk_inputs, plain_chunk_config)
    plain_chunk_no_base = len(qa_chunks)

    for chunk in plain_chunks:
        actual_chunk_no = plain_chunk_no_base + chunk.chunk_no
        prompt = build_prompt_text(
            plain_template,
            chunk,
            max_prompt_chars=plain_chunk_config_raw.max_chars,
        )
        chunk_rows.append(
            {
                "chunk_no": actual_chunk_no,
                "prompt_type": "plain",
                "prompt": prompt,
                "row_count": chunk.item_count,
                "status": "new",
                "created_at": created_at,
            }
        )

    if not chunk_rows:
        raise HTTPException(status_code=400, detail=f"chunk generation result is empty: {job_item_id}")

    return chunk_rows
