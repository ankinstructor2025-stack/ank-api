from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from google.cloud import storage

from .knowledge_generate_common import (
    BUCKET_NAME,
    PROMPT_TEMPLATE_PATHS,
    load_chunk_config,
    load_template_text,
    open_user_db,
)
from .content_splitter_csv import split_csv_records
from .content_splitter_pdf import split_pdf_records
from .content_splitter_text import split_text_records
from .openai_chunking import ChunkConfig, build_chunks, render_chunk_text


SOURCE_TYPE = "upload"
SUPPORTED_UPLOAD_EXTS = {"pdf", "csv", "txt", "json"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_upload_ext(ext: str | None) -> str:
    value = str(ext or "").strip().lower()
    if value == "text":
        return "txt"
    return value


def build_chunk_id(
    job_id: str,
    job_item_id: str,
    prompt_type: str,
    chunk_no: int,
) -> str:
    return f"{job_id}_{job_item_id}_{prompt_type}_{chunk_no}"


def stringify_csv_row(record: dict[str, Any]) -> str:
    values: list[str] = []

    for key, value in record.items():
        key_text = str(key or "").strip()
        value_text = str(value or "").strip()

        if not key_text and not value_text:
            continue

        if key_text:
            values.append(f"{key_text}: {value_text}")
        else:
            values.append(value_text)

    return " | ".join(values).strip()


def build_upload_gcs_path(file_row: sqlite3.Row) -> str:
    """
    upload_files に gcs_path がある場合はそれを優先。
    無い場合は file_name が users/ から始まるときだけ GCS パスとして扱う。
    """
    keys = set(file_row.keys())

    if "gcs_path" in keys:
        gcs_path = str(file_row["gcs_path"] or "").strip()
        if gcs_path:
            return gcs_path

    file_name = str(file_row["file_name"] or "").strip() if "file_name" in keys else ""
    if file_name.startswith("users/"):
        return file_name

    file_path = str(file_row["file_path"] or "").strip() if "file_path" in keys else ""
    if file_path.startswith("users/"):
        return file_path

    raise RuntimeError("upload gcs_path not found. upload_files.gcs_path is required.")


def download_gcs_binary(gcs_path: str) -> bytes:
    if not gcs_path:
        raise RuntimeError("gcs_path is empty")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)

    if not blob.exists():
        raise RuntimeError(f"GCS blob not found: gs://{BUCKET_NAME}/{gcs_path}")

    return blob.download_as_bytes()


def get_required_upload_chunk_conf(
    chunk_config: dict[str, Any],
    ext: str,
    prompt_type: str,
) -> dict[str, Any]:
    if not chunk_config:
        raise RuntimeError("openai_chunk.json is empty or not loaded")

    upload_conf = chunk_config.get("upload")
    if not upload_conf:
        raise RuntimeError("openai_chunk.json: 'upload' section missing")

    ext_norm = normalize_upload_ext(ext)

    # 新形式: upload.<ext>.<prompt_type>
    ext_conf = upload_conf.get(ext_norm)
    if isinstance(ext_conf, dict):
        conf = ext_conf.get(prompt_type)
        if conf:
            return conf

    # 後方互換: upload.<prompt_type>
    legacy_conf = upload_conf.get(prompt_type)
    if legacy_conf:
        return legacy_conf

    raise RuntimeError(f"openai_chunk.json: upload.{ext_norm}.{prompt_type} missing")


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


def fetch_upload_file_row(local_db_path: str, file_id: str) -> sqlite3.Row | None:
    conn = open_user_db(local_db_path)
    try:
        cur = conn.execute(
            """
            SELECT
                file_id,
                file_name,
                file_path,
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


def fetch_upload_content_rows(conn: sqlite3.Connection, job_item_id: str) -> list[sqlite3.Row]:
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
        ORDER BY sort_no, row_id
        """,
        (job_item_id,),
    )
    return cur.fetchall()


def split_upload_binary(
    binary: bytes,
    *,
    ext: str,
) -> list[dict[str, Any]]:
    ext_norm = normalize_upload_ext(ext)

    if ext_norm == "pdf":
        return split_pdf_records(binary, max_rows=2000)

    if ext_norm == "csv":
        return split_csv_records(binary, max_rows=2000)

    if ext_norm in {"txt", "json"}:
        return split_text_records(binary, max_rows=2000)

    raise RuntimeError(f"unsupported upload ext: {ext_norm}")


def record_to_content_text(
    ext: str,
    record: Any,
) -> str:
    ext_norm = normalize_upload_ext(ext)

    if ext_norm == "csv":
        if isinstance(record, dict):
            return stringify_csv_row(record)
        return str(record or "").strip()

    if isinstance(record, dict):
        return str(record.get("text") or "").strip()

    return str(record or "").strip()


def record_to_suffix(
    ext: str,
    record: Any,
    local_index: int,
) -> str:
    ext_norm = normalize_upload_ext(ext)

    if ext_norm == "pdf" and isinstance(record, dict):
        return f"page_{record.get('page', local_index)}"

    return str(local_index)


def expand_upload_contents_to_chunk_inputs(
    conn: sqlite3.Connection,
    job_item_id: str,
) -> tuple[list[dict[str, Any]], str]:
    content_rows = fetch_upload_content_rows(conn, job_item_id)

    if not content_rows:
        raise HTTPException(status_code=400, detail=f"knowledge_contents is empty: {job_item_id}")

    expanded_rows: list[dict[str, Any]] = []
    global_sort_no = 1
    detected_ext: str | None = None

    for content_row in content_rows:
        file_path = normalize_text(content_row["content_text"])
        ext = normalize_upload_ext(content_row["content_type"])

        if not file_path:
            continue

        if ext not in SUPPORTED_UPLOAD_EXTS:
            raise HTTPException(status_code=400, detail=f"unsupported upload ext: {ext}")

        if detected_ext is None:
            detected_ext = ext
        elif detected_ext != ext:
            raise HTTPException(
                status_code=400,
                detail=f"multiple upload content types are not supported in one job_item: {job_item_id}",
            )

        binary = download_gcs_binary(file_path)
        records = split_upload_binary(binary, ext=ext)

        base_source_item_id = normalize_text(content_row["source_item_id"]) or file_path
        base_row_id = normalize_text(content_row["row_id"]) or file_path

        local_index = 0
        for record in records:
            local_index += 1

            text = record_to_content_text(ext, record)
            if not text:
                continue

            suffix = record_to_suffix(ext, record, local_index)

            expanded_rows.append(
                {
                    "sort_no": global_sort_no,
                    "content_text": text,
                    "source_item_id": f"{base_source_item_id}:{suffix}",
                    "content_type": ext,
                    "row_id": f"{base_row_id}:{suffix}",
                }
            )
            global_sort_no += 1

    if not expanded_rows:
        raise HTTPException(status_code=400, detail=f"expanded content is empty: {job_item_id}")

    if not detected_ext:
        raise HTTPException(status_code=400, detail=f"upload ext could not be determined: {job_item_id}")

    return expanded_rows, detected_ext


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


def build_upload_chunk_rows(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
) -> list[dict[str, Any]]:
    chunk_config = load_chunk_config()
    if not chunk_config:
        raise HTTPException(status_code=500, detail="openai_chunk.json could not be loaded from GCS")

    expanded_rows, upload_ext = expand_upload_contents_to_chunk_inputs(conn, job_item_id)

    qa_conf = get_required_upload_chunk_conf(chunk_config, upload_ext, "qa")
    plain_conf = get_required_upload_chunk_conf(chunk_config, upload_ext, "plain")

    qa_chunk_config_raw = to_chunk_config(qa_conf)
    plain_chunk_config_raw = to_chunk_config(plain_conf)

    qa_template = load_template_text(PROMPT_TEMPLATE_PATHS["upload"]["qa"])
    plain_template = load_template_text(PROMPT_TEMPLATE_PATHS["upload"]["plain"])

    if not qa_template:
        raise HTTPException(status_code=500, detail="upload QA prompt template is empty")

    if not plain_template:
        raise HTTPException(status_code=500, detail="upload PLAIN prompt template is empty")

    qa_chunk_config = to_prompt_safe_chunk_config(qa_chunk_config_raw, qa_template)
    plain_chunk_config = to_prompt_safe_chunk_config(plain_chunk_config_raw, plain_template)

    created_at = utc_now_iso()
    chunk_rows: list[dict[str, Any]] = []

    qa_chunks = build_chunks(expanded_rows, qa_chunk_config)
    for chunk in qa_chunks:
        prompt = build_prompt_text(
            qa_template,
            chunk,
            max_prompt_chars=qa_chunk_config_raw.max_chars,
        )
        chunk_rows.append(
            {
                "chunk_id": build_chunk_id(job_id, job_item_id, "qa", chunk.chunk_no),
                "job_id": job_id,
                "job_item_id": job_item_id,
                "source_type": SOURCE_TYPE,
                "chunk_no": chunk.chunk_no,
                "prompt": prompt,
                "prompt_type": "qa",
                "row_count": chunk.item_count,
                "status": "new",
                "created_at": created_at,
            }
        )

    plain_chunks = build_chunks(expanded_rows, plain_chunk_config)
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
                "chunk_id": build_chunk_id(job_id, job_item_id, "plain", actual_chunk_no),
                "job_id": job_id,
                "job_item_id": job_item_id,
                "source_type": SOURCE_TYPE,
                "chunk_no": actual_chunk_no,
                "prompt": prompt,
                "prompt_type": "plain",
                "row_count": chunk.item_count,
                "status": "new",
                "created_at": created_at,
            }
        )

    if not chunk_rows:
        raise HTTPException(status_code=400, detail=f"chunk generation result is empty: {job_item_id}")

    return chunk_rows
