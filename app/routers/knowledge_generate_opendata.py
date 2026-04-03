from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from google.cloud import storage

from .knowledge_generate_common import (
    BUCKET_NAME,
    PROMPT_TEMPLATE_PATHS,
    load_chunk_config,
    load_template_text,
)
from .content_detector import detect_content_kind
from .content_splitter_csv import split_csv_records
from .content_splitter_pdf import split_pdf_records
from .content_splitter_text import split_text_records
from .openai_chunking import ChunkConfig, build_chunks
from .openai_prompt_builder import build_opendata_prompt_text


SOURCE_TYPE = "opendata"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_chunk_id(
    job_id: str,
    job_item_id: str,
    prompt_type: str,
    chunk_no: int,
) -> str:
    return f"{job_id}_{job_item_id}_{prompt_type}_{chunk_no}"


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
        ORDER BY sort_no, row_id
        """,
        (job_item_id,),
    )
    return cur.fetchall()


def fetch_job_item_info(
    conn: sqlite3.Connection,
    job_item_id: str,
) -> sqlite3.Row:
    cur = conn.execute(
        """
        SELECT
            job_item_id,
            parent_source_id,
            parent_key1,
            parent_key2,
            parent_label,
            row_count
        FROM knowledge_job_items
        WHERE job_item_id = ?
        LIMIT 1
        """,
        (job_item_id,),
    )
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"knowledge_job_items not found: job_item_id={job_item_id}")
    return row


def download_gcs_binary(gcs_path: str) -> bytes:
    if not gcs_path:
        raise RuntimeError("gcs_path is empty")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)

    if not blob.exists():
        raise RuntimeError(f"GCS blob not found: gs://{BUCKET_NAME}/{gcs_path}")

    return blob.download_as_bytes()


def stringify_csv_row(row: dict[str, Any]) -> str:
    values: list[str] = []

    for key, value in row.items():
        key_text = str(key or "").strip()
        value_text = str(value or "").strip()

        if not key_text and not value_text:
            continue

        if key_text:
            values.append(f"{key_text}: {value_text}")
        else:
            values.append(value_text)

    return " | ".join(values).strip()


def convert_json_text_to_records(
    text: str,
    max_rows: int = 2000,
) -> list[dict[str, Any]]:
    try:
        obj = json.loads(text)
        pretty = json.dumps(obj, ensure_ascii=False, indent=2)
        return split_text_records(pretty.encode("utf-8"), max_rows=max_rows)
    except Exception:
        return split_text_records(text.encode("utf-8"), max_rows=max_rows)


def split_binary_by_kind(
    binary: bytes,
    *,
    kind: str,
    max_rows: int = 2000,
) -> list[dict[str, Any]]:
    kind_norm = (kind or "").strip().lower()

    if kind_norm == "csv":
        return split_csv_records(binary, max_rows=max_rows)

    if kind_norm == "pdf":
        return split_pdf_records(binary, max_rows=max_rows)

    if kind_norm == "json":
        text = binary.decode("utf-8", errors="ignore")
        return convert_json_text_to_records(text, max_rows=max_rows)

    return split_text_records(binary, max_rows=max_rows)


def record_to_content_text(
    content_kind: str,
    record: Any,
) -> str:
    if isinstance(record, dict):
        if content_kind == "csv":
            return stringify_csv_row(record)
        return str(record.get("text") or "").strip()

    return str(record or "").strip()


def record_to_suffix(
    content_kind: str,
    record: Any,
    local_index: int,
) -> str:
    if content_kind == "pdf" and isinstance(record, dict):
        return f"page_{record.get('page', local_index)}"
    return str(local_index)


def is_probably_gcs_path(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False

    if value.startswith("users/"):
        return True

    if value.startswith("template/"):
        return True

    if value.count("/") >= 2 and "." in value.rsplit("/", 1)[-1]:
        return True

    return False


def build_records_from_content_row(
    content_row: sqlite3.Row,
    job_item_id: str,
    file_sort_no: int,
) -> tuple[list[dict[str, Any]], str, str, str]:
    content_text = (content_row["content_text"] or "").strip()
    explicit_content_type = (content_row["content_type"] or "").strip().lower()
    source_item_id = (content_row["source_item_id"] or "").strip()
    row_id = (content_row["row_id"] or "").strip()

    if not content_text:
        return [], "", "", ""

    if is_probably_gcs_path(content_text):
        binary = download_gcs_binary(content_text)
        detected_kind = explicit_content_type or detect_content_kind(
            filename=content_text.rsplit("/", 1)[-1],
            source_path=content_text,
        )
        records = split_binary_by_kind(binary, kind=detected_kind, max_rows=2000)
        base_source_item_id = source_item_id or content_text
        base_row_id = row_id or content_text
        content_kind = detected_kind or "text"
        return records, base_source_item_id, base_row_id, content_kind

    records = convert_json_text_to_records(content_text, max_rows=2000)
    base_source_item_id = source_item_id or f"{job_item_id}:{file_sort_no}"
    base_row_id = row_id or f"{job_item_id}:{file_sort_no}"
    content_kind = explicit_content_type or "text"
    return records, base_source_item_id, base_row_id, content_kind


def expand_opendata_contents_to_chunk_inputs(
    conn: sqlite3.Connection,
    job_item_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    job_item = fetch_job_item_info(conn, job_item_id)
    content_rows = fetch_opendata_content_rows(conn, job_item_id)

    if not content_rows:
        raise RuntimeError(f"knowledge_contents is empty for job_item_id={job_item_id}")

    expanded_rows: list[dict[str, Any]] = []
    global_sort_no = 1

    for content_row in content_rows:
        file_sort_no = int(content_row["sort_no"] or 0)

        records, base_source_item_id, base_row_id, content_kind = build_records_from_content_row(
            content_row=content_row,
            job_item_id=job_item_id,
            file_sort_no=file_sort_no,
        )

        if not records:
            continue

        local_index = 0

        for record in records:
            local_index += 1

            text = record_to_content_text(content_kind, record)
            if not text:
                continue

            suffix = record_to_suffix(content_kind, record, local_index)

            expanded_rows.append(
                {
                    "sort_no": global_sort_no,
                    "content_text": text,
                    "source_item_id": f"{base_source_item_id}:{suffix}",
                    "content_type": content_kind,
                    "row_id": f"{base_row_id}:{suffix}",
                }
            )
            global_sort_no += 1

    if not expanded_rows:
        raise RuntimeError(f"expanded content is empty for job_item_id={job_item_id}")

    metadata = {
        "parent_source_id": job_item["parent_source_id"],
        "parent_key1": job_item["parent_key1"],
        "parent_key2": job_item["parent_key2"],
        "parent_label": job_item["parent_label"],
        "row_count": int(job_item["row_count"] or 0),
    }

    return expanded_rows, metadata


def build_opendata_chunk_rows(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
) -> list[dict[str, Any]]:
    chunk_config = load_chunk_config()
    if not chunk_config:
        raise RuntimeError("openai_chunk.json could not be loaded from GCS")

    qa_conf = get_required_opendata_chunk_conf(chunk_config, "qa")
    plain_conf = get_required_opendata_chunk_conf(chunk_config, "plain")

    qa_chunk_config = to_chunk_config(qa_conf)
    plain_chunk_config = to_chunk_config(plain_conf)

    qa_template = load_template_text(PROMPT_TEMPLATE_PATHS["opendata"]["qa"])
    plain_template = load_template_text(PROMPT_TEMPLATE_PATHS["opendata"]["plain"])

    if not qa_template:
        raise RuntimeError("QA prompt template is empty")

    if not plain_template:
        raise RuntimeError("PLAIN prompt template is empty")

    expanded_rows, metadata = expand_opendata_contents_to_chunk_inputs(conn, job_item_id)
    total_input_count = len(expanded_rows)
    created_at = utc_now_iso()

    chunk_rows: list[dict[str, Any]] = []

    qa_chunks = build_chunks(expanded_rows, qa_chunk_config)
    for chunk in qa_chunks:
        prompt = build_opendata_prompt_text(
            job_item_id=job_item_id,
            prompt_template=qa_template,
            chunk=chunk,
            parent_source_id=metadata["parent_source_id"],
            parent_key1=metadata["parent_key1"],
            parent_key2=metadata["parent_key2"],
            parent_label=metadata["parent_label"],
            row_count=metadata["row_count"] or total_input_count,
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

        prompt = build_opendata_prompt_text(
            job_item_id=job_item_id,
            prompt_template=plain_template,
            chunk=chunk,
            parent_source_id=metadata["parent_source_id"],
            parent_key1=metadata["parent_key1"],
            parent_key2=metadata["parent_key2"],
            parent_label=metadata["parent_label"],
            row_count=metadata["row_count"] or total_input_count,
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
        raise RuntimeError("chunk generation result is empty")

    return chunk_rows


def insert_opendata_chunks(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
) -> int:
    chunk_rows = build_opendata_chunk_rows(conn, job_id, job_item_id)

    conn.execute(
        """
        DELETE FROM knowledge_job_chunks
        WHERE job_id = ?
          AND job_item_id = ?
          AND source_type = ?
        """,
        (job_id, job_item_id, SOURCE_TYPE),
    )

    for row in chunk_rows:
        conn.execute(
            """
            INSERT INTO knowledge_job_chunks (
                chunk_id,
                job_id,
                job_item_id,
                source_type,
                chunk_no,
                prompt,
                prompt_type,
                row_count,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["chunk_id"],
                row["job_id"],
                row["job_item_id"],
                row["source_type"],
                row["chunk_no"],
                row["prompt"],
                row["prompt_type"],
                row["row_count"],
                row["status"],
                row["created_at"],
            ),
        )

    conn.commit()
    return len(chunk_rows)
