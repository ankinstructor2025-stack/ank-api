from __future__ import annotations

import json
import re
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
from .openai_chunking import ChunkConfig, build_chunks, render_chunk_text


SOURCE_TYPE = "opendata"

TOC_SECTION_RE = re.compile(r"^第\s*[0-9０-９一二三四五六七八九十]+\s*[章節].*$")
TOC_BULLET_SECTION_RE = re.compile(r"^第\s*[0-9０-９一二三四五六七八九十]+\s*節\s*●.*$")
FIGURE_TITLE_RE = re.compile(r"^図表[0-9０-９\-－—―\.]+.*$")
ONLY_SHORT_WEAK_TEXT_RE = re.compile(r"^[\wぁ-んァ-ヶ一-龥・。．、，\-\s]+$")
BRACKET_LINE_NO_RE = re.compile(r"^\[\d+\]\s*")
TOC_WORD_RE = re.compile(r"(目\s*次|図\s*表\s*目\s*次)")
HEADER_REPEAT_RE = re.compile(r"^第\s*[0-9０-９一二三四五六七八九十]+\s*章\s+.*")
COVER_LIKE_RE = re.compile(r"(提出|表紙|この用紙は|食育推進施策)")
DOT_LEADER_RE = re.compile(r"[\.．・]{4,}|…{2,}")


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
            f"max_chars is too small for prompt template. max_chars={base_config.max_chars}, reserved={reserved}"
        )

    return ChunkConfig(
        max_chars=usable_chars,
        max_items=int(base_config.max_items),
        overlap_items=int(base_config.overlap_items),
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


def normalize_chunk_input_text(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""

    s = BRACKET_LINE_NO_RE.sub("", s).strip()
    s = re.sub(r"\s{2,}", " ", s)

    return s.strip()


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


def is_probably_toc_or_cover_text(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True

    if TOC_WORD_RE.search(s):
        return True

    if DOT_LEADER_RE.search(s):
        return True

    if COVER_LIKE_RE.search(s) and len(s) <= 200:
        return True

    if len(s) <= 120 and (TOC_SECTION_RE.fullmatch(s) or TOC_BULLET_SECTION_RE.fullmatch(s)):
        return True

    if len(s) <= 120 and FIGURE_TITLE_RE.fullmatch(s):
        return True

    if len(s) <= 20 and ONLY_SHORT_WEAK_TEXT_RE.fullmatch(s):
        return True

    if len(s) <= 250 and HEADER_REPEAT_RE.search(s) and "第1節" in s and "・・・・" in s:
        return True

    return False


def should_skip_pdf_chunk_input_text(text: str, page_no: int | None = None) -> bool:
    s = normalize_chunk_input_text(text)
    if not s:
        return True

    if page_no is not None and page_no <= 5:
        if is_probably_toc_or_cover_text(s):
            return True

    if is_probably_toc_or_cover_text(s):
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
) -> list[dict[str, Any]]:
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
            text = normalize_chunk_input_text(text)
            if not text:
                continue

            page_no = None
            if content_kind == "pdf" and isinstance(record, dict):
                try:
                    page_no = int(record.get("page", 0) or 0)
                except Exception:
                    page_no = None

            if content_kind == "pdf" and should_skip_pdf_chunk_input_text(text, page_no=page_no):
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

    return expanded_rows


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
            f"prompt exceeds max_chars after template merge: len={len(prompt)}, max_chars={max_prompt_chars}, chunk_no={chunk.chunk_no}"
        )

    return prompt


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

    qa_chunk_config_raw = to_chunk_config(qa_conf)
    plain_chunk_config_raw = to_chunk_config(plain_conf)

    qa_template = load_template_text(PROMPT_TEMPLATE_PATHS["opendata"]["qa"])
    plain_template = load_template_text(PROMPT_TEMPLATE_PATHS["opendata"]["plain"])

    if not qa_template:
        raise RuntimeError("QA prompt template is empty")

    if not plain_template:
        raise RuntimeError("PLAIN prompt template is empty")

    qa_chunk_config = to_prompt_safe_chunk_config(qa_chunk_config_raw, qa_template)
    plain_chunk_config = to_prompt_safe_chunk_config(plain_chunk_config_raw, plain_template)

    expanded_rows = expand_opendata_contents_to_chunk_inputs(conn, job_item_id)
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
