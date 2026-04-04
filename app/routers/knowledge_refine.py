from __future__ import annotations

import os
import json
import math
import re
import sqlite3
import logging
from typing import Optional, List

try:
    from janome.tokenizer import Tokenizer
except ImportError:
    Tokenizer = None
from zoneinfo import ZoneInfo
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel
from google.cloud import storage
from openai import OpenAI

from app.core.common import user_task_db_path, local_task_db_path

import firebase_admin
from firebase_admin import auth as fb_auth


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge/refine", tags=["knowledge_refine"])

JST = ZoneInfo("Asia/Tokyo")
BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

QA_DEDUP_THRESHOLD = float(os.getenv("KNOWLEDGE_QA_DEDUP_THRESHOLD", "0.85"))
PLAIN_DEDUP_THRESHOLD = float(os.getenv("KNOWLEDGE_PLAIN_DEDUP_THRESHOLD", "0.95"))


def job_status_json_path(uid: str, job_id: str) -> str:
    return f"users/{uid}/job_status/{job_id}.json"


def now_jst_iso() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def ensure_firebase_initialized():
    if firebase_admin._apps:
        return
    firebase_admin.initialize_app(options={"projectId": "ank-firebase"})


def get_uid_from_auth_header(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    token = authorization.replace("Bearer ", "", 1).strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")

    ensure_firebase_initialized()

    try:
        decoded = fb_auth.verify_id_token(token)
        uid = decoded.get("uid")
        if not uid:
            raise HTTPException(status_code=401, detail="uid not found in token")
        return uid
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid ID token: {e}")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def unique_keep_order(values: list[str]) -> list[str]:
    seen = set()
    result: list[str] = []

    for value in values:
        v = normalize_text(value)
        if not v:
            continue
        if v in seen:
            continue
        seen.add(v)
        result.append(v)

    return result




_janome_tokenizer: Optional[Tokenizer] = None


def get_janome_tokenizer() -> Tokenizer:
    global _janome_tokenizer

    if Tokenizer is None:
        raise HTTPException(status_code=500, detail="janome not installed. pip install janome")

    if _janome_tokenizer is None:
        _janome_tokenizer = Tokenizer()

    return _janome_tokenizer


def tokenize_text_for_fts(text: str | None) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    tokenizer = get_janome_tokenizer()
    tokens: list[str] = []

    for token in tokenizer.tokenize(normalized):
        surface = normalize_text(token.surface)
        if not surface:
            continue

        if surface in {"。", "、", ",", ".", "!", "?", "！", "？", "「", "」", "（", "）", "(", ")"}:
            continue

        parts = token.part_of_speech.split(",")
        pos_major = parts[0] if parts else ""

        base = getattr(token, "base_form", None) or surface
        if base == "*":
            base = surface
        base = normalize_text(base)

        # 品詞カットはしない。ただし最低限のノイズだけ抑える
        # 記号・空文字・1文字ノイズを減らす
        if len(surface) >= 2:
            tokens.append(surface)

        # 動詞・形容詞だけは原形も持っておく
        if pos_major in {"動詞", "形容詞"} and len(base) >= 2 and base != surface:
            tokens.append(base)

    return unique_keep_order(tokens)
def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def embed_normalized_fields(
    question_normalize: Optional[str],
    answer_normalize: Optional[str],
    content_normalize: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    targets = [
        ("question", normalize_text(question_normalize)),
        ("answer", normalize_text(answer_normalize)),
        ("content", normalize_text(content_normalize)),
    ]

    inputs: list[str] = []
    keys: list[str] = []

    for key, text in targets:
        if text:
            keys.append(key)
            inputs.append(text)

    if not inputs:
        return None, None, None

    client = get_openai_client()

    logger.info("EMBED request start: model=%s fields=%s", EMBEDDING_MODEL, ",".join(keys))

    try:
        res = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=inputs,
            encoding_format="float",
        )
    except Exception:
        logger.exception("EMBED generation failed")
        raise

    logger.info("EMBED response received: count=%s", len(res.data))

    vector_map: dict[str, str] = {}
    for key, item in zip(keys, res.data):
        vector_map[key] = json.dumps(item.embedding, ensure_ascii=False)

    return (
        vector_map.get("question"),
        vector_map.get("answer"),
        vector_map.get("content"),
    )


class RefineJobRow(BaseModel):
    job_id: str
    source_type: str
    source_name: Optional[str] = None
    request_type: Optional[str] = None
    status: str
    phase: Optional[str] = None
    selected_count: int = 0
    qa_count: int = 0
    plain_count: int = 0
    error_count: int = 0
    requested_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None


class RefineJobListResponse(BaseModel):
    jobs: List[RefineJobRow]
    total_count: int


class RefineJobItemRow(BaseModel):
    job_item_id: str
    job_id: str
    source_item_id: Optional[str] = None
    source_type: Optional[str] = None
    title: Optional[str] = None
    status: str = "new"
    qa_count: int = 0
    plain_count: int = 0
    error_count: int = 0
    error_message: Optional[str] = None
    requested_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class RefineJobItemListResponse(BaseModel):
    job_id: str
    items: List[RefineJobItemRow]
    total_count: int


class RefineActionResponse(BaseModel):
    ok: bool
    job_id: str
    action: str
    status: str
    message: str


def download_job_db(uid: str, job_id: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_task_db_path(uid, job_id)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=404,
            detail=f"job db not found: {db_gcs_path}",
        )

    local_db_path = local_task_db_path(uid, job_id)
    db_blob.download_to_filename(local_db_path)
    return local_db_path


def upload_job_db(uid: str, job_id: str, local_db_path: str) -> None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_task_db_path(uid, job_id)
    blob = bucket.blob(db_gcs_path)
    blob.upload_from_filename(local_db_path, content_type="application/octet-stream")
    print("UPLOAD FINISHED:", db_gcs_path)


def read_job_status_json(uid: str, job_id: str) -> dict:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(job_status_json_path(uid, job_id))
    if not blob.exists():
        return {}
    text = blob.download_as_text(encoding="utf-8")
    data = json.loads(text)
    return data if isinstance(data, dict) else {}


def write_job_status_json(uid: str, job_id: str, data: dict) -> None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(job_status_json_path(uid, job_id))
    blob.upload_from_string(
        json.dumps(data, ensure_ascii=False, indent=2),
        content_type="application/json",
    )


def update_job_status_json(uid: str, job_id: str, updates: dict) -> dict:
    data = read_job_status_json(uid, job_id)
    data.update(updates)
    data["updated_at"] = now_jst_iso()
    write_job_status_json(uid, job_id, data)
    return data


def knowledge_template_db_path() -> str:
    return "template/knowledge_template.sqlite"


def build_knowledge_db_filename() -> str:
    return f"knowledge_{datetime.now(JST).strftime('%Y%m%d%H%M%S')}.sqlite"


def knowledge_db_path(uid: str, filename: str) -> str:
    return f"users/{uid}/{filename}"


def download_knowledge_template_db(local_path: str) -> None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(knowledge_template_db_path())
    if not blob.exists():
        raise HTTPException(
            status_code=500,
            detail=f"knowledge template not found: {knowledge_template_db_path()}",
        )

    blob.download_to_filename(local_path)


def upload_knowledge_db(uid: str, local_path: str, filename: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    gcs_path = knowledge_db_path(uid, filename)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path, content_type="application/octet-stream")
    return gcs_path


def embed_search_text(search_text: str | None) -> Optional[str]:
    text = normalize_text(search_text)
    if not text:
        return None

    client = get_openai_client()

    try:
        res = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text],
            encoding_format="float",
        )
    except Exception:
        logger.exception("embed_search_text failed")
        raise

    if not res.data:
        return None

    return json.dumps(res.data[0].embedding, ensure_ascii=False)


def build_search_text_from_row(row: sqlite3.Row) -> str:
    title = row["title"] if "title" in row.keys() else None
    question = row["question"] if "question" in row.keys() else None
    answer = row["answer"] if "answer" in row.keys() else None
    content = row["content"] if "content" in row.keys() else None

    raw_parts = [
        normalize_text(title),
        normalize_text(question),
        normalize_text(answer),
        normalize_text(content),
    ]
    raw_parts = [x for x in raw_parts if x]

    token_parts: list[str] = []
    for value in [title, question, answer, content]:
        token_parts.extend(tokenize_text_for_fts(value))

    merged = unique_keep_order(raw_parts) + token_parts
    return " ".join(merged).strip()
def build_entry_title(row: sqlite3.Row) -> str:
    candidates = [
        row["title"] if "title" in row.keys() else None,
        row["question"] if "question" in row.keys() else None,
        row["content"] if "content" in row.keys() else None,
        row["source_item_id"] if "source_item_id" in row.keys() else None,
    ]

    for value in candidates:
        text = normalize_text(value)
        if text:
            return text[:120]

    return ""


def build_entry_embedding_json(row: sqlite3.Row, search_text: str) -> Optional[str]:
    knowledge_type = str(row["knowledge_type"] or "").lower()

    if knowledge_type == "qa":
        merged = mean_vectors([
            parse_vector(row["question_vector"]),
            parse_vector(row["answer_vector"]),
        ])
        if merged:
            return json.dumps(merged, ensure_ascii=False)

    if knowledge_type == "plain":
        content_vec = parse_vector(row["content_vector"])
        if content_vec:
            return json.dumps(content_vec, ensure_ascii=False)

    return embed_search_text(search_text)


def upsert_knowledge_meta(
    conn: sqlite3.Connection,
    meta_key: str,
    meta_value: str,
) -> None:
    conn.execute(
        """
        INSERT INTO knowledge_meta(meta_key, meta_value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(meta_key)
        DO UPDATE SET
            meta_value = excluded.meta_value,
            updated_at = excluded.updated_at
        """,
        (meta_key, meta_value, now_jst_iso()),
    )


def insert_knowledge_db_row(
    conn: sqlite3.Connection,
    database_name: str,
    source_type: str,
) -> None:
    conn.execute(
        """
        INSERT INTO knowledge_db (
            database_name,
            source_type,
            created_at
        )
        VALUES (?, ?, ?)
        """,
        (
            database_name,
            source_type,
            now_jst_iso(),
        ),
    )


def build_knowledge_db_for_job(
    conn: sqlite3.Connection,
    uid: str,
    job_id: str,
) -> dict[str, object]:
    columns = get_table_columns(conn, "knowledge_items")
    required_columns = {
        "knowledge_id",
        "job_id",
        "knowledge_type",
        "title",
        "question",
        "answer",
        "content",
        "question_vector",
        "answer_vector",
        "content_vector",
        "source_type",
        "source_item_id",
        "language",
        "sort_no",
        "status",
        "is_duplicate",
    }
    missing = [c for c in required_columns if c not in columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"knowledge_items missing columns: {', '.join(missing)}",
        )

    cur = conn.execute(
        """
        SELECT
            knowledge_id,
            job_id,
            knowledge_type,
            title,
            question,
            answer,
            content,
            question_vector,
            answer_vector,
            content_vector,
            source_type,
            source_item_id,
            language,
            sort_no
        FROM knowledge_items
        WHERE job_id = ?
          AND COALESCE(is_duplicate, 0) = 0
        ORDER BY sort_no ASC, knowledge_id ASC
        """,
        (job_id,),
    )
    rows = cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=409,
            detail="build target not found. no active representative knowledge_items",
        )

    filename = build_knowledge_db_filename()
    local_knowledge_db_path = f"/tmp/{filename}"

    download_knowledge_template_db(local_knowledge_db_path)

    kconn = sqlite3.connect(local_knowledge_db_path)
    kconn.row_factory = sqlite3.Row

    try:
        entry_columns = get_table_columns(kconn, "knowledge_entries")
        required_entry_columns = {
            "entry_id",
            "job_id",
            "knowledge_id",
            "knowledge_type",
            "title",
            "question",
            "answer",
            "content",
            "search_text",
            "embedding",
            "source_type",
            "source_item_id",
            "source_label",
            "language",
            "sort_no",
            "created_at",
            "updated_at",
        }
        missing_entry_columns = [c for c in required_entry_columns if c not in entry_columns]
        if missing_entry_columns:
            raise HTTPException(
                status_code=500,
                detail=f"knowledge_entries missing columns: {', '.join(missing_entry_columns)}",
            )

        kconn.execute("DELETE FROM knowledge_entries")

        entry_count = 0
        qa_count = 0
        plain_count = 0

        for row in rows:
            knowledge_type = str(row["knowledge_type"] or "").lower()
            title = build_entry_title(row)
            search_text = build_search_text_from_row(row)
            embedding_json = build_entry_embedding_json(row, search_text)

            source_label = title or normalize_text(row["source_item_id"])

            kconn.execute(
                """
                INSERT INTO knowledge_entries (
                    entry_id,
                    job_id,
                    knowledge_id,
                    knowledge_type,
                    title,
                    question,
                    answer,
                    content,
                    search_text,
                    embedding,
                    source_type,
                    source_item_id,
                    source_label,
                    language,
                    sort_no,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row["knowledge_id"]),
                    job_id,
                    str(row["knowledge_id"]),
                    knowledge_type,
                    title or None,
                    row["question"],
                    row["answer"],
                    row["content"],
                    search_text,
                    embedding_json,
                    row["source_type"],
                    row["source_item_id"],
                    source_label or None,
                    row["language"],
                    row["sort_no"],
                    now_jst_iso(),
                    now_jst_iso(),
                ),
            )

            entry_count += 1
            if knowledge_type == "qa":
                qa_count += 1
            elif knowledge_type == "plain":
                plain_count += 1

        if get_existing_table_name(kconn, ["knowledge_fts"]):
            kconn.execute("INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild')")

        if get_existing_table_name(kconn, ["knowledge_meta"]):
            upsert_knowledge_meta(kconn, "db_version", "1")
            upsert_knowledge_meta(kconn, "built_at", now_jst_iso())
            upsert_knowledge_meta(kconn, "source_job_id", job_id)
            upsert_knowledge_meta(kconn, "entry_count", str(entry_count))
            upsert_knowledge_meta(kconn, "qa_count", str(qa_count))
            upsert_knowledge_meta(kconn, "plain_count", str(plain_count))

        kconn.commit()

    except Exception:
        kconn.rollback()
        raise

    finally:
        kconn.close()

    gcs_path = upload_knowledge_db(uid, local_knowledge_db_path, filename)

    job_row = ensure_job_exists(conn, job_id)
    source_type = str(job_row["source_type"] or "").strip()
    if not source_type:
        raise HTTPException(
            status_code=500,
            detail=f"source_type not found for job_id={job_id}",
        )

    insert_knowledge_db_row(
        conn=conn,
        database_name=filename,
        source_type=source_type,
    )

    return {
        "filename": filename,
        "gcs_path": gcs_path,
        "entry_count": entry_count,
        "qa_count": qa_count,
        "plain_count": plain_count,
        "source_type": source_type,
    }


def get_existing_table_name(conn: sqlite3.Connection, candidates: list[str]) -> Optional[str]:
    for table_name in candidates:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (table_name,),
        )
        if cur.fetchone():
            return table_name
    return None


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    rows = cur.fetchall()
    return {row["name"] for row in rows}


def ensure_job_exists(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    cur = conn.execute(
        """
        SELECT
            job_id,
            source_type,
            source_name,
            request_type,
            status,
            phase,
            selected_count,
            qa_count,
            plain_count,
            error_count,
            requested_at,
            started_at,
            finished_at,
            error_message
        FROM knowledge_jobs
        WHERE job_id = ?
        """,
        (job_id,),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return row


def update_job_status(
    conn: sqlite3.Connection,
    job_id: str,
    new_status: str,
) -> None:
    columns = get_table_columns(conn, "knowledge_jobs")
    now_text = now_jst_iso()

    sets: list[str] = ["status = ?"]
    params: list[object] = [new_status]

    if "started_at" in columns:
        sets.append("started_at = COALESCE(started_at, ?)")
        params.append(now_text)

    if "finished_at" in columns:
        sets.append("finished_at = ?")
        params.append(now_text)

    if "error_message" in columns:
        sets.append("error_message = NULL")

    sql = f"""
        UPDATE knowledge_jobs
        SET {", ".join(sets)}
        WHERE job_id = ?
    """
    params.append(job_id)
    conn.execute(sql, params)


def update_job_phase(
    conn: sqlite3.Connection,
    job_id: str,
    new_phase: str,
) -> None:
    columns = get_table_columns(conn, "knowledge_jobs")
    if "phase" not in columns:
        return

    conn.execute(
        """
        UPDATE knowledge_jobs
        SET phase = ?
        WHERE job_id = ?
        """,
        (new_phase, job_id),
    )


def update_item_statuses_for_job(
    conn: sqlite3.Connection,
    job_id: str,
    new_status: str,
) -> int:
    table_name = get_existing_table_name(
        conn,
        ["knowledge_items", "knowledge_job_items", "job_items"],
    )
    if not table_name:
        return 0

    columns = get_table_columns(conn, table_name)
    if "job_id" not in columns or "status" not in columns:
        return 0

    now_text = now_jst_iso()
    sets: list[str] = ["status = ?"]
    params: list[object] = [new_status]

    if "started_at" in columns:
        sets.append("started_at = COALESCE(started_at, ?)")
        params.append(now_text)

    if "finished_at" in columns:
        sets.append("finished_at = ?")
        params.append(now_text)

    if "error_message" in columns:
        sets.append("error_message = NULL")

    sql = f"""
        UPDATE {table_name}
        SET {", ".join(sets)}
        WHERE job_id = ?
    """
    params.append(job_id)

    cur = conn.execute(sql, params)
    return cur.rowcount or 0


def vectorize_knowledge_items_for_job(
    conn: sqlite3.Connection,
    job_id: str,
) -> int:
    table_name = get_existing_table_name(conn, ["knowledge_items"])
    if not table_name:
        raise HTTPException(status_code=404, detail="knowledge_items table not found")

    columns = get_table_columns(conn, table_name)
    required_columns = {
        "knowledge_id",
        "job_id",
        "question_normalize",
        "answer_normalize",
        "content_normalize",
        "question_vector",
        "answer_vector",
        "content_vector",
    }
    missing = [c for c in required_columns if c not in columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"knowledge_items missing columns: {', '.join(missing)}",
        )

    update_sets = []
    if "updated_at" in columns:
        update_sets.append("updated_at = ?")
    if "review_status" in columns:
        update_sets.append("review_status = COALESCE(review_status, 'new')")

    updated_count = 0

    cur = conn.execute(
        """
        SELECT
            knowledge_id,
            question_normalize,
            answer_normalize,
            content_normalize
        FROM knowledge_items
        WHERE job_id = ?
        ORDER BY sort_no ASC, knowledge_id ASC
        """,
        (job_id,),
    )
    rows = cur.fetchall()

    for row in rows:
        question_vector, answer_vector, content_vector = embed_normalized_fields(
            question_normalize=row["question_normalize"],
            answer_normalize=row["answer_normalize"],
            content_normalize=row["content_normalize"],
        )

        params: list[object] = [
            question_vector,
            answer_vector,
            content_vector,
        ]

        sql = """
            UPDATE knowledge_items
            SET
                question_vector = ?,
                answer_vector = ?,
                content_vector = ?
        """

        if update_sets:
            sql += ", " + ", ".join(update_sets)
            if "updated_at" in columns:
                params.append(now_jst_iso())

        sql += " WHERE knowledge_id = ?"
        params.append(row["knowledge_id"])

        conn.execute(sql, params)
        updated_count += 1

    return updated_count


def normalize_knowledge_items_for_job(
    conn: sqlite3.Connection,
    job_id: str,
) -> int:
    table_name = get_existing_table_name(conn, ["knowledge_items"])
    if not table_name:
      raise HTTPException(status_code=404, detail="knowledge_items table not found")

    columns = get_table_columns(conn, table_name)
    required_columns = {
        "knowledge_id",
        "job_id",
        "question",
        "answer",
        "content",
        "question_normalize",
        "answer_normalize",
        "content_normalize",
    }
    missing = [c for c in required_columns if c not in columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"knowledge_items missing columns: {', '.join(missing)}",
        )

    updated_count = 0

    cur = conn.execute(
        """
        SELECT
            knowledge_id,
            question,
            answer,
            content
        FROM knowledge_items
        WHERE job_id = ?
        ORDER BY sort_no ASC, knowledge_id ASC
        """,
        (job_id,),
    )
    rows = cur.fetchall()

    for row in rows:
        question_normalize = normalize_text(row["question"])
        answer_normalize = normalize_text(row["answer"])
        content_normalize = normalize_text(row["content"])

        conn.execute(
            """
            UPDATE knowledge_items
            SET
                question_normalize = ?,
                answer_normalize = ?,
                content_normalize = ?,
                updated_at = ?
            WHERE knowledge_id = ?
            """,
            (
                question_normalize or None,
                answer_normalize or None,
                content_normalize or None,
                now_jst_iso(),
                row["knowledge_id"],
            ),
        )
        updated_count += 1

    return updated_count


def parse_vector(value: Optional[str]) -> Optional[list[float]]:
    if not value:
        return None

    try:
        data = json.loads(value)
    except Exception:
        return None

    if not isinstance(data, list) or not data:
        return None

    try:
        return [float(x) for x in data]
    except Exception:
        return None


def mean_vectors(vectors: list[Optional[list[float]]]) -> Optional[list[float]]:
    valid = [v for v in vectors if v]
    if not valid:
        return None

    dim = len(valid[0])
    if any(len(v) != dim for v in valid):
        return None

    result = [0.0] * dim
    for vec in valid:
        for i, x in enumerate(vec):
            result[i] += x

    size = float(len(valid))
    return [x / size for x in result]


def cosine_similarity(vec1: Optional[list[float]], vec2: Optional[list[float]]) -> float:
    if not vec1 or not vec2:
        return -1.0
    if len(vec1) != len(vec2):
        return -1.0

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0

    for a, b in zip(vec1, vec2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b

    if norm1 <= 0.0 or norm2 <= 0.0:
        return -1.0

    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


def build_qa_vector(row: sqlite3.Row) -> Optional[list[float]]:
    qv = parse_vector(row["question_vector"])
    av = parse_vector(row["answer_vector"])
    return mean_vectors([qv, av])


def build_plain_vector(row: sqlite3.Row) -> Optional[list[float]]:
    return parse_vector(row["content_vector"])


def choose_representative(rows: list[sqlite3.Row]) -> sqlite3.Row:
    def score(row: sqlite3.Row) -> tuple[int, int, int, str]:
        question = normalize_text(row["question"] if "question" in row.keys() else "")
        answer = normalize_text(row["answer"] if "answer" in row.keys() else "")
        content = normalize_text(row["content"] if "content" in row.keys() else "")
        total_len = len(question) + len(answer) + len(content)
        has_answer = 1 if answer else 0
        sort_no = int(row["sort_no"] or 0)
        knowledge_id = str(row["knowledge_id"])
        return (has_answer, total_len, -sort_no, knowledge_id)

    return max(rows, key=score)


def build_group_id(job_id: str, knowledge_type: str, seq_no: int) -> str:
    prefix = "QG" if knowledge_type == "qa" else "PG"
    return f"{job_id}_{prefix}_{seq_no:04d}"


def cluster_rows_by_similarity(
    rows: list[sqlite3.Row],
    vector_builder,
    threshold: float,
) -> list[list[sqlite3.Row]]:
    if not rows:
        return []

    parent: dict[str, str] = {str(row["knowledge_id"]): str(row["knowledge_id"]) for row in rows}
    vectors: dict[str, Optional[list[float]]] = {
        str(row["knowledge_id"]): vector_builder(row) for row in rows
    }

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    n = len(rows)
    for i in range(n):
        row_i = rows[i]
        id_i = str(row_i["knowledge_id"])
        vec_i = vectors[id_i]
        if not vec_i:
            continue

        for j in range(i + 1, n):
            row_j = rows[j]
            id_j = str(row_j["knowledge_id"])
            vec_j = vectors[id_j]
            if not vec_j:
                continue

            sim = cosine_similarity(vec_i, vec_j)
            if sim >= threshold:
                union(id_i, id_j)

    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        root = find(str(row["knowledge_id"]))
        grouped.setdefault(root, []).append(row)

    return list(grouped.values())


def mark_single_item(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    dedup_version: int,
) -> None:
    conn.execute(
        """
        UPDATE knowledge_items
        SET
            review_status = 'single',
            dedup_group_id = NULL,
            representative_knowledge_id = knowledge_id,
            is_representative = 1,
            is_duplicate = 0,
            dedup_method = NULL,
            dedup_score = NULL,
            dedup_key = NULL,
            dedup_version = ?,
            dedup_at = ?,
            updated_at = ?
        WHERE knowledge_id = ?
        """,
        (
            dedup_version,
            now_jst_iso(),
            now_jst_iso(),
            row["knowledge_id"],
        ),
    )


def mark_group_item(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    review_status: str,
    dedup_group_id: str,
    representative_knowledge_id: str,
    is_representative: int,
    is_duplicate: int,
    dedup_method: str,
    dedup_score: Optional[float],
    dedup_key: Optional[str],
    dedup_version: int,
) -> None:
    conn.execute(
        """
        UPDATE knowledge_items
        SET
            review_status = ?,
            dedup_group_id = ?,
            representative_knowledge_id = ?,
            is_representative = ?,
            is_duplicate = ?,
            dedup_method = ?,
            dedup_score = ?,
            dedup_key = ?,
            dedup_version = ?,
            dedup_at = ?,
            updated_at = ?
        WHERE knowledge_id = ?
        """,
        (
            review_status,
            dedup_group_id,
            representative_knowledge_id,
            is_representative,
            is_duplicate,
            dedup_method,
            dedup_score,
            dedup_key,
            dedup_version,
            now_jst_iso(),
            now_jst_iso(),
            row["knowledge_id"],
        ),
    )


def get_next_dedup_version(conn: sqlite3.Connection, job_id: str) -> int:
    cur = conn.execute(
        """
        SELECT COALESCE(MAX(dedup_version), 0) AS max_ver
        FROM knowledge_items
        WHERE job_id = ?
        """,
        (job_id,),
    )
    row = cur.fetchone()
    max_ver = int(row["max_ver"] or 0)
    return max_ver + 1


def deduplicate_knowledge_items_for_job(
    conn: sqlite3.Connection,
    job_id: str,
    qa_threshold: float = QA_DEDUP_THRESHOLD,
    plain_threshold: float = PLAIN_DEDUP_THRESHOLD,
) -> dict[str, int]:
    columns = get_table_columns(conn, "knowledge_items")
    required_columns = {
        "knowledge_id",
        "job_id",
        "job_item_id",
        "knowledge_type",
        "question",
        "answer",
        "content",
        "question_vector",
        "answer_vector",
        "content_vector",
        "sort_no",
        "status",
        "review_status",
        "dedup_group_id",
        "representative_knowledge_id",
        "is_representative",
        "is_duplicate",
        "dedup_method",
        "dedup_score",
        "dedup_key",
        "dedup_version",
        "dedup_at",
        "updated_at",
    }
    missing = [c for c in required_columns if c not in columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"knowledge_items missing columns: {', '.join(missing)}",
        )

    cur = conn.execute(
        """
        SELECT
            knowledge_id,
            job_id,
            job_item_id,
            source_type,
            source_id,
            source_item_id,
            row_id,
            knowledge_type,
            title,
            question,
            answer,
            content,
            question_normalize,
            answer_normalize,
            content_normalize,
            question_vector,
            answer_vector,
            content_vector,
            summary,
            keywords,
            language,
            sort_no,
            status,
            review_status,
            dedup_group_id,
            representative_knowledge_id,
            is_representative,
            is_duplicate,
            dedup_method,
            dedup_score,
            dedup_key,
            dedup_version,
            dedup_at,
            created_at,
            updated_at
        FROM knowledge_items
        WHERE job_id = ?
        ORDER BY sort_no ASC, knowledge_id ASC
        """,
        (job_id,),
    )
    rows = cur.fetchall()

    qa_rows = [row for row in rows if str(row["knowledge_type"] or "") == "qa"]
    plain_rows = [row for row in rows if str(row["knowledge_type"] or "") == "plain"]

    qa_groups = cluster_rows_by_similarity(
        qa_rows,
        vector_builder=build_qa_vector,
        threshold=qa_threshold,
    )
    plain_groups = cluster_rows_by_similarity(
        plain_rows,
        vector_builder=build_plain_vector,
        threshold=plain_threshold,
    )

    dedup_version = get_next_dedup_version(conn, job_id)

    group_seq = 0
    group_count = 0
    representative_count = 0
    duplicate_count = 0
    qa_count = 0
    plain_count = 0

    for group in qa_groups:
        if not group:
            continue

        if len(group) == 1:
            mark_single_item(conn, group[0], dedup_version=dedup_version)
            representative_count += 1
            qa_count += 1
            continue

        group_seq += 1
        group_count += 1

        representative = choose_representative(group)
        representative_id = str(representative["knowledge_id"])
        group_id = build_group_id(job_id, "qa", group_seq)
        rep_vector = build_qa_vector(representative)

        for row in group:
            row_vector = build_qa_vector(row)
            sim = cosine_similarity(rep_vector, row_vector)
            is_rep = 1 if str(row["knowledge_id"]) == representative_id else 0

            if is_rep:
                mark_group_item(
                    conn,
                    row,
                    review_status="representative",
                    dedup_group_id=group_id,
                    representative_knowledge_id=representative_id,
                    is_representative=1,
                    is_duplicate=0,
                    dedup_method="qa_vector",
                    dedup_score=1.0 if sim < 0 else sim,
                    dedup_key=None,
                    dedup_version=dedup_version,
                )
                representative_count += 1
                qa_count += 1
            else:
                mark_group_item(
                    conn,
                    row,
                    review_status="duplicate",
                    dedup_group_id=group_id,
                    representative_knowledge_id=representative_id,
                    is_representative=0,
                    is_duplicate=1,
                    dedup_method="qa_vector",
                    dedup_score=None if sim < 0 else sim,
                    dedup_key=None,
                    dedup_version=dedup_version,
                )
                duplicate_count += 1

    for group in plain_groups:
        if not group:
            continue

        if len(group) == 1:
            mark_single_item(conn, group[0], dedup_version=dedup_version)
            representative_count += 1
            plain_count += 1
            continue

        group_seq += 1
        group_count += 1

        representative = choose_representative(group)
        representative_id = str(representative["knowledge_id"])
        group_id = build_group_id(job_id, "plain", group_seq)
        rep_vector = build_plain_vector(representative)

        for row in group:
            row_vector = build_plain_vector(row)
            sim = cosine_similarity(rep_vector, row_vector)
            dedup_key = normalize_text(row["content_normalize"]) if "content_normalize" in row.keys() else None
            is_rep = 1 if str(row["knowledge_id"]) == representative_id else 0

            if is_rep:
                mark_group_item(
                    conn,
                    row,
                    review_status="representative",
                    dedup_group_id=group_id,
                    representative_knowledge_id=representative_id,
                    is_representative=1,
                    is_duplicate=0,
                    dedup_method="plain_vector",
                    dedup_score=1.0 if sim < 0 else sim,
                    dedup_key=dedup_key,
                    dedup_version=dedup_version,
                )
                representative_count += 1
                plain_count += 1
            else:
                mark_group_item(
                    conn,
                    row,
                    review_status="duplicate",
                    dedup_group_id=group_id,
                    representative_knowledge_id=representative_id,
                    is_representative=0,
                    is_duplicate=1,
                    dedup_method="plain_vector",
                    dedup_score=None if sim < 0 else sim,
                    dedup_key=dedup_key,
                    dedup_version=dedup_version,
                )
                duplicate_count += 1

    conn.execute(
        """
        UPDATE knowledge_jobs
        SET
            qa_count = ?,
            plain_count = ?
        WHERE job_id = ?
        """,
        (qa_count, plain_count, job_id),
    )

    return {
        "group_count": group_count,
        "representative_count": representative_count,
        "duplicate_count": duplicate_count,
        "qa_count": qa_count,
        "plain_count": plain_count,
        "dedup_version": dedup_version,
    }


@router.get("/jobs", response_model=RefineJobListResponse)
def list_refine_jobs(
    authorization: str | None = Header(default=None),
    source_type: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    uid = get_uid_from_auth_header(authorization)

    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        prefix = f"users/{uid}/job_status/"
        blobs = client.list_blobs(bucket, prefix=prefix)

        jobs: list[RefineJobRow] = []

        for blob in blobs:
            if not blob.name.endswith(".json"):
                continue

            data = json.loads(blob.download_as_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue

            base_status = str(data.get("status") or "").lower()
            error_chunks = int(data.get("error_chunks") or 0)
            if base_status != "done" or error_chunks != 0:
                continue

            refine_status = str(data.get("refine_status") or "new")
            refine_phase = data.get("refine_phase")

            if source_type and data.get("source_type") != source_type:
                continue
            if status and refine_status != status:
                continue

            jobs.append(
                RefineJobRow(
                    job_id=str(data.get("job_id") or ""),
                    source_type=str(data.get("source_type") or ""),
                    source_name=data.get("source_name"),
                    request_type=data.get("request_type"),
                    status=refine_status,
                    phase=refine_phase,
                    selected_count=int(data.get("selected_count") or 0),
                    qa_count=int(data.get("qa_count") or 0),
                    plain_count=int(data.get("plain_count") or 0),
                    error_count=1 if refine_status == "error" else 0,
                    requested_at=data.get("refine_requested_at") or data.get("requested_at"),
                    started_at=data.get("refine_started_at"),
                    finished_at=data.get("refine_finished_at"),
                    error_message=data.get("refine_error_message"),
                )
            )

        jobs.sort(key=lambda x: x.requested_at or "", reverse=True)
        jobs = jobs[:limit]

        return RefineJobListResponse(
            jobs=jobs,
            total_count=len(jobs),
        )

    except Exception as e:
        logger.exception("list_refine_jobs failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/items", response_model=RefineJobItemListResponse)
def list_refine_job_items(
    job_id: str,
    authorization: str | None = Header(default=None),
    limit: int = Query(default=500, ge=1, le=2000),
):
    uid = get_uid_from_auth_header(authorization)
    update_job_status_json(uid, job_id, {
        "refine_status": "running",
        "refine_phase": "cleanse",
        "refine_requested_at": read_job_status_json(uid, job_id).get("refine_requested_at") or now_jst_iso(),
        "refine_started_at": now_jst_iso(),
        "refine_finished_at": None,
        "refine_error_message": None,
    })
    local_db_path = download_job_db(uid, job_id)

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row

    try:
        table_name = get_existing_table_name(
            conn,
            ["knowledge_job_items", "job_items", "knowledge_items"],
        )
        if not table_name:
            raise HTTPException(
                status_code=404,
                detail="job item table not found. expected one of: knowledge_job_items, job_items, knowledge_items",
            )

        columns = get_table_columns(conn, table_name)
        if "job_id" not in columns:
            raise HTTPException(
                status_code=500,
                detail=f"{table_name} does not have job_id column",
            )

        select_parts = []

        if "job_item_id" in columns:
            select_parts.append("job_item_id")
        else:
            select_parts.append("rowid AS job_item_id")

        select_parts.append("job_id")

        if "source_item_id" in columns:
            select_parts.append("source_item_id")
        else:
            select_parts.append("NULL AS source_item_id")

        if "source_type" in columns:
            select_parts.append("source_type")
        else:
            select_parts.append("NULL AS source_type")

        if "title" in columns:
            select_parts.append("title")
        elif "source_title" in columns:
            select_parts.append("source_title AS title")
        elif "row_title" in columns:
            select_parts.append("row_title AS title")
        elif "display_title" in columns:
            select_parts.append("display_title AS title")
        else:
            select_parts.append("NULL AS title")

        if "status" in columns:
            select_parts.append("status")
        else:
            select_parts.append("'new' AS status")

        if "qa_count" in columns:
            select_parts.append("qa_count")
        else:
            select_parts.append("0 AS qa_count")

        if "plain_count" in columns:
            select_parts.append("plain_count")
        else:
            select_parts.append("0 AS plain_count")

        if "error_count" in columns:
            select_parts.append("error_count")
        else:
            select_parts.append("0 AS error_count")

        if "error_message" in columns:
            select_parts.append("error_message")
        else:
            select_parts.append("NULL AS error_message")

        if "requested_at" in columns:
            select_parts.append("requested_at")
        else:
            select_parts.append("NULL AS requested_at")

        if "started_at" in columns:
            select_parts.append("started_at")
        else:
            select_parts.append("NULL AS started_at")

        if "finished_at" in columns:
            select_parts.append("finished_at")
        else:
            select_parts.append("NULL AS finished_at")

        order_by = []
        if "requested_at" in columns:
            order_by.append("requested_at DESC")
        if "row_index" in columns:
            order_by.append("row_index ASC")
        if "job_item_id" in columns:
            order_by.append("job_item_id ASC")
        else:
            order_by.append("rowid ASC")

        sql = f"""
            SELECT
                {", ".join(select_parts)}
            FROM {table_name}
            WHERE job_id = ?
            ORDER BY {", ".join(order_by)}
            LIMIT ?
        """

        cur = conn.execute(sql, (job_id, limit))
        rows = cur.fetchall()

        items = [
            RefineJobItemRow(
                job_item_id=str(row["job_item_id"]),
                job_id=row["job_id"],
                source_item_id=row["source_item_id"],
                source_type=row["source_type"],
                title=row["title"],
                status=row["status"] or "new",
                qa_count=row["qa_count"] or 0,
                plain_count=row["plain_count"] or 0,
                error_count=row["error_count"] or 0,
                error_message=row["error_message"],
                requested_at=row["requested_at"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
            )
            for row in rows
        ]

        return RefineJobItemListResponse(
            job_id=job_id,
            items=items,
            total_count=len(items),
        )

    except sqlite3.Error as e:
        logger.exception("list_refine_job_items sqlite error")
        raise HTTPException(status_code=500, detail=f"sqlite error: {e}")

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("list_refine_job_items failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()


@router.post("/jobs/{job_id}/cleanse", response_model=RefineActionResponse)
def normalize_refine_job(
    job_id: str,
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)
    local_db_path = download_job_db(uid, job_id)

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row

    try:
        print("=== CLEANSE START ===")
        print("JOB_ID:", job_id)
        print("DB_PATH:", local_db_path)

        job_row = ensure_job_exists(conn, job_id)
        print("STATUS BEFORE:", job_row["status"], job_row["phase"])

        current_status = str(job_row["status"] or "").lower()
        if current_status not in {"new", "ready", "done"}:
            raise HTTPException(
                status_code=409,
                detail=f"normalize is not allowed for status={job_row['status']}",
            )

        row_count = conn.execute(
            "SELECT COUNT(*) FROM knowledge_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()[0]
        print("MATCH COUNT:", row_count)

        normalized_count = normalize_knowledge_items_for_job(conn, job_id)

        columns = get_table_columns(conn, "knowledge_jobs")
        print("COLUMNS:", sorted(list(columns)))

        print("=== UPDATE PHASE ===")
        update_job_phase(conn, job_id, "cleansed")

        print("=== UPDATE STATUS ===")
        update_job_status(conn, job_id, "done")

        print("=== UPDATE ITEMS ===")
        update_item_statuses_for_job(conn, job_id, "done")

        after_row = conn.execute(
            "SELECT status, phase FROM knowledge_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        print("AFTER UPDATE:", after_row["status"], after_row["phase"])

        conn.commit()
        print("COMMIT DONE")

        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        print("JOURNAL MODE:", journal_mode)

        if str(journal_mode).lower() == "wal":
            conn.execute("PRAGMA wal_checkpoint(FULL)")
            print("WAL CHECKPOINT DONE")

        conn.close()
        conn = None
        print("CONN CLOSED")

        upload_job_db(uid, job_id, local_db_path)
        print("UPLOAD DONE")

        print("=== CLEANSE END ===")

        update_job_status_json(uid, job_id, {
            "refine_status": "done",
            "refine_phase": "cleansed",
            "refine_finished_at": now_jst_iso(),
            "refine_error_message": None,
        })

        return RefineActionResponse(
            ok=True,
            job_id=job_id,
            action="cleanse",
            status="done",
            message=f"cleansed: {normalized_count}",
        )

    except HTTPException as e:
        update_job_status_json(uid, job_id, {
            "refine_status": "error",
            "refine_error_message": e.detail if hasattr(e, "detail") else str(e),
            "refine_finished_at": now_jst_iso(),
        })
        if conn is not None:
            conn.close()
            conn = None
        raise

    except sqlite3.Error as e:
        if conn is not None:
            conn.rollback()
            conn.close()
            conn = None
        print("SQLITE ERROR:", e)
        update_job_status_json(uid, job_id, {
            "refine_status": "error",
            "refine_error_message": f"sqlite error: {e}",
            "refine_finished_at": now_jst_iso(),
        })
        raise HTTPException(status_code=500, detail=f"sqlite error: {e}")

    except Exception as e:
        if conn is not None:
            conn.rollback()
            conn.close()
            conn = None
        print("GENERAL ERROR:", e)
        update_job_status_json(uid, job_id, {
            "refine_status": "error",
            "refine_error_message": str(e),
            "refine_finished_at": now_jst_iso(),
        })
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if conn is not None:
            conn.close()


@router.post("/jobs/{job_id}/vectorize", response_model=RefineActionResponse)
def vectorize_refine_job(
    job_id: str,
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)
    update_job_status_json(uid, job_id, {
        "refine_status": "running",
        "refine_phase": "vectorize",
        "refine_requested_at": read_job_status_json(uid, job_id).get("refine_requested_at") or now_jst_iso(),
        "refine_started_at": now_jst_iso(),
        "refine_finished_at": None,
        "refine_error_message": None,
    })
    local_db_path = download_job_db(uid, job_id)

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row

    try:
        print("=== VECTORIZe START ===")
        print("JOB_ID:", job_id)
        print("DB_PATH:", local_db_path)

        job_row = ensure_job_exists(conn, job_id)
        print("STATUS BEFORE:", job_row["status"], job_row["phase"])

        current_phase = str(job_row["phase"] or "created").lower()
        if current_phase != "cleansed":
            raise HTTPException(
                status_code=409,
                detail=f"vectorize is not allowed for phase={job_row['phase']}",
            )

        vectorized_count = vectorize_knowledge_items_for_job(conn, job_id)

        update_job_phase(conn, job_id, "vectorized")
        update_job_status(conn, job_id, "done")
        update_item_statuses_for_job(conn, job_id, "done")

        after_row = conn.execute(
            "SELECT status, phase FROM knowledge_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        print("AFTER UPDATE:", after_row["status"], after_row["phase"])

        conn.commit()
        print("COMMIT DONE")

        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        print("JOURNAL MODE:", journal_mode)

        if str(journal_mode).lower() == "wal":
            conn.execute("PRAGMA wal_checkpoint(FULL)")
            print("WAL CHECKPOINT DONE")

        conn.close()
        conn = None
        print("CONN CLOSED")

        upload_job_db(uid, job_id, local_db_path)
        print("UPLOAD DONE")

        print("=== VECTORIZE END ===")

        update_job_status_json(uid, job_id, {
            "refine_status": "done",
            "refine_phase": "vectorized",
            "refine_finished_at": now_jst_iso(),
            "refine_error_message": None,
        })

        return RefineActionResponse(
            ok=True,
            job_id=job_id,
            action="vectorize",
            status="done",
            message=f"vectorized: {vectorized_count}",
        )

    except HTTPException as e:
        update_job_status_json(uid, job_id, {
            "refine_status": "error",
            "refine_error_message": e.detail if hasattr(e, "detail") else str(e),
            "refine_finished_at": now_jst_iso(),
        })
        if conn is not None:
            conn.close()
            conn = None
        raise

    except sqlite3.Error as e:
        if conn is not None:
            conn.rollback()
            conn.close()
            conn = None
        print("SQLITE ERROR:", e)
        update_job_status_json(uid, job_id, {
            "refine_status": "error",
            "refine_error_message": f"sqlite error: {e}",
            "refine_finished_at": now_jst_iso(),
        })
        raise HTTPException(status_code=500, detail=f"sqlite error: {e}")

    except Exception as e:
        if conn is not None:
            conn.rollback()
            conn.close()
            conn = None
        print("GENERAL ERROR:", e)
        update_job_status_json(uid, job_id, {
            "refine_status": "error",
            "refine_error_message": str(e),
            "refine_finished_at": now_jst_iso(),
        })
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if conn is not None:
            conn.close()


@router.post("/jobs/{job_id}/deduplicate", response_model=RefineActionResponse)
def deduplicate_refine_job(
    job_id: str,
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)
    update_job_status_json(uid, job_id, {
        "refine_status": "running",
        "refine_phase": "deduplicate",
        "refine_requested_at": read_job_status_json(uid, job_id).get("refine_requested_at") or now_jst_iso(),
        "refine_started_at": now_jst_iso(),
        "refine_finished_at": None,
        "refine_error_message": None,
    })
    local_db_path = download_job_db(uid, job_id)

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row

    try:
        print("=== DEDUPLICATE START ===")
        print("JOB_ID:", job_id)
        print("DB_PATH:", local_db_path)

        job_row = ensure_job_exists(conn, job_id)
        print("STATUS BEFORE:", job_row["status"], job_row["phase"])

        current_phase = str(job_row["phase"] or "created").lower()
        if current_phase != "vectorized":
            raise HTTPException(
                status_code=409,
                detail=f"deduplicate is not allowed for phase={job_row['phase']}",
            )

        result = deduplicate_knowledge_items_for_job(
            conn=conn,
            job_id=job_id,
            qa_threshold=QA_DEDUP_THRESHOLD,
            plain_threshold=PLAIN_DEDUP_THRESHOLD,
        )

        update_job_phase(conn, job_id, "deduplicated")
        update_job_status(conn, job_id, "done")
        update_item_statuses_for_job(conn, job_id, "done")

        after_row = conn.execute(
            "SELECT status, phase FROM knowledge_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        print("AFTER UPDATE:", after_row["status"], after_row["phase"])

        conn.commit()
        print("COMMIT DONE")

        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        print("JOURNAL MODE:", journal_mode)

        if str(journal_mode).lower() == "wal":
            conn.execute("PRAGMA wal_checkpoint(FULL)")
            print("WAL CHECKPOINT DONE")

        conn.close()
        conn = None
        print("CONN CLOSED")

        upload_job_db(uid, job_id, local_db_path)
        print("UPLOAD DONE")

        print("=== DEDUPLICATE END ===")

        update_job_status_json(uid, job_id, {
            "refine_status": "done",
            "refine_phase": "deduplicated",
            "refine_finished_at": now_jst_iso(),
            "refine_error_message": None,
        })

        return RefineActionResponse(
            ok=True,
            job_id=job_id,
            action="deduplicate",
            status="done",
            message=(
                "deduplicated: "
                f"groups={result['group_count']}, "
                f"representatives={result['representative_count']}, "
                f"duplicates={result['duplicate_count']}, "
                f"qa={result['qa_count']}, "
                f"plain={result['plain_count']}, "
                f"version={result['dedup_version']}"
            ),
        )

    except HTTPException as e:
        update_job_status_json(uid, job_id, {
            "refine_status": "error",
            "refine_error_message": e.detail if hasattr(e, "detail") else str(e),
            "refine_finished_at": now_jst_iso(),
        })
        if conn is not None:
            conn.close()
            conn = None
        raise

    except sqlite3.Error as e:
        if conn is not None:
            conn.rollback()
            conn.close()
            conn = None
        print("SQLITE ERROR:", e)
        update_job_status_json(uid, job_id, {
            "refine_status": "error",
            "refine_error_message": f"sqlite error: {e}",
            "refine_finished_at": now_jst_iso(),
        })
        raise HTTPException(status_code=500, detail=f"sqlite error: {e}")

    except Exception as e:
        if conn is not None:
            conn.rollback()
            conn.close()
            conn = None
        print("GENERAL ERROR:", e)
        update_job_status_json(uid, job_id, {
            "refine_status": "error",
            "refine_error_message": str(e),
            "refine_finished_at": now_jst_iso(),
        })
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if conn is not None:
            conn.close()


@router.post("/jobs/{job_id}/build-knowledge-db", response_model=RefineActionResponse)
def build_knowledge_db_job(
    job_id: str,
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)
    update_job_status_json(uid, job_id, {
        "refine_status": "running",
        "refine_phase": "build-knowledge-db",
        "refine_requested_at": read_job_status_json(uid, job_id).get("refine_requested_at") or now_jst_iso(),
        "refine_started_at": now_jst_iso(),
        "refine_finished_at": None,
        "refine_error_message": None,
    })
    local_db_path = download_job_db(uid, job_id)

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row

    try:
        print("=== BUILD KNOWLEDGE DB START ===")
        print("JOB_ID:", job_id)
        print("DB_PATH:", local_db_path)

        job_row = ensure_job_exists(conn, job_id)
        print("STATUS BEFORE:", job_row["status"], job_row["phase"])

        current_phase = str(job_row["phase"] or "created").lower()
        if current_phase != "deduplicated":
            raise HTTPException(
                status_code=409,
                detail=f"build-knowledge-db is not allowed for phase={job_row['phase']}",
            )

        result = build_knowledge_db_for_job(conn, uid, job_id)

        update_job_phase(conn, job_id, "built")
        update_job_status(conn, job_id, "done")
        update_item_statuses_for_job(conn, job_id, "done")

        after_row = conn.execute(
            "SELECT status, phase FROM knowledge_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        print("AFTER UPDATE:", after_row["status"], after_row["phase"])

        conn.commit()
        print("COMMIT DONE")

        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        print("JOURNAL MODE:", journal_mode)

        if str(journal_mode).lower() == "wal":
            conn.execute("PRAGMA wal_checkpoint(FULL)")
            print("WAL CHECKPOINT DONE")

        conn.close()
        conn = None
        print("CONN CLOSED")

        upload_job_db(uid, job_id, local_db_path)
        print("UPLOAD DONE")

        print("=== BUILD KNOWLEDGE DB END ===")

        update_job_status_json(uid, job_id, {
            "refine_status": "done",
            "refine_phase": "built",
            "refine_finished_at": now_jst_iso(),
            "refine_error_message": None,
            "knowledge_db_filename": result.get("filename"),
            "knowledge_db_gcs_path": result.get("gcs_path"),
        })

        return RefineActionResponse(
            ok=True,
            job_id=job_id,
            action="build-knowledge-db",
            status="done",
            message=(
                f"knowledge db created: {result['filename']} / "
                f"entries={result['entry_count']} / "
                f"qa={result['qa_count']} / "
                f"plain={result['plain_count']}"
            ),
        )

    except HTTPException as e:
        update_job_status_json(uid, job_id, {
            "refine_status": "error",
            "refine_error_message": e.detail if hasattr(e, "detail") else str(e),
            "refine_finished_at": now_jst_iso(),
        })
        if conn is not None:
            conn.close()
            conn = None
        raise

    except sqlite3.Error as e:
        if conn is not None:
            conn.rollback()
            conn.close()
            conn = None
        print("SQLITE ERROR:", e)
        update_job_status_json(uid, job_id, {
            "refine_status": "error",
            "refine_error_message": f"sqlite error: {e}",
            "refine_finished_at": now_jst_iso(),
        })
        raise HTTPException(status_code=500, detail=f"sqlite error: {e}")

    except Exception as e:
        if conn is not None:
            conn.rollback()
            conn.close()
            conn = None
        print("GENERAL ERROR:", e)
        update_job_status_json(uid, job_id, {
            "refine_status": "error",
            "refine_error_message": str(e),
            "refine_finished_at": now_jst_iso(),
        })
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if conn is not None:
            conn.close()
