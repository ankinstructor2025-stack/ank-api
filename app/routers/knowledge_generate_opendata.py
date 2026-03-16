from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Optional, Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth
from openai import OpenAI


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge/opendata", tags=["knowledge_opendata"])

JST = ZoneInfo("Asia/Tokyo")
BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
SOURCE_TYPE = "opendata"

OPENDATA_QA_PROMPT_PATH = "template/opendata_qa_prompt.txt"
OPENDATA_PLAIN_PROMPT_PATH = "template/opendata_plain_prompt.txt"


DEFAULT_OPENDATA_QA_PROMPT = """あなたは、オープンデータから検索に使えるQAを抽出するアシスタントです。

入力として、同一データセットに属する複数の行データや説明文が与えられます。
内容を読み取り、利用価値のあるQAを抽出してください。

目的は、チャットボットや検索システムで再利用できるナレッジを作ることです。
そのため、表面的な言い換えではなく、意味のある質問と回答の組を作成してください。

出力は必ずJSONオブジェクトで返してください。
形式:
{
  "job_item_id": "...",
  "qa_list": [
    {
      "question": "...",
      "answer": "..."
    }
  ]
}

注意:
- 根拠が弱いものは作らない
- 回答は入力に含まれる情報だけを使う
- 推測で補わない
- 同じ意味のQAを重複して作らない
"""

DEFAULT_OPENDATA_PLAIN_PROMPT = """あなたは、オープンデータから検索に使える説明文を抽出するアシスタントです。

入力として、同一データセットに属する複数の行データや説明文が与えられます。
内容を読み取り、検索や要約に使える平文ナレッジを抽出してください。

出力は必ずJSONオブジェクトで返してください。
形式:
{
  "job_item_id": "...",
  "plain_list": [
    {
      "content": "..."
    }
  ]
}

注意:
- 重要な定義、制度概要、項目説明、集計の意味などを優先する
- 行データの断片をそのまま大量に返さない
- 推測で補わない
- 同じ意味の説明文を重複して作らない
"""


def now_iso() -> str:
    return datetime.now(tz=JST).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


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


def load_json_safe(text: str) -> dict | list | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def flatten_json_like(value: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []

    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            lines.extend(flatten_json_like(v, key))
        return lines

    if isinstance(value, list):
        for idx, item in enumerate(value):
            key = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            lines.extend(flatten_json_like(item, key))
        return lines

    text = normalize_text(value)
    if not text:
        return []

    if prefix:
        return [f"{prefix}: {text}"]
    return [text]


def extract_row_text(content_raw: str | None) -> str:
    src = normalize_text(content_raw)
    if not src:
        return ""

    parsed = load_json_safe(src)
    if parsed is None:
        return src

    lines = flatten_json_like(parsed)
    if not lines:
        return src

    return "\n".join(lines)


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def strip_code_fence(text: str) -> str:
    s = (text or "").strip()

    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    if s.lower().startswith("json"):
        s = s[4:].strip()

    return s


def extract_json_candidate(text: str) -> str:
    s = strip_code_fence(text)

    start_obj = s.find("{")
    start_arr = s.find("[")

    candidates = [x for x in [start_obj, start_arr] if x >= 0]
    if not candidates:
        return s

    start = min(candidates)
    return s[start:].strip()


def parse_llm_json(text: str) -> dict:
    raw = extract_json_candidate(text)

    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    repaired = raw
    repaired = repaired.replace("\r\n", "\n").replace("\r", "\n")
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    repaired = re.sub(
        r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)',
        r'\1"\2"\3',
        repaired,
    )

    try:
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else {}
    except Exception as e:
        logger.error("parse_llm_json failed. raw=%s", raw[:2000])
        logger.error("parse_llm_json repaired=%s", repaired[:2000])
        raise e


def run_llm_json(prompt_text: str, log_prefix: str) -> dict:
    client = get_openai_client()

    try:
        logger.info("%s request start", log_prefix)

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": prompt_text}
            ],
        )

        logger.info("%s response received", log_prefix)

        content = res.choices[0].message.content or ""
        content = content.strip()

        logger.info("%s raw content: %s", log_prefix, content[:2000])

        result = parse_llm_json(content)

        logger.info("%s JSON parse success", log_prefix)
        return result

    except Exception:
        logger.exception("%s generation failed", log_prefix)
        raise


def run_opendata_qa_llm(prompt_text: str) -> dict:
    return run_llm_json(prompt_text, "LLM OPENDATA QA")


def run_opendata_plain_llm(prompt_text: str) -> dict:
    return run_llm_json(prompt_text, "LLM OPENDATA PLAIN")


def load_template_text(path: str, default_text: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)

    if not blob.exists():
        return default_text.strip()

    return blob.download_as_bytes().decode("utf-8").strip()


def insert_qa_items_from_llm_result(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
    source_id: str,
    llm_result: dict,
) -> int:
    qa_list = llm_result.get("qa_list") or []
    if not isinstance(qa_list, list):
        raise Exception("qa_list is not list")

    inserted_count = 0
    now = now_iso()
    sort_no = 200000

    for qa in qa_list:
        if not isinstance(qa, dict):
            continue

        question_raw = normalize_text(qa.get("question"))
        answer_raw = normalize_text(qa.get("answer"))

        if not question_raw or not answer_raw:
            continue

        content_raw = f"[Q]\n{question_raw}\n\n[A]\n{answer_raw}"

        conn.execute(
            """
            INSERT INTO knowledge_items (
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
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id(),
                job_id,
                job_item_id,
                SOURCE_TYPE,
                source_id,
                None,
                None,
                "qa",
                None,
                question_raw,
                answer_raw,
                content_raw,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "ja",
                sort_no,
                "active",
                "new",
                now,
                now,
            ),
        )

        inserted_count += 1
        sort_no += 1

    return inserted_count


def insert_plain_items_from_llm_result(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
    source_id: str,
    llm_result: dict,
) -> int:
    plain_list = llm_result.get("plain_list") or []
    if not isinstance(plain_list, list):
        raise Exception("plain_list is not list")

    inserted_count = 0
    now = now_iso()
    sort_no = 300000

    for item in plain_list:
        if not isinstance(item, dict):
            continue

        content_raw = normalize_text(item.get("content"))
        if not content_raw:
            continue

        conn.execute(
            """
            INSERT INTO knowledge_items (
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
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id(),
                job_id,
                job_item_id,
                SOURCE_TYPE,
                source_id,
                None,
                None,
                "plain",
                None,
                None,
                None,
                content_raw,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "ja",
                sort_no,
                "active",
                "new",
                now,
                now,
            ),
        )

        inserted_count += 1
        sort_no += 1

    return inserted_count


def insert_opendata_contents(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
    source_id: str,
) -> int:
    cur = conn.execute(
        """
        SELECT
            row_id,
            row_index,
            source_item_id,
            content
        FROM row_data
        WHERE source_type = 'opendata'
          AND file_id = ?
        ORDER BY row_index
        """,
        (source_id,),
    )
    rows = cur.fetchall()

    inserted_count = 0
    now = now_iso()

    for idx, row in enumerate(rows, start=1):
        content_text = extract_row_text(row["content"])
        if not content_text:
            continue

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
                source_id,
                row["source_item_id"],
                row["row_id"],
                "row",
                content_text,
                idx,
                now,
                now,
            ),
        )
        inserted_count += 1

    return inserted_count


def build_opendata_prompt_text_from_contents(
    conn: sqlite3.Connection,
    job_item_id: str,
    template_path: str,
    default_template: str,
) -> str:
    cur = conn.execute(
        """
        SELECT
            ji.job_item_id,
            ji.parent_source_id,
            ji.parent_label,
            ji.parent_key1,
            ji.parent_key2,
            ji.row_count
        FROM knowledge_job_items ji
        WHERE ji.job_item_id = ?
        LIMIT 1
        """,
        (job_item_id,),
    )
    item = cur.fetchone()
    if not item:
        raise HTTPException(status_code=404, detail=f"knowledge_job_items not found: {job_item_id}")

    prompt_template = load_template_text(template_path, default_template)

    cur = conn.execute(
        """
        SELECT
            content_type,
            content_text,
            sort_no,
            source_item_id
        FROM knowledge_contents
        WHERE job_item_id = ?
        ORDER BY sort_no
        """,
        (job_item_id,),
    )
    rows = cur.fetchall()

    lines: list[str] = []
    for row in rows:
        text = normalize_text(row["content_text"])
        if not text:
            continue

        source_item_id = normalize_text(row["source_item_id"])
        if source_item_id:
            lines.append(f"[{row['sort_no']}] ({source_item_id}) {text}")
        else:
            lines.append(f"[{row['sort_no']}] {text}")

    input_text = "\n\n".join(lines).strip()
    if not input_text:
        raise HTTPException(status_code=400, detail=f"knowledge_contents not found: {job_item_id}")

    return (
        f"対象: {item['parent_label'] or item['parent_source_id'] or 'オープンデータ'}\n"
        f"job_item_id: {item['job_item_id']}\n\n"
        f"{prompt_template}\n\n"
        f"【データセット情報】\n"
        f"parent_source_id: {item['parent_source_id'] or ''}\n"
        f"parent_key1: {item['parent_key1'] or ''}\n"
        f"parent_key2: {item['parent_key2'] or ''}\n"
        f"parent_label: {item['parent_label'] or ''}\n"
        f"row_count: {item['row_count'] or 0}\n"
        f"job_item_id: {item['job_item_id']}\n\n"
        f"【行データ一覧】\n"
        f"{input_text}\n"
    )


class KnowledgeTargetItem(BaseModel):
    source_type: Optional[str] = Field(default=SOURCE_TYPE, description="opendata")
    parent_source_id: Optional[str] = None
    parent_key1: Optional[str] = None
    parent_key2: Optional[str] = None
    parent_label: Optional[str] = None
    row_count: int = 0


class KnowledgeJobCreateRequest(BaseModel):
    source_type: Optional[str] = Field(default=SOURCE_TYPE, description="opendata")
    source_name: Optional[str] = None
    request_type: str = "extract_knowledge"
    items: List[KnowledgeTargetItem]
    preview_only: bool = False


class PromptPreviewItem(BaseModel):
    job_item_id: str
    parent_source_id: Optional[str] = None
    parent_label: Optional[str] = None
    prompt_type: str
    prompt_text: str


class KnowledgeDebugItem(BaseModel):
    job_item_id: str
    parent_label: Optional[str] = None
    status: str
    qa_count: int = 0
    plain_count: int = 0
    error_message: Optional[str] = None
    llm_result: Optional[Any] = None


class KnowledgeJobCreateResponse(BaseModel):
    job_id: str
    selected_count: int
    created_item_count: int
    status: str
    prompt_previews: List[PromptPreviewItem] = []
    debug_items: List[KnowledgeDebugItem] = []


def validate_request(body: KnowledgeJobCreateRequest) -> None:
    if body.source_type not in (None, "", SOURCE_TYPE):
        raise HTTPException(status_code=400, detail=f"source_type must be '{SOURCE_TYPE}'")
    if not body.items:
        raise HTTPException(status_code=400, detail="items is empty")

    for item in body.items:
        if item.source_type not in (None, "", SOURCE_TYPE):
            raise HTTPException(status_code=400, detail=f"item.source_type must be '{SOURCE_TYPE}'")


@router.post("/job", response_model=KnowledgeJobCreateResponse)
def create_opendata_job(
    body: KnowledgeJobCreateRequest,
    authorization: str | None = Header(default=None),
):
    validate_request(body)

    uid = get_uid_from_auth_header(authorization)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}",
        )

    local_db_path = f"/tmp/ank_{uid}_knowledge_opendata.db"
    db_blob.download_to_filename(local_db_path)

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row

    try:
        job_id = new_id()
        requested_at = now_iso()

        unique_items: List[KnowledgeTargetItem] = []
        seen_keys = set()

        for item in body.items:
            key = (
                SOURCE_TYPE,
                item.parent_source_id or "",
                item.parent_key1 or "",
                item.parent_key2 or "",
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_items.append(item)

        selected_count = len(unique_items)

        conn.execute("BEGIN")

        conn.execute(
            """
            INSERT INTO knowledge_jobs (
                job_id,
                source_type,
                source_name,
                request_type,
                status,
                selected_count,
                qa_count,
                plain_count,
                error_count,
                requested_at,
                started_at,
                finished_at,
                error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, ?, NULL, NULL, NULL)
            """,
            (
                job_id,
                SOURCE_TYPE,
                body.source_name or "オープンデータ",
                body.request_type,
                "preview" if body.preview_only else "running",
                selected_count,
                requested_at,
            ),
        )

        created_item_count = 0
        total_plain_count = 0
        total_qa_count = 0
        total_error_count = 0
        prompt_previews: List[PromptPreviewItem] = []
        debug_items: List[KnowledgeDebugItem] = []

        for item in unique_items:
            job_item_id = new_id()

            conn.execute(
                """
                INSERT INTO knowledge_job_items (
                    job_item_id,
                    job_id,
                    source_type,
                    parent_source_id,
                    parent_key1,
                    parent_key2,
                    parent_label,
                    row_count,
                    status,
                    knowledge_count,
                    error_message,
                    created_at,
                    started_at,
                    finished_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, NULL)
                """,
                (
                    job_item_id,
                    job_id,
                    SOURCE_TYPE,
                    item.parent_source_id,
                    item.parent_key1,
                    item.parent_key2,
                    item.parent_label,
                    item.row_count,
                    "preview" if body.preview_only else "running",
                    requested_at,
                    requested_at,
                ),
            )

            try:
                source_id = item.parent_source_id or ""
                if not source_id:
                    raise HTTPException(status_code=400, detail="parent_source_id is required")

                qa_count = 0
                plain_count = 0
                qa_llm_result_for_debug: Optional[dict] = None
                plain_llm_result_for_debug: Optional[dict] = None

                insert_opendata_contents(
                    conn=conn,
                    job_id=job_id,
                    job_item_id=job_item_id,
                    source_id=source_id,
                )

                qa_prompt_text = build_opendata_prompt_text_from_contents(
                    conn=conn,
                    job_item_id=job_item_id,
                    template_path=OPENDATA_QA_PROMPT_PATH,
                    default_template=DEFAULT_OPENDATA_QA_PROMPT,
                )

                plain_prompt_text = build_opendata_prompt_text_from_contents(
                    conn=conn,
                    job_item_id=job_item_id,
                    template_path=OPENDATA_PLAIN_PROMPT_PATH,
                    default_template=DEFAULT_OPENDATA_PLAIN_PROMPT,
                )

                prompt_previews.append(
                    PromptPreviewItem(
                        job_item_id=job_item_id,
                        parent_source_id=item.parent_source_id,
                        parent_label=item.parent_label,
                        prompt_type="qa",
                        prompt_text=qa_prompt_text,
                    )
                )

                prompt_previews.append(
                    PromptPreviewItem(
                        job_item_id=job_item_id,
                        parent_source_id=item.parent_source_id,
                        parent_label=item.parent_label,
                        prompt_type="plain",
                        prompt_text=plain_prompt_text,
                    )
                )

                if not body.preview_only:
                    qa_llm_result = run_opendata_qa_llm(qa_prompt_text)
                    qa_llm_result_for_debug = qa_llm_result

                    if qa_llm_result.get("job_item_id") not in (None, "", job_item_id):
                        raise Exception("qa job_item_id mismatch")

                    qa_count = insert_qa_items_from_llm_result(
                        conn=conn,
                        job_id=job_id,
                        job_item_id=job_item_id,
                        source_id=source_id,
                        llm_result=qa_llm_result,
                    )

                    plain_llm_result = run_opendata_plain_llm(plain_prompt_text)
                    plain_llm_result_for_debug = plain_llm_result

                    if plain_llm_result.get("job_item_id") not in (None, "", job_item_id):
                        raise Exception("plain job_item_id mismatch")

                    plain_count = insert_plain_items_from_llm_result(
                        conn=conn,
                        job_id=job_id,
                        job_item_id=job_item_id,
                        source_id=source_id,
                        llm_result=plain_llm_result,
                    )

                finished_at = now_iso()
                item_status = "preview" if body.preview_only else "ready"

                conn.execute(
                    """
                    UPDATE knowledge_job_items
                    SET status = ?,
                        knowledge_count = ?,
                        finished_at = ?,
                        error_message = NULL
                    WHERE job_item_id = ?
                    """,
                    (
                        item_status,
                        qa_count + plain_count,
                        finished_at,
                        job_item_id,
                    ),
                )

                debug_items.append(
                    KnowledgeDebugItem(
                        job_item_id=job_item_id,
                        parent_label=item.parent_label,
                        status=item_status,
                        qa_count=qa_count,
                        plain_count=plain_count,
                        error_message=None,
                        llm_result={
                            "qa": qa_llm_result_for_debug,
                            "plain": plain_llm_result_for_debug,
                        },
                    )
                )

                created_item_count += 1
                total_qa_count += qa_count
                total_plain_count += plain_count

            except Exception as e:
                logger.exception("opendata job item failed: job_id=%s job_item_id=%s", job_id, job_item_id)

                finished_at = now_iso()

                conn.execute(
                    """
                    UPDATE knowledge_job_items
                    SET status = ?,
                        finished_at = ?,
                        error_message = ?
                    WHERE job_item_id = ?
                    """,
                    (
                        "error",
                        finished_at,
                        str(e),
                        job_item_id,
                    ),
                )

                raise

        finished_at = now_iso()
        final_status = (
            "partial_error"
            if total_error_count > 0
            else ("preview" if body.preview_only else "ready")
        )

        conn.execute(
            """
            UPDATE knowledge_jobs
            SET status = ?,
                qa_count = ?,
                plain_count = ?,
                error_count = ?,
                started_at = COALESCE(started_at, ?),
                finished_at = ?
            WHERE job_id = ?
            """,
            (
                final_status,
                total_qa_count,
                total_plain_count,
                total_error_count,
                requested_at,
                finished_at,
                job_id,
            ),
        )

        conn.commit()
        db_blob.upload_from_filename(local_db_path)

        return KnowledgeJobCreateResponse(
            job_id=job_id,
            selected_count=selected_count,
            created_item_count=created_item_count,
            status=final_status,
            prompt_previews=prompt_previews,
            debug_items=debug_items,
        )

    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise HTTPException(status_code=409, detail=f"duplicate or integrity error: {e}")

    except Exception as e:
        conn.rollback()
        logger.exception("create_opendata_job failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()
