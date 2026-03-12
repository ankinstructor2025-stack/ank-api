from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth


router = APIRouter(prefix="/knowledge", tags=["knowledge"])

JST = ZoneInfo("Asia/Tokyo")
BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
KOKKAI_QA_PROMPT_PATH = "template/kokkai_qa_prompt.txt"

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


def load_json_safe(text: str) -> dict:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def extract_speech_text(content_obj: dict) -> str:
    return normalize_text(content_obj.get("speech"))


def extract_speech_id(content_obj: dict) -> str | None:
    v = content_obj.get("speechID")
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def extract_speaker(content_obj: dict) -> str:
    return normalize_text(content_obj.get("speaker"))


def load_template_text(path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)

    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"{path} not found")

    return blob.download_as_bytes().decode("utf-8")


def build_kokkai_prompt_text_from_contents(
    conn: sqlite3.Connection,
    job_item_id: str,
) -> str:
    cur = conn.execute(
        """
        SELECT
            ji.job_item_id,
            ji.parent_source_id,
            ji.parent_label,
            d.name_of_house,
            d.name_of_meeting,
            d.logical_name
        FROM knowledge_job_items ji
        LEFT JOIN kokkai_documents d
          ON d.source_id = ji.parent_source_id
        WHERE ji.job_item_id = ?
        LIMIT 1
        """,
        (job_item_id,),
    )
    item = cur.fetchone()
    if not item:
        raise HTTPException(status_code=404, detail=f"knowledge_job_items not found: {job_item_id}")

    prompt_template = load_template_text(KOKKAI_QA_PROMPT_PATH)

    cur = conn.execute(
        """
        SELECT
            content_type,
            content_text,
            sort_no
        FROM knowledge_contents
        WHERE job_item_id = ?
        ORDER BY sort_no
        """,
        (job_item_id,),
    )
    rows = cur.fetchall()

    lines: List[str] = []
    for row in rows:
        if (row["content_type"] or "") != "speech":
            continue
        text = normalize_text(row["content_text"])
        if not text:
            continue
        lines.append(f"[{row['sort_no']}] {text}")

    input_text = "\n\n".join(lines).strip()
    if not input_text:
        raise HTTPException(status_code=400, detail=f"knowledge_contents not found: {job_item_id}")

    header = (
        f"対象: {item['name_of_house'] or ''} / {item['name_of_meeting'] or ''}\n"
        f"job_item_id: {item['job_item_id']}\n\n"
    )

    return (
        f"{header}"
        f"{prompt_template}\n\n"
        f"【会議情報】\n"
        f"院: {item['name_of_house'] or ''}\n"
        f"会議名: {item['name_of_meeting'] or ''}\n"
        f"名称: {item['logical_name'] or item['parent_label'] or ''}\n"
        f"job_item_id: {item['job_item_id']}\n\n"
        f"【発言一覧】\n"
        f"{input_text}\n"
    )


def is_questioner(speaker: str) -> bool:
    s = normalize_text(speaker)
    if not s:
        return False
    return (
        s.endswith("君")
        and "委員長" not in s
        and "議長" not in s
        and "大臣" not in s
        and "政府参考人" not in s
        and "参考人" not in s
    )


def is_answerer(speaker: str) -> bool:
    s = normalize_text(speaker)
    if not s:
        return False
    return (
        "政府参考人" in s
        or "参考人" in s
        or "大臣" in s
        or "副大臣" in s
        or "政務官" in s
        or "長官" in s
    )


def insert_qa_candidate_items_from_row_data(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
    source_type: str,
    source_id: str,
) -> int:
    cur = conn.execute(
        """
        SELECT row_id, row_index, content
        FROM row_data
        WHERE source_type = 'kokkai'
          AND file_id = ?
        ORDER BY row_index
        """,
        (source_id,),
    )
    rows = cur.fetchall()

    speeches = []
    for row in rows:
        content_obj = load_json_safe(row["content"] or "")
        speaker = extract_speaker(content_obj)
        speech_text = extract_speech_text(content_obj)
        source_item_id = extract_speech_id(content_obj)

        if not speech_text:
            continue

        speeches.append(
            {
                "row_id": row["row_id"],
                "row_index": row["row_index"],
                "speaker": speaker,
                "speech_text": speech_text,
                "source_item_id": source_item_id,
            }
        )

    inserted_count = 0
    now = now_iso()
    sort_no = 100000

    for i in range(len(speeches) - 1):
        q = speeches[i]
        if not is_questioner(q["speaker"]):
            continue

        answer = None
        for j in range(i + 1, len(speeches)):
            cand = speeches[j]
            if is_questioner(cand["speaker"]):
                break
            if is_answerer(cand["speaker"]):
                answer = cand
                break

        if not answer:
            continue

        question_text = q["speech_text"]
        answer_text = answer["speech_text"]
        merged_text = f"[Q]\n{question_text}\n\n[A]\n{answer_text}"

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
                summary,
                keywords,
                language,
                sort_no,
                status,
                review_status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id(),
                job_id,
                job_item_id,
                source_type,
                source_id,
                q["source_item_id"],
                q["row_id"],
                "qa",
                None,
                question_text,
                answer_text,
                merged_text,
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


def insert_kokkai_contents(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
    source_id: str,
) -> int:
    cur = conn.execute(
        """
        SELECT row_id, row_index, content
        FROM row_data
        WHERE source_type = 'kokkai'
          AND file_id = ?
        ORDER BY row_index
        """,
        (source_id,),
    )
    rows = cur.fetchall()

    inserted_count = 0
    now = now_iso()

    for idx, row in enumerate(rows, start=1):
        content_obj = load_json_safe(row["content"] or "")
        speech_text = extract_speech_text(content_obj)
        if not speech_text:
            continue

        source_item_id = extract_speech_id(content_obj)

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
                "kokkai",
                source_id,
                source_item_id,
                row["row_id"],
                "speech",
                speech_text,
                idx,
                now,
                now,
            ),
        )
        inserted_count += 1

    return inserted_count


def insert_plain_items_from_contents(
    conn: sqlite3.Connection,
    job_id: str,
    job_item_id: str,
    source_type: str,
    source_id: str,
) -> int:
    cur = conn.execute(
        """
        SELECT
            source_item_id,
            row_id,
            content_type,
            content_text,
            sort_no
        FROM knowledge_contents
        WHERE job_id = ?
          AND job_item_id = ?
        ORDER BY sort_no
        """,
        (job_id, job_item_id),
    )
    rows = cur.fetchall()

    inserted_count = 0
    now = now_iso()

    for row in rows:
        if (row["content_type"] or "") != "speech":
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
                summary,
                keywords,
                language,
                sort_no,
                status,
                review_status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id(),
                job_id,
                job_item_id,
                source_type,
                source_id,
                row["source_item_id"],
                row["row_id"],
                "plain",
                None,
                None,
                None,
                row["content_text"],
                None,
                None,
                "ja",
                row["sort_no"],
                "active",
                "new",
                now,
                now,
            ),
        )
        inserted_count += 1

    return inserted_count


class KnowledgeTargetItem(BaseModel):
    source_type: str = Field(..., description="kokkai / opendata / public_url / upload")
    parent_source_id: Optional[str] = None
    parent_key1: Optional[str] = None
    parent_key2: Optional[str] = None
    parent_label: Optional[str] = None
    row_count: int = 0


class KnowledgeJobCreateRequest(BaseModel):
    source_type: str = Field(..., description="kokkai / opendata / public_url / upload")
    source_name: Optional[str] = None
    request_type: str = "extract_knowledge"
    items: List[KnowledgeTargetItem]
    preview_only: bool = False


class PromptPreviewItem(BaseModel):
    job_item_id: str
    parent_source_id: Optional[str] = None
    parent_label: Optional[str] = None
    prompt_text: str


class KnowledgeJobCreateResponse(BaseModel):
    job_id: str
    selected_count: int
    created_item_count: int
    status: str
    prompt_previews: List[PromptPreviewItem] = []


@router.post("/jobs", response_model=KnowledgeJobCreateResponse)
def create_knowledge_job(
    body: KnowledgeJobCreateRequest,
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)

    if not body.items:
        raise HTTPException(status_code=400, detail="items is empty")

    if body.source_type != "kokkai":
        raise HTTPException(
            status_code=400,
            detail="currently only source_type='kokkai' is supported",
        )

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}",
        )

    local_db_path = f"/tmp/ank_{uid}_knowledge.db"
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
                item.source_type or "",
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
                body.source_type,
                body.source_name,
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
                    item.source_type,
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
                contents_count = 0
                plain_count = 0
                qa_count = 0

                if item.source_type == "kokkai":
                    source_id = item.parent_source_id or ""
                    if not source_id:
                        raise HTTPException(status_code=400, detail="parent_source_id is required")

                    contents_count = insert_kokkai_contents(
                        conn=conn,
                        job_id=job_id,
                        job_item_id=job_item_id,
                        source_id=source_id,
                    )

                    prompt_text = build_kokkai_prompt_text_from_contents(
                        conn=conn,
                        job_item_id=job_item_id,
                    )

                    prompt_previews.append(
                        PromptPreviewItem(
                            job_item_id=job_item_id,
                            parent_source_id=item.parent_source_id,
                            parent_label=item.parent_label,
                            prompt_text=prompt_text,
                        )
                    )

                    plain_count = 0
                    qa_count = 0

                    finished_at = now_iso()

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
                            "preview" if body.preview_only else "ready",
                            0,
                            finished_at,
                            job_item_id,
                        ),
                    )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"unsupported item.source_type: {item.source_type}",
                    )


                created_item_count += 1
                total_plain_count += plain_count
                total_qa_count += qa_count

            except Exception as e:
                finished_at = now_iso()
                total_error_count += 1

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
        )

    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise HTTPException(status_code=409, detail=f"duplicate or integrity error: {e}")

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()
