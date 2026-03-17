import os
import sqlite3
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Header, HTTPException
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth

import ulid

from .content_detector import detect_content_kind
from .content_splitter_csv import split_csv_records
from .content_splitter_pdf import split_pdf_records
from .content_splitter_text import split_text_records


router = APIRouter()
JST = ZoneInfo("Asia/Tokyo")
BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")


def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


def ensure_firebase_initialized():
    if firebase_admin._apps:
        return
    firebase_admin.initialize_app(options={"projectId": "ank-firebase"})


def get_uid_from_auth_header(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    token = authorization.replace("Bearer ", "", 1).strip()

    ensure_firebase_initialized()

    try:
        decoded = fb_auth.verify_id_token(token)
        uid = decoded.get("uid")
        if not uid:
            raise HTTPException(status_code=401, detail="uid not found")
        return uid
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid ID token: {e}")


def split_json_records(binary: bytes) -> list[str]:
    text = binary.decode("utf-8", errors="replace")
    obj = json.loads(text)

    if isinstance(obj, list):
        return [json.dumps(r, ensure_ascii=False) for r in obj]

    return [json.dumps(obj, ensure_ascii=False)]


def split_text_block_records(binary: bytes) -> list[str]:
    records = split_text_records(binary)

    return [
        json.dumps(record, ensure_ascii=False)
        for record in records
        if record and record.get("text")
    ]


def split_text_lines(binary: bytes) -> list[str]:
    text = binary.decode("utf-8", errors="replace")
    return [line for line in text.splitlines() if line.strip()]


def build_row_contents(file_bytes: bytes, original_filename: str, ext: str) -> list[str]:
    kind = detect_content_kind(
        filename=original_filename,
        declared_format=ext,
    )

    if kind == "csv":
        records = split_csv_records(file_bytes)
        return [json.dumps(r, ensure_ascii=False) for r in records]

    if kind == "pdf":
        records = split_pdf_records(file_bytes)
        return [json.dumps(r, ensure_ascii=False) for r in records]

    if kind == "json":
        return split_json_records(file_bytes)

    if kind == "text":
        records = split_text_block_records(file_bytes)
        if records:
            return records

    return split_text_lines(file_bytes)


@router.post("/ingest_uploaded_file/{file_id}")
def ingest_uploaded_file(
    file_id: str,
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_blob_path = user_db_path(uid)
    db_blob = bucket.blob(db_blob_path)

    if not db_blob.exists():
        raise HTTPException(status_code=400, detail="ank.db not found")

    local_db_path = f"/tmp/ank_{uid}_{file_id}.db"
    db_blob.download_to_filename(local_db_path)

    try:
        conn = sqlite3.connect(local_db_path)
        cur = conn.cursor()

        cur.execute(
            """
            SELECT logical_name, original_name, ext
            FROM upload_files
            WHERE file_id = ?
            """,
            (file_id,),
        )

        row = cur.fetchone()

        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="file not found")

        logical_name, original_filename, ext = row

        cur.execute("SELECT 1 FROM row_data WHERE file_id = ? LIMIT 1", (file_id,))
        if cur.fetchone():
            conn.close()
            raise HTTPException(status_code=409, detail="already ingested")

        upload_blob_path = f"users/{uid}/uploads/{file_id}_{original_filename}"
        upload_blob = bucket.blob(upload_blob_path)

        if not upload_blob.exists():
            conn.close()
            raise HTTPException(status_code=404, detail="uploaded file not found")

        file_bytes = upload_blob.download_as_bytes()

        try:
            rows = build_row_contents(file_bytes, original_filename, ext)
        except json.JSONDecodeError as e:
            conn.close()
            raise HTTPException(status_code=400, detail=f"json parse error: {e}")
        except Exception as e:
            conn.close()
            raise HTTPException(status_code=400, detail=f"split error: {e}")

        created_at = datetime.now(tz=JST).isoformat()

        for i, content in enumerate(rows):
            cur.execute(
                """
                INSERT INTO row_data
                (row_id, file_id, source_type, source_key, row_index, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(ulid.new()),
                    file_id,
                    "upload",
                    logical_name,
                    i,
                    content,
                    created_at,
                ),
            )

        conn.commit()
        conn.close()

        db_blob.upload_from_filename(local_db_path)

        return {
            "file_id": file_id,
            "logical_name": logical_name,
            "row_count": len(rows),
        }

    finally:
        if os.path.exists(local_db_path):
            try:
                os.remove(local_db_path)
            except Exception:
                pass
