# routers/upload_and_register.py
import os
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, UploadFile, File, Header, HTTPException
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth

import ulid


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


@router.post("/upload_and_register")
async def upload_and_register(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    """
    仕様:
    - uidは Authorization Bearer の Firebase IDトークンから取得
    - GCS: users/{uid}/uploads/{file_id}_{original_filename} にアップロード
    - SQLite(ank.db): source_documents に1行INSERT
    - 同名(logical_name)は upload 内で弾く（409）
    - ank.db は GCS: users/{uid}/ank.db を /tmp に落として更新→上書き戻し
    """

    uid = get_uid_from_auth_header(authorization)

    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is empty")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    original_filename = file.filename
    logical_name = original_filename
    ext = original_filename.rsplit(".", 1)[-1] if "." in original_filename else ""

    file_id = str(ulid.new())
    source_type = "upload"
    source_key = None
    source_item_id = logical_name
    source_url = None

    db_blob_path = user_db_path(uid)
    db_blob = bucket.blob(db_blob_path)

    local_db_path = f"/tmp/ank_{uid}_{file_id}.db"

    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path=gs://{BUCKET_NAME}/{db_blob_path}"
        )

    db_blob.download_to_filename(local_db_path)

    upload_blob = None
    try:
        conn = sqlite3.connect(local_db_path)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS source_documents (
                source_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                logical_name TEXT NOT NULL,
                original_name TEXT,
                ext TEXT,
                source_key TEXT,
                source_item_id TEXT,
                source_url TEXT,
                created_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS ix_source_documents_created_at
            ON source_documents(created_at DESC)
        """)

        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS ux_source_documents_item
            ON source_documents(source_type, source_item_id)
        """)

        cur.execute("""
            SELECT 1
            FROM source_documents
            WHERE source_type = ? AND source_item_id = ?
            LIMIT 1
        """, (source_type, source_item_id))
        if cur.fetchone():
            conn.close()
            raise HTTPException(status_code=409, detail="同名ファイルはアップロードできません")

        upload_blob_path = f"users/{uid}/uploads/{file_id}_{original_filename}"
        upload_blob = bucket.blob(upload_blob_path)

        file.file.seek(0)
        upload_blob.upload_from_file(file.file)

        created_at = datetime.now(tz=JST).isoformat()

        try:
            cur.execute(
                """
                INSERT INTO source_documents
                  (source_id, source_type, logical_name, original_name, ext, source_key, source_item_id, source_url, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    source_type,
                    logical_name,
                    original_filename,
                    ext,
                    source_key,
                    source_item_id,
                    source_url,
                    created_at,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            if upload_blob is not None:
                try:
                    upload_blob.delete()
                except Exception:
                    pass
            raise HTTPException(status_code=409, detail="同名ファイルはアップロードできません")

        conn.close()

        db_blob.upload_from_filename(local_db_path)

        return {
            "file_id": file_id,
            "logical_name": logical_name,
            "original_filename": original_filename,
            "ext": ext,
            "created_at": created_at,
            "gcs_path": upload_blob_path,
            "db_gcs_path": db_blob_path,
        }

    finally:
        try:
            if os.path.exists(local_db_path):
                os.remove(local_db_path)
        except Exception:
            pass
