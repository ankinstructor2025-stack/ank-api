import os
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, UploadFile, File, Header, HTTPException
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth

import ulid


router = APIRouter(prefix="/upload", tags=["upload"])
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

    uid = get_uid_from_auth_header(authorization)

    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is empty")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    original_filename = file.filename
    logical_name = original_filename
    ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else ""

    file_id = str(ulid.new())

    db_blob_path = user_db_path(uid)
    db_blob = bucket.blob(db_blob_path)

    local_db_path = f"/tmp/ank_{uid}_{file_id}.db"

    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found"
        )

    db_blob.download_to_filename(local_db_path)

    upload_blob = None

    try:
        conn = sqlite3.connect(local_db_path)
        cur = conn.cursor()

        # 同名チェック
        cur.execute("""
            SELECT 1
            FROM upload_files
            WHERE original_name = ?
            LIMIT 1
        """, (original_filename,))
        if cur.fetchone():
            conn.close()
            raise HTTPException(status_code=409, detail="同名ファイルはアップロードできません")

        upload_blob_path = f"users/{uid}/uploads/{file_id}_{original_filename}"
        upload_blob = bucket.blob(upload_blob_path)

        file.file.seek(0)
        upload_blob.upload_from_file(file.file)

        created_at = datetime.now(tz=JST).isoformat()

        cur.execute(
            """
            INSERT INTO upload_files
            (file_id, logical_name, original_name, ext, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                file_id,
                logical_name,
                original_filename,
                ext,
                created_at,
            ),
        )

        conn.commit()
        conn.close()

        db_blob.upload_from_filename(local_db_path)

        return {
            "file_id": file_id,
            "logical_name": logical_name,
            "original_filename": original_filename,
            "ext": ext,
            "created_at": created_at,
            "gcs_path": upload_blob_path
        }

    finally:
        try:
            if os.path.exists(local_db_path):
                os.remove(local_db_path)
        except Exception:
            pass
