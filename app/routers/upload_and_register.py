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


# user_init.py と合わせる
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
    - SQLite(ank.db): uploaded_files に1行INSERT
    - 同名(logical_name)は弾く（409）
    - ank.db は GCS: users/{uid}/ank.db を /tmp に落として更新→上書き戻し
    """

    uid = get_uid_from_auth_header(authorization)

    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is empty")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    original_filename = file.filename
    logical_name = original_filename  # 同名判定の基準（画面表示名）
    ext = original_filename.rsplit(".", 1)[-1] if "." in original_filename else ""

    file_id = str(ulid.new())

    # -------- ank.db を /tmp に用意 --------
    db_blob_path = user_db_path(uid)
    db_blob = bucket.blob(db_blob_path)

    local_db_path = f"/tmp/ank_{uid}_{file_id}.db"

    # ank.db は user_init で作成される前提だが、念のため無い場合は 400 にする
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path=gs://{BUCKET_NAME}/{db_blob_path}"
        )

    # ダウンロード
    db_blob.download_to_filename(local_db_path)

    upload_blob = None
    try:
        # -------- SQLite 更新 --------
        conn = sqlite3.connect(local_db_path)
        cur = conn.cursor()

        # logical_name UNIQUE
        cur.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_files (
                file_id TEXT PRIMARY KEY,
                logical_name TEXT NOT NULL UNIQUE,
                original_filename TEXT NOT NULL,
                ext TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # 同名チェック（アップロード前）
        cur.execute(
            "SELECT 1 FROM uploaded_files WHERE logical_name = ? LIMIT 1",
            (logical_name,),
        )
        if cur.fetchone():
            conn.close()
            raise HTTPException(status_code=409, detail="同名ファイルはアップロードできません")

        # -------- GCS uploads へアップロード --------
        upload_blob_path = f"users/{uid}/uploads/{file_id}_{original_filename}"
        upload_blob = bucket.blob(upload_blob_path)

        file.file.seek(0)
        upload_blob.upload_from_file(file.file)

        # INSERT
        created_at = datetime.now(tz=JST).isoformat()
        try:
            cur.execute(
                """
                INSERT INTO uploaded_files
                  (file_id, logical_name, original_filename, ext, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (file_id, logical_name, original_filename, ext, created_at),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            # UNIQUE違反（同時実行など）
            conn.close()
            if upload_blob is not None:
                try:
                    upload_blob.delete()
                except Exception:
                    pass
            raise HTTPException(status_code=409, detail="同名ファイルはアップロードできません")

        conn.close()

        # -------- DB を GCS へ上書き --------
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
