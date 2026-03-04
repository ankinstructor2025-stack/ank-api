import os
import sqlite3
import json
import csv
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Header, HTTPException
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


@router.post("/ingest_uploaded_file/{file_id}")
def ingest_uploaded_file(
    file_id: str,
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # -------- ank.db を取得 --------
    db_blob_path = user_db_path(uid)
    db_blob = bucket.blob(db_blob_path)

    if not db_blob.exists():
        raise HTTPException(status_code=400, detail="ank.db not found")

    local_db_path = f"/tmp/ank_{uid}_{file_id}.db"
    db_blob.download_to_filename(local_db_path)

    conn = sqlite3.connect(local_db_path)
    cur = conn.cursor()

    # uploaded_files から対象取得
    cur.execute("""
        SELECT original_filename, ext
        FROM uploaded_files
        WHERE file_id = ?
    """, (file_id,))
    row = cur.fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="file not found")

    original_filename, ext = row

    # -------- row_data テーブル作成 --------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS row_data (
            row_id TEXT PRIMARY KEY,
            file_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_key TEXT,
            row_index INTEGER,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # 二重取り込み防止
    cur.execute("SELECT 1 FROM row_data WHERE file_id = ? LIMIT 1", (file_id,))
    if cur.fetchone():
        conn.close()
        os.remove(local_db_path)
        raise HTTPException(status_code=409, detail="already ingested")

    # -------- GCS uploads から読み込み --------
    upload_blob_path = f"users/{uid}/uploads/{file_id}_{original_filename}"
    upload_blob = bucket.blob(upload_blob_path)

    if not upload_blob.exists():
        conn.close()
        os.remove(local_db_path)
        raise HTTPException(status_code=404, detail="uploaded file not found")

    file_bytes = upload_blob.download_as_bytes()
    text = file_bytes.decode("utf-8", errors="replace")

    # -------- 拡張子別処理 --------
    if ext == "csv":
        reader = csv.DictReader(text.splitlines())
        rows = [json.dumps(r, ensure_ascii=False) for r in reader]
    elif ext == "json":
        obj = json.loads(text)
        if isinstance(obj, list):
            rows = [json.dumps(r, ensure_ascii=False) for r in obj]
        else:
            rows = [json.dumps(obj, ensure_ascii=False)]
    else:
        rows = [line for line in text.splitlines() if line.strip()]

    # -------- INSERT --------
    created_at = datetime.now(tz=JST).isoformat()

    for i, content in enumerate(rows):
        cur.execute("""
            INSERT INTO row_data
            (row_id, file_id, source_type, source_key, row_index, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(ulid.new()),
            file_id,
            "upload",
            None,
            i,
            content,
            created_at
        ))

    conn.commit()
    conn.close()

    # DBをGCSへ戻す
    db_blob.upload_from_filename(local_db_path)

    os.remove(local_db_path)

    return {
        "file_id": file_id,
        "row_count": len(rows),
        "db_gcs_path": db_blob_path
    }
