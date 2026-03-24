from fastapi import APIRouter, HTTPException, Header, Query
from fastapi.responses import StreamingResponse
from urllib.parse import quote
import io
import os
import sqlite3

from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth


router = APIRouter(prefix="/upload", tags=["upload"])

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")


def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


def build_upload_blob_path(uid: str, file_id: str, file_name: str) -> str:
    return f"users/{uid}/uploads/{file_id}_{file_name}"


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


def download_user_db(uid: str, suffix: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)

    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail="ank.db not found"
        )

    local_db_path = f"/tmp/ank_{uid}_{suffix}.db"
    db_blob.download_to_filename(local_db_path)
    return local_db_path


def fetch_upload_file_rows(uid: str) -> list[dict]:
    local_db_path = download_user_db(uid, "upload_files")

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
              file_id,
              file_name,
              ext,
              created_at
            FROM upload_files
            ORDER BY created_at DESC
            """
        )
        rows = [dict(r) for r in cur.fetchall()]
        return rows

    finally:
        conn.close()
        try:
            if os.path.exists(local_db_path):
                os.remove(local_db_path)
        except Exception:
            pass


def fetch_upload_file_row(uid: str, file_id: str) -> dict:
    local_db_path = download_user_db(uid, f"upload_file_{file_id}")

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
              file_id,
              file_name,
              ext,
              created_at
            FROM upload_files
            WHERE file_id = ?
            LIMIT 1
            """,
            (file_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="file not found")
        return dict(row)

    finally:
        conn.close()
        try:
            if os.path.exists(local_db_path):
                os.remove(local_db_path)
        except Exception:
            pass


@router.get("/files")
def upload_files(
    authorization: str | None = Header(default=None),
):
    """
    upload_files を親一覧として返す
    row_data は参照しない
    """
    uid = get_uid_from_auth_header(authorization)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    rows = fetch_upload_file_rows(uid)

    files = []
    for row in rows:
        gcs_path = build_upload_blob_path(uid, row["file_id"], row["file_name"])
        blob = bucket.blob(gcs_path)

        file_size = None
        if blob.exists():
            try:
                blob.reload()
                file_size = int(blob.size or 0)
            except Exception:
                file_size = None

        files.append(
            {
                "file_id": row["file_id"],
                "title": row["file_name"],
                "file_name": row["file_name"],
                "ext": row["ext"],
                "created_at": row["created_at"],
                "gcs_path": gcs_path,
                "file_size": file_size,
            }
        )

    return {
        "mode": "files",
        "file_count": len(files),
        "files": files,
    }


@router.get("/download")
def upload_download(
    file_id: str = Query(...),
    authorization: str | None = Header(default=None),
):
    """
    file_id から元ファイルを返す
    """
    uid = get_uid_from_auth_header(authorization)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    row = fetch_upload_file_row(uid, file_id)

    gcs_path = build_upload_blob_path(uid, row["file_id"], row["file_name"])
    blob = bucket.blob(gcs_path)

    if not blob.exists():
        raise HTTPException(status_code=404, detail="uploaded file not found in gcs")

    binary = blob.download_as_bytes()

    file_name = row["file_name"] or f"{file_id}"
    media_type = "application/octet-stream"

    ext = (row.get("ext") or "").lower()
    if ext in ("txt", "text", "log", "md"):
        media_type = "text/plain; charset=utf-8"
    elif ext == "json":
        media_type = "application/json"
    elif ext == "csv":
        media_type = "text/csv; charset=utf-8"
    elif ext == "pdf":
        media_type = "application/pdf"

    # 日本語ファイル名で落ちにくいようにする
    safe_ascii_name = "download.bin"
    if "." in file_name:
        ext2 = file_name.rsplit(".", 1)[-1]
        safe_ascii_name = f"download.{ext2}"

    quoted_name = quote(file_name)

    headers = {
        "Content-Disposition": (
            f"attachment; filename=\"{safe_ascii_name}\"; "
            f"filename*=UTF-8''{quoted_name}"
        )
    }

    return StreamingResponse(
        io.BytesIO(binary),
        media_type=media_type,
        headers=headers,
    )
