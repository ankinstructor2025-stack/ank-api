from fastapi import APIRouter, HTTPException, Header
import os
import sqlite3
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth


router = APIRouter(prefix="/upload", tags=["upload"])

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


def download_user_db(uid: str, suffix: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)

    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found"
        )

    local_db_path = f"/tmp/ank_{uid}_{suffix}.db"
    db_blob.download_to_filename(local_db_path)
    return local_db_path


@router.get("/files")
def upload_files(
    authorization: str | None = Header(default=None),
):
    """
    upload_files を親一覧として返す
    row_data は参照しない
    """
    uid = get_uid_from_auth_header(authorization)
    local_db_path = download_user_db(uid, "upload_files")

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
              file_id,
              logical_name,
              original_name,
              ext,
              created_at
            FROM upload_files
            ORDER BY created_at DESC
            """
        )
        rows = [dict(r) for r in cur.fetchall()]

    finally:
        conn.close()
        try:
            if os.path.exists(local_db_path):
                os.remove(local_db_path)
        except Exception:
            pass

    files = []
    for row in rows:
        files.append(
            {
                "file_id": row["file_id"],
                "title": row["logical_name"],
                "logical_name": row["logical_name"],
                "original_name": row["original_name"],
                "ext": row["ext"],
                "created_at": row["created_at"],
            }
        )

    return {
        "mode": "files",
        "file_count": len(files),
        "files": files,
    }
