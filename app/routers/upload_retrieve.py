from fastapi import APIRouter, HTTPException, Header, Query
import os
import sqlite3
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth


router = APIRouter(prefix="/upload", tags=["upload_retrieve"])

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


def download_user_db(uid: str, suffix: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)

    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}",
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
    row_data(source_type='upload') があるものだけ返す
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
              f.file_id,
              f.logical_name,
              f.original_name,
              f.ext,
              COUNT(r.row_id) AS row_count,
              f.created_at
            FROM upload_files f
            INNER JOIN row_data r
              ON r.file_id = f.file_id
             AND r.source_type = 'upload'
            GROUP BY
              f.file_id,
              f.logical_name,
              f.original_name,
              f.ext,
              f.created_at
            ORDER BY f.created_at DESC
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
                "row_count": row["row_count"],
                "created_at": row["created_at"],
            }
        )

    return {
        "mode": "files",
        "file_count": len(files),
        "files": files,
    }


@router.get("/rows")
def upload_rows(
    file_id: str = Query(...),
    authorization: str | None = Header(default=None),
):
    """
    指定 file_id の row_data(source_type='upload') を返す
    """
    uid = get_uid_from_auth_header(authorization)
    local_db_path = download_user_db(uid, f"upload_rows_{file_id}")

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT 1
            FROM upload_files
            WHERE file_id = ?
            LIMIT 1
            """,
            (file_id,),
        )
        exists = cur.fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="file not found")

        cur.execute(
            """
            SELECT
              row_id,
              file_id,
              source_type,
              source_key,
              source_item_id,
              row_index,
              content,
              created_at
            FROM row_data
            WHERE source_type = 'upload'
              AND file_id = ?
            ORDER BY row_index
            """,
            (file_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]

    finally:
        conn.close()
        try:
            if os.path.exists(local_db_path):
                os.remove(local_db_path)
        except Exception:
            pass

    return {
        "rows": rows,
        "count": len(rows),
    }
