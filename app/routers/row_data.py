from fastapi import APIRouter, HTTPException, Header, Query
import os
import sqlite3
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth

router = APIRouter(prefix="/row_data", tags=["row_data"])

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
        raise HTTPException(status_code=401, detail=str(e))


def normalize_source_type(source_type: str | None) -> str | None:
    if not source_type:
        return None

    mapping = {
        "api_kokkai": "kokkai",
        "api_datago": "opendata",
        "url_egov": "egov",
        "url_caa": "caa",
        "file_upload": "upload",
    }
    return mapping.get(source_type, source_type)


@router.get("")
def get_row_data(
    source_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)
    db_source_type = normalize_source_type(source_type)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)

    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}",
        )

    local_db_path = f"/tmp/ank_{uid}_row_data.db"
    db_blob.download_to_filename(local_db_path)

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()

        where_sql = ""
        params: list = []

        if db_source_type:
            where_sql = "WHERE source_type = ?"
            params.append(db_source_type)

        count_sql = f"""
            SELECT COUNT(*) AS total_count
            FROM row_data
            {where_sql}
        """
        total_count = cur.execute(count_sql, params).fetchone()["total_count"]

        select_sql = f"""
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
            {where_sql}
            ORDER BY created_at DESC, row_index ASC
            LIMIT ?
            OFFSET ?
        """
        select_params = params + [limit, offset]
        rows = cur.execute(select_sql, select_params).fetchall()

        result = []
        for r in rows:
            result.append({
                "row_id": r["row_id"],
                "file_id": r["file_id"],
                "source_type": r["source_type"],
                "source_key": r["source_key"],
                "source_item_id": r["source_item_id"],
                "row_index": r["row_index"],
                "content": r["content"],
                "created_at": r["created_at"],
            })

        return {
            "rows": result,
            "count": len(result),
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "source_type": db_source_type,
        }

    finally:
        conn.close()
