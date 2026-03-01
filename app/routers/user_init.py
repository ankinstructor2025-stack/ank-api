# routers/user_init.py
import os
from fastapi import APIRouter, Header, HTTPException
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth, credentials

router = APIRouter()

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")

# GCS上のテンプレDB（あなたの現状に合わせて）
TEMPLATE_DB_PATH = "template/template.sqlite"

# ユーザDBの保存先（デモはこれ固定が一番ラク）
def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


# Firebase Admin 初期化（1回だけ）
def ensure_firebase_initialized():
    if firebase_admin._apps:
        return
    # Cloud RunならADCでOK（環境変数不要）
    firebase_admin.initialize_app()


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


def ensure_user_db_in_gcs(uid: str) -> dict:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    dest_path = user_db_path(uid)
    dest_blob = bucket.blob(dest_path)

    # 既にある → 何もしない（冪等）
    if dest_blob.exists():
        return {"created": False, "db_gcs_path": dest_path}

    # テンプレの存在確認
    src_blob = bucket.blob(TEMPLATE_DB_PATH)
    if not src_blob.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Template DB not found: gs://{BUCKET_NAME}/{TEMPLATE_DB_PATH}"
        )

    # GCS内コピー（高速）
    bucket.copy_blob(src_blob, bucket, new_name=dest_path)

    return {"created": True, "db_gcs_path": dest_path}


@router.post("/v1/user/init")
def user_init(Authorization: str | None = Header(default=None)):
    uid = get_uid_from_auth_header(Authorization)
    r = ensure_user_db_in_gcs(uid)

    return {
        "ok": True,
        "user_id": uid,
        "created": r["created"],
        "db_gcs_path": r["db_gcs_path"],
    }
