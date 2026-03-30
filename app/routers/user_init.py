# routers/user_init.py
import os
from fastapi import APIRouter, Header, HTTPException
from google.cloud import storage, tasks_v2
from google.api_core.exceptions import NotFound

import firebase_admin
from firebase_admin import auth as fb_auth, credentials

router = APIRouter()

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")

TEMPLATE_DB_PATH = "template/template.sqlite"
TEMPLATE_JSON_PATH = "template/knowledge_generate.json"

# ユーザDBの保存先（デモはこれ固定が一番ラク）
def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


# Firebase Admin 初期化（1回だけ）
def ensure_firebase_initialized():
    if firebase_admin._apps:
        return
    # Cloud RunならADCでOK（環境変数不要）
    firebase_admin.initialize_app(
        options={"projectId": "ank-firebase"}
    )

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


def ensure_user_json_in_gcs(uid: str) -> dict:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    dest_path = f"users/{uid}/knowledge_generate.json"
    dest_blob = bucket.blob(dest_path)

    # 既にある → 何もしない
    if dest_blob.exists():
        return {"created": False, "json_gcs_path": dest_path}

    # テンプレ確認
    src_blob = bucket.blob(TEMPLATE_JSON_PATH)
    if not src_blob.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Template JSON not found: gs://{BUCKET_NAME}/{TEMPLATE_JSON_PATH}"
        )

    bucket.copy_blob(src_blob, bucket, new_name=dest_path)

    return {"created": True, "json_gcs_path": dest_path}


PROJECT_ID = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = "asia-northeast1"

def user_queue_name(uid: str) -> str:
    return f"queue-{uid}"


def ensure_user_queue(uid: str) -> dict:
    client = tasks_v2.CloudTasksClient()

    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID not found")

    queue_id = user_queue_name(uid)

    parent = client.common_location_path(PROJECT_ID, LOCATION)
    queue_path = client.queue_path(PROJECT_ID, LOCATION, queue_id)

    try:
        client.get_queue(name=queue_path)
        return {"created": False, "queue": queue_id}
    except NotFound:
        pass

    queue = {
        "name": queue_path,
        "rate_limits": {
            "max_dispatches_per_second": 1,
            "max_concurrent_dispatches": 1,
        },
        "retry_config": {
            "max_attempts": 3,
        },
    }

    client.create_queue(parent=parent, queue=queue)

    return {"created": True, "queue": queue_id}


@router.post("/user/init")
def user_init(authorization: str | None = Header(default=None)):
    uid = get_uid_from_auth_header(authorization)

    r_db = ensure_user_db_in_gcs(uid)
    r_json = ensure_user_json_in_gcs(uid)

    r_queue = ensure_user_queue(uid)

    return {
        "ok": True,
        "user_id": uid,
        "db_created": r_db["created"],
        "json_created": r_json["created"],
        "queue_created": r_queue["created"],
        "queue_name": r_queue["queue"],
        "db_gcs_path": r_db["db_gcs_path"],
        "json_gcs_path": r_json["json_gcs_path"],
    }
