# routers/admin_users.py
import os
from fastapi import APIRouter, Header, HTTPException
from google.cloud import storage, tasks_v2
from google.api_core.exceptions import NotFound

import firebase_admin
from firebase_admin import auth as fb_auth

from app.core.common import user_queue_name

router = APIRouter()

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = "asia-northeast1"

# あなたのメールアドレスに置き換える
ADMIN_EMAILS = {
    "your-email@example.com",
}


def ensure_firebase_initialized():
    if firebase_admin._apps:
        return
    firebase_admin.initialize_app(
        options={"projectId": "ank-firebase"}
    )


def get_admin_user(authorization: str | None) -> dict:
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
        email = decoded.get("email", "")
        uid = decoded.get("uid", "")

        if not email or email not in ADMIN_EMAILS:
            raise HTTPException(status_code=403, detail="Admin only")

        return {
            "uid": uid,
            "email": email,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid ID token: {e}")


def list_user_prefixes() -> list[dict]:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # users/{uid}/ の uid 部分を拾う
    blobs = client.list_blobs(BUCKET_NAME, prefix="users/")
    uid_map = {}

    for blob in blobs:
        name = blob.name
        # users/<uid>/...
        parts = name.split("/")
        if len(parts) >= 2 and parts[0] == "users" and parts[1]:
            uid = parts[1]
            if uid not in uid_map:
                uid_map[uid] = {
                    "uid": uid,
                    "prefix": f"users/{uid}/",
                    "file_count": 0,
                }
            uid_map[uid]["file_count"] += 1

    return sorted(uid_map.values(), key=lambda x: x["uid"])


def delete_user_gcs_prefix(uid: str) -> dict:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    prefix = f"users/{uid}/"

    blobs = list(client.list_blobs(BUCKET_NAME, prefix=prefix))
    deleted_count = 0

    for blob in blobs:
        blob.delete()
        deleted_count += 1

    return {
        "uid": uid,
        "prefix": prefix,
        "deleted_blob_count": deleted_count,
    }


def delete_user_queue(uid: str) -> dict:
    client = tasks_v2.CloudTasksClient()

    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID not found")

    queue_id = user_queue_name(uid)
    queue_path = client.queue_path(PROJECT_ID, LOCATION, queue_id)

    try:
        client.get_queue(name=queue_path)
    except NotFound:
        return {"queue": queue_id, "deleted": False}

    client.delete_queue(name=queue_path)
    return {"queue": queue_id, "deleted": True}


@router.get("/admin/users")
def admin_list_users(authorization: str | None = Header(default=None)):
    admin = get_admin_user(authorization)
    users = list_user_prefixes()

    return {
        "ok": True,
        "admin_email": admin["email"],
        "users": users,
    }


@router.delete("/admin/users/{uid}")
def admin_delete_user(uid: str, authorization: str | None = Header(default=None)):
    admin = get_admin_user(authorization)

    gcs_result = delete_user_gcs_prefix(uid)

    queue_errors = []
    try:
        queue_result = delete_user_queue(uid)
    except Exception as e:
        queue_result = {"queue": user_queue_name(uid), "deleted": False}
        queue_errors.append(str(e))

    return {
        "ok": len(queue_errors) == 0,
        "admin_email": admin["email"],
        "uid": uid,
        "gcs": gcs_result,
        "queue": queue_result,
        "errors": queue_errors,
    }
