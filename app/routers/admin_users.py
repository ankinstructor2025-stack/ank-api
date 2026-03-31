# routers/admin_users.py
import json
import os

from fastapi import APIRouter, Header, HTTPException
from google.cloud import storage, tasks_v2
from google.api_core.exceptions import NotFound

import firebase_admin
from firebase_admin import auth as fb_auth

router = APIRouter()

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
ADMIN_EMAILS_BLOB_PATH = "template/admin_emails.json"
PROJECT_ID = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
TASKS_LOCATION = os.getenv("TASKS_LOCATION", "asia-northeast1")


def ensure_firebase_initialized():
    if firebase_admin._apps:
        return
    firebase_admin.initialize_app(
        options={"projectId": "ank-firebase"}
    )


def load_admin_emails_from_gcs() -> set[str]:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(ADMIN_EMAILS_BLOB_PATH)

    if not blob.exists():
        raise RuntimeError(f"{ADMIN_EMAILS_BLOB_PATH} not found in GCS")

    try:
        text = blob.download_as_text(encoding="utf-8")
        data = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"failed to load {ADMIN_EMAILS_BLOB_PATH}: {e}")

    emails = data.get("emails")
    if not isinstance(emails, list):
        raise RuntimeError(f"invalid format: {ADMIN_EMAILS_BLOB_PATH} must contain 'emails' array")

    normalized = {
        str(email).strip().lower()
        for email in emails
        if str(email).strip()
    }
    return normalized


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
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid ID token: {e}")

    email = str(decoded.get("email", "")).strip().lower()
    uid = str(decoded.get("uid", "")).strip()

    try:
        admin_emails = load_admin_emails_from_gcs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not email or email not in admin_emails:
        raise HTTPException(status_code=403, detail="Admin only")

    return {
        "uid": uid,
        "email": email,
    }


def list_user_prefixes() -> list[dict]:
    client = storage.Client()
    blobs = client.list_blobs(BUCKET_NAME, prefix="users/")
    uid_map = {}

    for blob in blobs:
        name = blob.name
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


def get_queue_json_path(uid: str) -> str:
    return f"users/{uid}/queue.json"


def load_user_queue_info(uid: str) -> dict | None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(get_queue_json_path(uid))

    if not blob.exists():
        return None

    try:
        text = blob.download_as_text(encoding="utf-8")
        if not text.strip():
            return None
        data = json.loads(text)
        if not isinstance(data, dict):
            raise RuntimeError("queue.json is not a JSON object")
        return data
    except Exception as e:
        raise RuntimeError(f"failed to load queue.json for uid={uid}: {e}")


def build_queue_full_name(queue_name: str) -> str:
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID or GOOGLE_CLOUD_PROJECT is not set")
    return f"projects/{PROJECT_ID}/locations/{TASKS_LOCATION}/queues/{queue_name}"


def delete_user_queue_from_json(uid: str) -> dict:
    queue_info = load_user_queue_info(uid)

    if not queue_info:
        return {
            "found": False,
            "deleted": False,
            "queue_name": None,
            "queue_full_name": None,
        }

    queue_name = str(queue_info.get("queue_name", "")).strip()
    queue_full_name = str(queue_info.get("queue_full_name", "")).strip()

    if not queue_full_name and queue_name:
        queue_full_name = build_queue_full_name(queue_name)

    if not queue_full_name:
        raise RuntimeError(f"queue.json invalid: queue_name/queue_full_name not found for uid={uid}")

    client = tasks_v2.CloudTasksClient()

    try:
        client.delete_queue(name=queue_full_name)
        return {
            "found": True,
            "deleted": True,
            "queue_name": queue_name or None,
            "queue_full_name": queue_full_name,
        }
    except NotFound:
        return {
            "found": True,
            "deleted": False,
            "queue_name": queue_name or None,
            "queue_full_name": queue_full_name,
            "reason": "queue not found",
        }


def delete_user_gcs_prefix(uid: str) -> dict:
    client = storage.Client()
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

    errors = []
    queue_result = None

    try:
        queue_result = delete_user_queue_from_json(uid)
    except Exception as e:
        errors.append(f"delete queue failed: {e}")

    try:
        gcs_result = delete_user_gcs_prefix(uid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"delete user gcs failed: {e}")

    return {
        "ok": len(errors) == 0,
        "admin_email": admin["email"],
        "uid": uid,
        "queue": queue_result,
        "gcs": gcs_result,
        "errors": errors,
    }
