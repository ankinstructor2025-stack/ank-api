# routers/admin_users.py
import json
import os

from fastapi import APIRouter, Header, HTTPException
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth

router = APIRouter()

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
ADMIN_EMAILS_BLOB_PATH = "template/admin_emails.json"


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

    gcs_result = delete_user_gcs_prefix(uid)

    return {
        "ok": True,
        "admin_email": admin["email"],
        "uid": uid,
        "gcs": gcs_result,
        "errors": [],
    }
