import os
import json
import hashlib
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException
from google.cloud import storage, tasks_v2
from google.api_core.exceptions import AlreadyExists, NotFound

import firebase_admin
from firebase_admin import auth as fb_auth

router = APIRouter()

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
PROJECT_ID = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
TASKS_LOCATION = os.getenv("TASKS_LOCATION", "asia-northeast1")

TEMPLATE_DB_PATH = "template/template.sqlite"
TEMPLATE_JSON_PATH = "template/knowledge_generate.json"


# ユーザDBの保存先
def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


def user_json_path(uid: str) -> str:
    return f"users/{uid}/knowledge_generate.json"


def user_queue_json_path(uid: str) -> str:
    return f"users/{uid}/queue.json"


# Firebase Admin 初期化（1回だけ）
def ensure_firebase_initialized():
    if firebase_admin._apps:
        return
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

    if dest_blob.exists():
        return {"created": False, "db_gcs_path": dest_path}

    src_blob = bucket.blob(TEMPLATE_DB_PATH)
    if not src_blob.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Template DB not found: gs://{BUCKET_NAME}/{TEMPLATE_DB_PATH}"
        )

    bucket.copy_blob(src_blob, bucket, new_name=dest_path)
    return {"created": True, "db_gcs_path": dest_path}


def ensure_user_json_in_gcs(uid: str) -> dict:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    dest_path = user_json_path(uid)
    dest_blob = bucket.blob(dest_path)

    if dest_blob.exists():
        return {"created": False, "json_gcs_path": dest_path}

    src_blob = bucket.blob(TEMPLATE_JSON_PATH)
    if not src_blob.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Template JSON not found: gs://{BUCKET_NAME}/{TEMPLATE_JSON_PATH}"
        )

    bucket.copy_blob(src_blob, bucket, new_name=dest_path)
    return {"created": True, "json_gcs_path": dest_path}


def delete_gcs_blob_if_exists(path: str) -> None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)

    if blob.exists():
        blob.delete()


def load_gcs_json_if_exists(path: str) -> dict | None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)

    if not blob.exists():
        return None

    text = blob.download_as_text(encoding="utf-8")
    if not text.strip():
        return None

    return json.loads(text)


def upload_json_to_gcs(path: str, data: dict) -> None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)
    blob.upload_from_string(
        json.dumps(data, ensure_ascii=False, indent=2),
        content_type="application/json; charset=utf-8",
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_queue_name(uid: str) -> tuple[str, str]:
    uid_hash = hashlib.sha1(uid.encode("utf-8")).hexdigest()[:10]
    ts_for_name = utc_now().strftime("%Y%m%d%H%M")
    suffix = secrets.token_hex(2)
    queue_name = f"q-{uid_hash}-{ts_for_name}-{suffix}"
    return queue_name, ts_for_name


def get_queue_parent() -> str:
    if not PROJECT_ID:
        raise HTTPException(status_code=500, detail="PROJECT_ID or GOOGLE_CLOUD_PROJECT is not set")
    return f"projects/{PROJECT_ID}/locations/{TASKS_LOCATION}"


def get_queue_full_name(queue_name: str) -> str:
    return f"{get_queue_parent()}/queues/{queue_name}"


def create_cloud_tasks_queue(queue_name: str) -> str:
    client = tasks_v2.CloudTasksClient()
    parent = get_queue_parent()

    queue = {
        "name": f"{parent}/queues/{queue_name}",
        "rate_limits": {
            "max_dispatches_per_second": 1.0,
            "max_concurrent_dispatches": 1,
        },
    }

    try:
        created = client.create_queue(parent=parent, queue=queue)
        return created.name
    except AlreadyExists:
        raise HTTPException(status_code=409, detail=f"Queue already exists: {queue_name}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"create queue failed: {e}")


def delete_cloud_tasks_queue_if_exists(queue_name: str) -> None:
    client = tasks_v2.CloudTasksClient()
    queue_full_name = get_queue_full_name(queue_name)

    try:
        client.delete_queue(name=queue_full_name)
    except NotFound:
        pass


def ensure_user_queue(uid: str) -> dict:
    queue_json_gcs_path = user_queue_json_path(uid)
    saved = load_gcs_json_if_exists(queue_json_gcs_path)

    if saved:
        queue_name = saved.get("queue_name")
        queue_full_name = saved.get("queue_full_name")
        created_at = saved.get("created_at")
        created_at_name = saved.get("created_at_name")

        if not queue_name:
            raise HTTPException(
                status_code=500,
                detail=f"queue.json is invalid: gs://{BUCKET_NAME}/{queue_json_gcs_path}"
            )

        if not queue_full_name:
            queue_full_name = get_queue_full_name(queue_name)

        return {
            "created": False,
            "queue_name": queue_name,
            "queue_full_name": queue_full_name,
            "queue_json_gcs_path": queue_json_gcs_path,
            "created_at": created_at,
            "created_at_name": created_at_name,
        }

    queue_name, created_at_name = build_queue_name(uid)
    queue_full_name = create_cloud_tasks_queue(queue_name)

    queue_info = {
        "queue_name": queue_name,
        "queue_full_name": queue_full_name,
        "project_id": PROJECT_ID,
        "location": TASKS_LOCATION,
        "created_at": utc_now().strftime("%Y-%m-%d %H:%M"),
        "created_at_name": created_at_name,
        "status": "active",
    }

    upload_json_to_gcs(queue_json_gcs_path, queue_info)

    return {
        "created": True,
        "queue_name": queue_name,
        "queue_full_name": queue_full_name,
        "queue_json_gcs_path": queue_json_gcs_path,
        "created_at": queue_info["created_at"],
        "created_at_name": queue_info["created_at_name"],
    }


def rollback_user_init(
    r_db: dict | None,
    r_json: dict | None,
    r_queue: dict | None,
) -> None:
    if r_queue and r_queue.get("created"):
        try:
            delete_gcs_blob_if_exists(r_queue["queue_json_gcs_path"])
        except Exception:
            pass

        try:
            delete_cloud_tasks_queue_if_exists(r_queue["queue_name"])
        except Exception:
            pass

    if r_json and r_json.get("created"):
        try:
            delete_gcs_blob_if_exists(r_json["json_gcs_path"])
        except Exception:
            pass

    if r_db and r_db.get("created"):
        try:
            delete_gcs_blob_if_exists(r_db["db_gcs_path"])
        except Exception:
            pass


@router.post("/user/init")
def user_init(authorization: str | None = Header(default=None)):
    uid = get_uid_from_auth_header(authorization)

    r_db = None
    r_json = None
    r_queue = None

    try:
        r_db = ensure_user_db_in_gcs(uid)
        r_json = ensure_user_json_in_gcs(uid)
        r_queue = ensure_user_queue(uid)

        return {
            "ok": True,
            "user_id": uid,
            "db_created": r_db["created"],
            "json_created": r_json["created"],
            "queue_created": r_queue["created"],
            "db_gcs_path": r_db["db_gcs_path"],
            "json_gcs_path": r_json["json_gcs_path"],
            "queue_json_gcs_path": r_queue["queue_json_gcs_path"],
            "queue_name": r_queue["queue_name"],
            "queue_full_name": r_queue["queue_full_name"],
            "queue_created_at": r_queue["created_at"],
            "queue_created_at_name": r_queue["created_at_name"],
        }

    except HTTPException:
        rollback_user_init(r_db, r_json, r_queue)
        raise
    except Exception as e:
        rollback_user_init(r_db, r_json, r_queue)
        raise HTTPException(status_code=500, detail=f"user init failed: {e}")
