import json
import os
from typing import Any

from google.cloud import storage

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")

CLOUD_RUN_BASE_URL = os.getenv("CLOUD_RUN_BASE_URL")

if not CLOUD_RUN_BASE_URL:
    raise RuntimeError("CLOUD_RUN_BASE_URL is not set")

CLOUD_RUN_BASE_URL = CLOUD_RUN_BASE_URL.rstrip("/")

def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)

    # 改行統一
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # タブ→スペース（地味に効く）
    text = text.replace("\t", " ")

    # 行単位でtrim
    lines = [line.strip() for line in text.split("\n")]

    # 空行削除して再結合
    return "\n".join(line for line in lines if line)

def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"

def local_user_db_path(uid: str) -> str:
    return f"/tmp/ank_{uid}.db"

def user_task_db_path(uid: str, job_id: str) -> str:
    return f"users/{uid}/{job_id}.db"

def local_task_db_path(uid: str, job_id: str) -> str:
    return f"/tmp/{uid}_{job_id}.db"

def user_queue_json_path(uid: str) -> str:
    return f"users/{uid}/queue.json"

def load_user_queue_config(uid: str) -> dict:
    if not uid:
        raise RuntimeError("uid is empty")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    gcs_path = user_queue_json_path(uid)
    blob = bucket.blob(gcs_path)

    if not blob.exists():
        raise RuntimeError(f"queue.json not found: gs://{BUCKET_NAME}/{gcs_path}")

    text = blob.download_as_text(encoding="utf-8")
    data = json.loads(text)

    if not isinstance(data, dict):
        raise RuntimeError("queue.json is invalid")

    queue_name = (data.get("queue_name") or "").strip()
    queue_full_name = (data.get("queue_full_name") or "").strip()

    if not queue_name:
        raise RuntimeError("queue_name is missing in queue.json")

    if not queue_full_name:
        raise RuntimeError("queue_full_name is missing in queue.json")

    return data
