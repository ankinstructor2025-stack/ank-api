import os
from fastapi import APIRouter, Header, HTTPException
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth

router = APIRouter()

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")

TEMPLATE_DB_PATH = "template/template.sqlite"
TEMPLATE_JSON_PATH = "template/knowledge_generate.json"

# ユーザDBの保存先
def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


def user_json_path(uid: str) -> str:
    return f"users/{uid}/knowledge_generate.json"


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


def rollback_user_init(
    r_db: dict | None,
    r_json: dict | None,
) -> None:
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

    try:
        r_db = ensure_user_db_in_gcs(uid)
        r_json = ensure_user_json_in_gcs(uid)

        return {
            "ok": True,
            "user_id": uid,
            "db_created": r_db["created"],
            "json_created": r_json["created"],
            "db_gcs_path": r_db["db_gcs_path"],
            "json_gcs_path": r_json["json_gcs_path"],
        }

    except HTTPException:
        rollback_user_init(r_db, r_json)
        raise
    except Exception as e:
        rollback_user_init(r_db, r_json)
        raise HTTPException(status_code=500, detail=f"user init failed: {e}")
