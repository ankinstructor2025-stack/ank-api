from fastapi import APIRouter, HTTPException, Request
from google.cloud import storage
import os
import json

from app.routers.user_init import get_uid_from_auth_header

router = APIRouter(prefix="/job-status", tags=["job-status"])

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")


def get_gcs_client():
    return storage.Client()


@router.get("")
async def list_job_status(request: Request):
    try:
        uid = get_uid_from_auth_header(request.headers.get("Authorization"))

        client = get_gcs_client()
        bucket = client.bucket(BUCKET_NAME)

        prefix = f"users/{uid}/job_status/"

        blobs = client.list_blobs(bucket, prefix=prefix)

        jobs = []

        for blob in blobs:
            if not blob.name.endswith(".json"):
                continue

            content = blob.download_as_text()
            data = json.loads(content)

            jobs.append(data)

        # 更新日時でソート（新しい順）
        jobs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        return {"jobs": jobs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
