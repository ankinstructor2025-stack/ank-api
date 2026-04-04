from fastapi import APIRouter, HTTPException, Request
from google.cloud import storage
import os
import json

router = APIRouter(prefix="/v1/job-status", tags=["job-status"])

BUCKET_NAME = os.getenv("BUCKET_NAME")


def get_gcs_client():
    return storage.Client()


@router.get("")
async def list_job_status(request: Request):
    try:
        uid = request.state.uid  # 既存の認証使う前提

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
