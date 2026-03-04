# app/routers/kokkai_router.py

import os
import json
from fastapi import APIRouter, HTTPException
import requests
from google.cloud import storage

router = APIRouter(
    prefix="/kokkai",
    tags=["kokkai"]
)

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
KOKKAI_TEMPLATE_PATH = "template/kokkai.json"


def load_kokkai_template() -> dict:
    """
    GCS: template/kokkai.json を読み込んで dict で返す
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(KOKKAI_TEMPLATE_PATH)

    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"{KOKKAI_TEMPLATE_PATH} not found")

    text = blob.download_as_text(encoding="utf-8")
    try:
        return json.loads(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"invalid template json: {e}")


# ----------------------------------------
# 取得テスト（まずはここだけ）
# ----------------------------------------
@router.post("/test")
def kokkai_test():
    """
    国会議事録API 接続テスト
    デモ用途：
      - 外部API取得確認
      - ingest導線確認
    template/kokkai.json を正として参照する
    """

    tpl = load_kokkai_template()

    url = tpl.get("endpoint") or "https://kokkai.ndl.go.jp/api/meeting"
    params = tpl.get("params") or {}

    # デモなので最低限の安全策（テンプレが欠けてても動く）
    params.setdefault("maximumRecords", 10)
    params.setdefault("recordPacking", "json")

    try:
        res = requests.get(url, params=params, timeout=20)
        res.raise_for_status()

        data = res.json()
        meetings = data.get("meetingRecord", [])

        if not meetings:
            return {
                "count": 0,
                "message": "meeting not found"
            }

        meeting = meetings[0]

        return {
            "count": len(meetings),
            "meeting_name": meeting.get("nameOfMeeting"),
            "date": meeting.get("date")
        }

    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"kokkai api error: {str(e)}"
        )
