# app/routers/kokkai_router.py

from fastapi import APIRouter, HTTPException
import requests

router = APIRouter(
    prefix="/kokkai",
    tags=["kokkai"]
)


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
    """

    url = "https://kokkai.ndl.go.jp/api/meeting"

    params = {
        "nameOfHouse": "衆議院",
        "nameOfMeeting": "本会議",
        "maximumRecords": 1,
        "recordPacking": "json",
    }

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
