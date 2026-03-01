# app/routers/ingest/kokkai_test.py

from fastapi import APIRouter
import requests

router = APIRouter(prefix="/kokkai", tags=["kokkai"])


@router.post("/test")
def kokkai_test():

    url = "https://kokkai.ndl.go.jp/api/meeting"

    params = {
        "keyword": "AI",
        "maximumRecords": 1
    }

    r = requests.get(url, params=params)
    r.raise_for_status()

    data = r.json()

    meetings = data.get("meetingRecord", [])

    return {
        "count": len(meetings)
    }
