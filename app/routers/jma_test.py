# app/routers/jma_test.py
from fastapi import APIRouter
import requests

router = APIRouter(prefix="/jma", tags=["jma"])

@router.get("/test")
def jma_test():
    # 例：東京の概況（JSON）
    url = "https://www.jma.go.jp/bosai/forecast/data/overview_forecast/130000.json"

    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()

    # 画面側は {count} を見ているので count を返す
    # 概況は1件扱い
    return {
        "count": 1,
        "title": data.get("headlineText") or "",
        "reportDatetime": data.get("reportDatetime") or ""
    }
