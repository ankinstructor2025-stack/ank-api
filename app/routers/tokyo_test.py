from fastapi import APIRouter
import requests

router = APIRouter(prefix="/tokyo", tags=["tokyo"])

@router.get("/test")
def tokyo_test():
    # 東京都公式サイト（トップ）
    url = "https://www.metro.tokyo.lg.jp/"

    headers = {"User-Agent": "Mozilla/5.0 (compatible; ank-bot/1.0)"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    html = r.text or ""
    return {
        "count": 1,
        "bytes": len(html.encode("utf-8")),
        "status": r.status_code
    }
