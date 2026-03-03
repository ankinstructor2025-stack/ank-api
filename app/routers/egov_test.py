from fastapi import APIRouter
import requests

router = APIRouter(prefix="/egov", tags=["egov"])

@router.get("/test")
def egov_test():
    # e-Gov（法令・制度ページ）
    url = "https://elaws.e-gov.go.jp/"

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ank-bot/1.0; +https://example.invalid)"
    }

    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    html = r.text or ""
    # 画面側が {count} を期待しているので count を返す
    return {
        "count": 1,
        "bytes": len(html.encode("utf-8")),
        "status": r.status_code
    }
