from fastapi import APIRouter
import requests

router = APIRouter(prefix="/caa", tags=["caa"])

@router.get("/test")
def caa_test():
    url = "https://www.caa.go.jp/policies/policy/consumer_policy/"

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ank-bot/1.0)"
    }

    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    html = r.text or ""

    return {
        "count": 1,
        "bytes": len(html.encode("utf-8")),
        "status": r.status_code
    }
