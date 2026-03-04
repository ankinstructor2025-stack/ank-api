# app/routers/kokkai_router.py

import os
import json
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
import requests
from google.cloud import storage

router = APIRouter(prefix="/kokkai", tags=["kokkai"])

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


def _is_speech(url: str) -> bool:
    return "/api/speech" in (url or "").lower()


def _record_key(url: str) -> str:
    return "speechRecord" if _is_speech(url) else "meetingRecord"


def _clamp_maximum_records(url: str, value: Any) -> int:
    """
    speech: 1..100
    meeting: 条件次第で10制限になりやすいので 1..10
    """
    try:
        n = int(value)
    except Exception:
        n = 10

    if _is_speech(url):
        return max(1, min(100, n))
    return max(1, min(10, n))


def _fetch_json(url: str, params: dict) -> Tuple[dict, str]:
    """
    (json, requested_url) を返す
    """
    try:
        res = requests.get(url, params=params, timeout=20)
        res.raise_for_status()
        return res.json(), res.url
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"kokkai api error: {str(e)}")


def _extract_records(data: dict, key: str) -> List[Dict[str, Any]]:
    v = data.get(key)
    return v if isinstance(v, list) else []


def _next_record_position(data: dict) -> Optional[int]:
    v = data.get("nextRecordPosition")
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _summarize_first(url: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}

    r = records[0]
    if _is_speech(url):
        return {
            "speech_id": r.get("speechID") or r.get("speechId") or r.get("speech_id"),
            "speaker": r.get("speaker"),
            "date": r.get("date"),
            "nameOfMeeting": r.get("nameOfMeeting"),
        }

    return {
        "meeting_name": r.get("nameOfMeeting"),
        "date": r.get("date"),
    }


@router.post("/test")
def kokkai_test():
    """
    国会議事録API 接続テスト（テンプレ参照）
    - endpoint が meeting/speech どちらでも動く
    - fetch.all=true の場合は startRecord でページングして max_total まで取得
    """

    tpl = load_kokkai_template()

    url = tpl.get("endpoint") or "https://kokkai.ndl.go.jp/api/meeting"
    params = dict(tpl.get("params") or {})
    fetch_cfg = dict(tpl.get("fetch") or {})

    # 必須系（テンプレが欠けてても動く）
    params.setdefault("recordPacking", "json")
    params["maximumRecords"] = _clamp_maximum_records(url, params.get("maximumRecords", 10))

    key = _record_key(url)

    all_mode = bool(fetch_cfg.get("all", False))
    try:
        max_total = int(fetch_cfg.get("max_total", params["maximumRecords"]))
    except Exception:
        max_total = params["maximumRecords"]
    if max_total < 1:
        max_total = 1

    # ---- 1回だけ ----
    if not all_mode:
        data, requested_url = _fetch_json(url, params)
        records = _extract_records(data, key)

        return {
            "mode": "single",
            "record_type": key,
            "requested_url": requested_url,
            "count": len(records),
            "first": _summarize_first(url, records),
            # 0件切り分け用（デモで便利。不要なら後で消してOK）
            "top_keys": list(data.keys()),
            "numberOfRecords": data.get("numberOfRecords"),
            "nextRecordPosition": data.get("nextRecordPosition"),
        }

    # ---- all=true（ページング） ----
    collected: List[Dict[str, Any]] = []
    start = int(params.get("startRecord", 1) or 1)
    requested_url_last = ""

    while len(collected) < max_total:
        params["startRecord"] = start

        data, requested_url_last = _fetch_json(url, params)
        records = _extract_records(data, key)

        if not records:
            break

        remain = max_total - len(collected)
        collected.extend(records[:remain])

        nxt = _next_record_position(data)
        if not nxt:
            break
        start = nxt

    return {
        "mode": "all",
        "record_type": key,
        "requested_url": requested_url_last,
        "count": len(collected),
        "max_total": max_total,
        "first": _summarize_first(url, collected),
    }
