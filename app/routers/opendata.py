# app/routers/opendata.py

from fastapi import APIRouter, HTTPException, Query, Header
import os
import json
import requests
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth

router = APIRouter(prefix="/opendata", tags=["opendata"])

# 既存の直書きは残す（/datasets/* が依存しているため）
CKAN_BASE = "https://data.e-gov.go.jp/data/api/action"  # e-Govデータポータル CKAN API base

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
OPENDATA_TEMPLATE_PATH = "template/opendata.json"

def ensure_firebase_initialized():
    if firebase_admin._apps:
        return
    firebase_admin.initialize_app(options={"projectId": "ank-firebase"})

def get_uid_from_auth_header(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    token = authorization.replace("Bearer ", "", 1).strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")

    ensure_firebase_initialized()
    try:
        decoded = fb_auth.verify_id_token(token)
        uid = decoded.get("uid")
        if not uid:
            raise HTTPException(status_code=401, detail="uid not found in token")
        return uid
    except Exception as e:
        # まず原因を見える化（401の理由を返す）
        raise HTTPException(status_code=401, detail=str(e))

# ---------------------------
# Template 読み込み
# ---------------------------
def load_opendata_template() -> dict:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(OPENDATA_TEMPLATE_PATH)

    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"{OPENDATA_TEMPLATE_PATH} not found")

    text = blob.download_as_text(encoding="utf-8")
    try:
        return json.loads(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"invalid template json: {e}")

def _resolve_template_endpoint_and_params(tpl: dict) -> tuple[str, dict]:
    # 形式1: { "endpoint": "...", "params": {...} }
    if isinstance(tpl.get("endpoint"), str):
        return tpl["endpoint"], (tpl.get("params") or {})

    # 形式2（拡張）: { "dataset_search": { "endpoint": "...", "params": {...} }, ... }
    ds = tpl.get("dataset_search") or {}
    if isinstance(ds.get("endpoint"), str):
        return ds["endpoint"], (ds.get("params") or {})

    raise HTTPException(status_code=500, detail="template missing: endpoint")

# ---------------------------
# CKAN helper（既存）
# ---------------------------
def _fetch_any_json(url: str, params: dict | None = None) -> tuple[dict, str]:
    try:
        res = requests.get(url, params=params or {}, timeout=20)
        res.raise_for_status()
        return res.json(), res.url
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"ckan api error: {str(e)}")

def _ckan_get(action: str, params: dict | None = None) -> dict:
    url = f"{CKAN_BASE}/{action}"
    payload, _requested_url = _fetch_any_json(url, params=params)
    if not payload.get("success", False):
        raise HTTPException(status_code=502, detail=f"ckan api error: success=false ({action})")
    return payload.get("result")

def _normalize_ckan_result(payload: dict) -> object:
    # CKAN形式なら result を返す
    if isinstance(payload, dict) and "success" in payload and "result" in payload:
        if not payload.get("success", False):
            raise HTTPException(status_code=502, detail="ckan api error: success=false")
        return payload.get("result")
    return payload

def _calc_count_and_items(result: object) -> tuple[int, list]:
    # package_list: result は list
    if isinstance(result, list):
        return len(result), result[:20]

    # package_search: result は dict（count/results）
    if isinstance(result, dict) and isinstance(result.get("results"), list):
        items = result["results"][:20]
        return len(items), items

    return 0, []

# ----------------------------------------
# 取得テスト（呼び出し側を kokkai に合わせる）
#  - Authorization 必須
#  - template/opendata.json 参照
#  - count を返す
# ----------------------------------------
@router.get("/fetch_and_register")
def opendata_fetch_and_register(authorization: str | None = Header(default=None)):
    _uid = get_uid_from_auth_header(authorization)  # 認証だけ合わせる（今はuid未使用）

    tpl = load_opendata_template()
    endpoint, params = _resolve_template_endpoint_and_params(tpl)

    payload, requested_url = _fetch_any_json(endpoint, params=params)
    result = _normalize_ckan_result(payload)

    count, items = _calc_count_and_items(result)

    return {
        "mode": "fetch_and_register",
        "requested_url": requested_url,
        "count": count,
        "items": items
    }


# ---- 以下は既存のまま（必要なら後で認証を揃える） ----

@router.get("/datasets/search")
def search_datasets(
    q: str = Query(..., description="検索キーワード（CKAN package_search の q）"),
    rows: int = Query(10, ge=1, le=100, description="取得件数"),
    start: int = Query(0, ge=0, description="開始オフセット"),
):
    result = _ckan_get("package_search", params={"q": q, "rows": rows, "start": start})

    items = []
    for ds in result.get("results", []):
        items.append({
            "id": ds.get("id") or ds.get("name"),
            "name": ds.get("name"),
            "title": ds.get("title"),
            "notes": ds.get("notes"),
            "organization": (ds.get("organization") or {}).get("title"),
            "num_resources": len(ds.get("resources", []) or []),
        })

    return {"count": result.get("count", 0), "rows": rows, "start": start, "items": items}


@router.get("/datasets/recent")
def recent_datasets(
    limit: int = Query(10, ge=1, le=100, description="取得件数"),
    offset: int = Query(0, ge=0, description="開始オフセット"),
):
    result = _ckan_get("current_package_list_with_resources", params={"limit": limit, "offset": offset})

    items = []
    for ds in result or []:
        items.append({
            "id": ds.get("id") or ds.get("name"),
            "name": ds.get("name"),
            "title": ds.get("title"),
            "organization": (ds.get("organization") or {}).get("title"),
            "num_resources": len(ds.get("resources", []) or []),
        })

    return {"count": len(items), "limit": limit, "offset": offset, "items": items}


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    ds = _ckan_get("package_show", params={"id": dataset_id})

    resources = []
    for r in ds.get("resources", []) or []:
        resources.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "format": r.get("format"),
            "mimetype": r.get("mimetype"),
            "url": r.get("url"),
            "last_modified": r.get("last_modified"),
        })

    return {
        "id": ds.get("id") or ds.get("name"),
        "name": ds.get("name"),
        "title": ds.get("title"),
        "notes": ds.get("notes"),
        "organization": (ds.get("organization") or {}).get("title"),
        "license_id": ds.get("license_id"),
        "metadata_created": ds.get("metadata_created"),
        "metadata_modified": ds.get("metadata_modified"),
        "resources": resources,
    }


@router.get("/resources/{resource_id}")
def get_resource(resource_id: str):
    r = _ckan_get("resource_show", params={"id": resource_id})
    return {
        "id": r.get("id"),
        "name": r.get("name"),
        "format": r.get("format"),
        "mimetype": r.get("mimetype"),
        "url": r.get("url"),
        "last_modified": r.get("last_modified"),
        "created": r.get("created"),
    }
