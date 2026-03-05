# app/routers/opendata_test.py

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

# template/opendata.json を参照
BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
OPENDATA_TEMPLATE_PATH = "template/opendata.json"


# ---------------------------
# Firebase 認証（kokkai と同じ）
# ---------------------------
def ensure_firebase_initialized():
    if firebase_admin._apps:
        return
    firebase_admin.initialize_app()


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
        return decoded["uid"]

    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

# ---------------------------
# Template 読み込み
# ---------------------------
def load_opendata_template() -> dict:
    """
    GCS: template/opendata.json を読み込んで dict で返す
    """
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


# ---------------------------
# HTTP / CKAN helper
# ---------------------------
def _fetch_any_json(url: str, params: dict | None = None) -> tuple[dict, str]:
    """
    (json, requested_url) を返す
    """
    try:
        res = requests.get(url, params=params or {}, timeout=20)
        res.raise_for_status()
        return res.json(), res.url
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"opendata api error: {str(e)}")


def _ckan_get(action: str, params: dict | None = None) -> dict:
    """
    CKAN action API を GET で呼び出して result を返す（既存維持）
    """
    url = f"{CKAN_BASE}/{action}"
    payload, _requested_url = _fetch_any_json(url, params=params)
    if not payload.get("success", False):
        raise HTTPException(status_code=502, detail=f"ckan api error: success=false ({action})")
    return payload.get("result")


def _resolve_template_endpoint_and_params(tpl: dict) -> tuple[str, dict]:
    """
    template/opendata.json から endpoint/params を取り出す

    1) シンプル形式
       { "endpoint": "...", "params": {...} }

    2) 拡張形式
       { "dataset_search": { "endpoint": "...", "params": {...} }, ... }

    /test はどちらでも動くようにする
    """
    if isinstance(tpl.get("endpoint"), str):
        return tpl["endpoint"], (tpl.get("params") or {})

    ds = tpl.get("dataset_search") or {}
    if isinstance(ds.get("endpoint"), str):
        return ds["endpoint"], (ds.get("params") or {})

    raise HTTPException(status_code=500, detail="template missing: endpoint")


def _normalize_ckan_result(payload: dict) -> tuple[object, bool]:
    """
    CKAN形式 {"success": true, "result": ...} なら result を返す。
    そうでなければ payload をそのまま返す。
    """
    if isinstance(payload, dict) and "success" in payload and "result" in payload:
        if not payload.get("success", False):
            return payload, False
        return payload.get("result"), True
    return payload, True


def _calc_count(result: object) -> tuple[int, int | None]:
    """
    UI向け count を安定して返す
    - list => len(list)
    - dict with results(list) => len(results), total=result.get("count")
    - other => 0
    """
    if isinstance(result, list):
        return len(result), None

    if isinstance(result, dict):
        results = result.get("results")
        if isinstance(results, list):
            total = result.get("count")
            try:
                total = int(total) if total is not None else None
            except Exception:
                total = None
            return len(results), total

    return 0, None


# ----------------------------------------
# 取得テスト（テンプレ参照 + 認証必須）
# 呼び出し側（kokkai方式）に合わせる
# ----------------------------------------
@router.get("/test")
def opendata_test(
    authorization: str | None = Header(default=None),
):
    """
    e-Govデータポータル（CKAN）接続テスト（template/opendata.json を参照）
    - Authorization: Bearer <idToken> を必須（kokkai と同じ）
    - endpoint / params はテンプレに寄せる
    - count は必ず返す
    """
    _uid = get_uid_from_auth_header(authorization)  # いまは認証確認のみ（将来の行登録で利用）

    tpl = load_opendata_template()
    endpoint, params = _resolve_template_endpoint_and_params(tpl)

    payload, requested_url = _fetch_any_json(endpoint, params=params)
    result, ok = _normalize_ckan_result(payload)
    if not ok:
        raise HTTPException(status_code=502, detail="ckan api error: success=false")

    count, total = _calc_count(result)

    resp = {
        "mode": "test",
        "requested_url": requested_url,
        "count": count,
    }
    if total is not None:
        resp["total"] = total  # package_searchの総件数など

    # 返しすぎない（デモなので先頭だけ）
    if isinstance(result, list):
        resp["items"] = result[:20]
    elif isinstance(result, dict) and isinstance(result.get("results"), list):
        resp["items"] = result["results"][:20]
    else:
        resp["items"] = []

    return resp


# ---- 以下は既存のまま（直書き） ----

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

    return {
        "count": result.get("count", 0),
        "rows": rows,
        "start": start,
        "items": items
    }


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

    return {
        "count": len(items),
        "limit": limit,
        "offset": offset,
        "items": items
    }


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
