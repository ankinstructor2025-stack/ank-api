# app/routers/opendata.py

from fastapi import APIRouter, HTTPException, Query, Header
import os
import json
import requests
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth

import ulid

router = APIRouter(prefix="/opendata", tags=["opendata"])

JST = ZoneInfo("Asia/Tokyo")

CKAN_BASE = "https://data.e-gov.go.jp/data/api/action"

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
OPENDATA_TEMPLATE_PATH = "template/opendata.json"


def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


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
        raise HTTPException(status_code=401, detail=str(e))


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
    if isinstance(tpl.get("endpoint"), str):
        return tpl["endpoint"], (tpl.get("params") or {})

    ds = tpl.get("dataset_search") or {}
    if isinstance(ds.get("endpoint"), str):
        return ds["endpoint"], (ds.get("params") or {})

    raise HTTPException(status_code=500, detail="template missing: endpoint")


def _get_dataset_limit_from_template(tpl: dict) -> Optional[int]:
    df = tpl.get("data_fetch") or {}
    try:
        n = int(df.get("max_rows"))
    except Exception:
        return None
    return n if n > 0 else None


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
    if isinstance(payload, dict) and "success" in payload and "result" in payload:
        if not payload.get("success", False):
            raise HTTPException(status_code=502, detail="ckan api error: success=false")
        return payload.get("result")
    return payload


def _calc_count_and_items(result: object) -> tuple[int, list]:
    if isinstance(result, list):
        return len(result), result[:20]

    if isinstance(result, dict) and isinstance(result.get("results"), list):
        items = result["results"][:20]
        return len(items), items

    return 0, []


def _fetch_ckan_package_search_all(
    endpoint: str,
    params: dict,
    max_items: Optional[int],
) -> tuple[int, list[dict], str]:
    try:
        rows = int(params.get("rows", 100))
    except Exception:
        rows = 100
    if rows < 1:
        rows = 100

    try:
        start = int(params.get("start", 0))
    except Exception:
        start = 0
    if start < 0:
        start = 0

    all_items: list[dict] = []
    total_count: Optional[int] = None
    requested_url_last = endpoint

    while True:
        p = dict(params)
        p["rows"] = rows
        p["start"] = start

        payload, requested_url = _fetch_any_json(endpoint, params=p)
        requested_url_last = requested_url

        result = _normalize_ckan_result(payload)
        if not isinstance(result, dict) or "results" not in result:
            raise HTTPException(status_code=502, detail="unexpected ckan result shape (package_search)")

        if total_count is None:
            try:
                total_count = int(result.get("count", 0))
            except Exception:
                total_count = 0

        batch = result.get("results") or []
        if not isinstance(batch, list):
            batch = []

        if len(batch) == 0:
            break

        for x in batch:
            if isinstance(x, dict):
                all_items.append(x)
            else:
                all_items.append({"value": x})

            if max_items is not None and len(all_items) >= max_items:
                all_items = all_items[:max_items]
                break

        if max_items is not None and len(all_items) >= max_items:
            break

        if total_count is not None and len(all_items) >= total_count:
            break

        start += rows

    return (total_count or 0), all_items, requested_url_last


def _dataset_id(item: Any) -> str:
    if isinstance(item, dict):
        v = item.get("id") or item.get("name") or item.get("title")
        return str(v or "")
    return str(item or "")


def _dataset_title(item: dict) -> str:
    return str(item.get("title") or item.get("name") or item.get("id") or "オープンデータ")


def _dataset_source_path(item: dict, requested_url: str) -> str:
    """
    row_data.content に入れる source_path
    優先順:
    1. resources[0].url
    2. datasetページURL
    3. requested_url
    """
    resources = item.get("resources") or []
    if isinstance(resources, list):
        for r in resources:
            if isinstance(r, dict):
                u = str(r.get("url") or "").strip()
                if u:
                    return u

    dataset_name = str(item.get("name") or item.get("id") or "").strip()
    if dataset_name:
        return f"https://data.e-gov.go.jp/data/dataset/{dataset_name}"

    return requested_url


def _build_logical_name(items: list[dict], requested_url: str) -> str:
    if not items:
        return "オープンデータ"

    first = items[0]
    title = _dataset_title(first)
    return title if title else requested_url


def _ensure_tables(cur: sqlite3.Cursor):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS source_documents (
            source_id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            logical_name TEXT NOT NULL,
            original_name TEXT,
            ext TEXT,
            source_key TEXT,
            source_item_id TEXT NOT NULL,
            source_url TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS ix_source_documents_created_at
        ON source_documents(created_at DESC)
    """)

    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_source_documents_item
        ON source_documents(source_type, source_item_id)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS row_data (
            row_id TEXT PRIMARY KEY,
            file_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_key TEXT,
            source_item_id TEXT,
            row_index INTEGER,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_row_data_file_id
        ON row_data(file_id)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_row_data_source_type
        ON row_data(source_type)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_row_data_source_key
        ON row_data(source_key)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_row_data_file_row
        ON row_data(file_id, row_index)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_row_data_type_created
        ON row_data(source_type, created_at DESC)
    """)

    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_row_data_source_item
        ON row_data(source_type, source_item_id)
    """)


def _opendata_fetch_and_register_impl(authorization: str | None):
    uid = get_uid_from_auth_header(authorization)

    tpl = load_opendata_template()

    ds = tpl.get("dataset_search") or {}
    endpoint = ds.get("endpoint")
    params = ds.get("params") or {}

    if not isinstance(endpoint, str) or not endpoint:
        endpoint, params = _resolve_template_endpoint_and_params(tpl)

    if not isinstance(params, dict):
        params = {}

    max_items = _get_dataset_limit_from_template(tpl)

    total_count, items, requested_url = _fetch_ckan_package_search_all(endpoint, params, max_items)

    fetched = len(items)
    if fetched == 0:
        return {
            "mode": "fetch_and_register",
            "requested_url": requested_url,
            "total_count": total_count,
            "fetched": 0,
            "inserted": 0,
            "skipped": 0,
        }

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}",
        )

    local_db_path = f"/tmp/ank_{uid}_opendata.db"
    db_blob.download_to_filename(local_db_path)

    file_id = str(ulid.new())
    created_at = datetime.now(tz=JST).isoformat()

    source_type = "opendata"
    source_key = OPENDATA_TEMPLATE_PATH
    source_item_id = requested_url
    source_url = requested_url
    logical_name = _build_logical_name(items, requested_url)
    original_name = logical_name
    ext = "json"

    inserted = 0
    skipped = 0

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()
        _ensure_tables(cur)

        cur.execute("""
            SELECT source_id
            FROM source_documents
            WHERE source_type = ? AND source_item_id = ?
            LIMIT 1
        """, (source_type, source_item_id))
        exists = cur.fetchone()

        if exists:
            raise HTTPException(
                status_code=409,
                detail="同じ取得条件のオープンデータは登録済みです"
            )

        cur.execute(
            """
            INSERT INTO source_documents
              (source_id, source_type, logical_name, original_name, ext, source_key, source_item_id, source_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                source_type,
                logical_name,
                original_name,
                ext,
                source_key,
                source_item_id,
                source_url,
                created_at,
            ),
        )

        row_index = 1
        for item in items:
            did = _dataset_id(item)
            if not did:
                skipped += 1
                row_index += 1
                continue

            row_id = str(ulid.new())
            source_path = _dataset_source_path(item, requested_url)

            content_obj = {
                "dataset_id": did,
                "title": _dataset_title(item),
                "source_path": source_path,
                "data": item,
            }
            content = json.dumps(content_obj, ensure_ascii=False)

            cur.execute(
                """
                INSERT OR IGNORE INTO row_data
                  (row_id, file_id, source_type, source_key, source_item_id, row_index, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    file_id,
                    source_type,
                    source_key,
                    did,
                    row_index,
                    content,
                    created_at,
                ),
            )

            if cur.rowcount == 1:
                inserted += 1
            else:
                skipped += 1

            row_index += 1

        conn.commit()
    finally:
        conn.close()

    db_blob.upload_from_filename(local_db_path)

    return {
        "mode": "fetch_and_register",
        "requested_url": requested_url,
        "file_id": file_id,
        "logical_name": logical_name,
        "count": fetched,
        "total_count": total_count,
        "fetched": fetched,
        "inserted": inserted,
        "skipped": skipped,
    }


@router.get("/fetch_and_register")
def opendata_fetch_and_register_get(authorization: str | None = Header(default=None)):
    return _opendata_fetch_and_register_impl(authorization)


@router.post("/fetch_and_register")
def opendata_fetch_and_register_post(authorization: str | None = Header(default=None)):
    return _opendata_fetch_and_register_impl(authorization)


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
