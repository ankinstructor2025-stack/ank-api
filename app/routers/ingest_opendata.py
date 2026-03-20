# app/routers/opendata.py

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse, unquote
from zoneinfo import ZoneInfo

import firebase_admin
import requests
import ulid
from fastapi import APIRouter, Header, HTTPException
from firebase_admin import auth as fb_auth
from google.cloud import storage

from .content_detector import detect_resource_kind, guess_ext_from_url, normalize_text

router = APIRouter(prefix="/opendata", tags=["opendata"])

JST = ZoneInfo("Asia/Tokyo")

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
OPENDATA_TEMPLATE_PATH = "template/opendata.json"


ALLOWED_EXTS = {"pdf", "csv", "json"}


def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


def opendata_object_path(uid: str, source_id: str, ext: str) -> str:
    return f"users/{uid}/uploads/{source_id}.{ext}"


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
        raise HTTPException(status_code=401, detail=f"Invalid ID token: {e}")


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


def now_iso() -> str:
    return datetime.now(tz=JST).isoformat()


def safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def get_fetch_max_total(tpl: dict) -> int:
    fetch_cfg = tpl.get("fetch") or {}
    ds = tpl.get("dataset_search") or {}
    params = ds.get("params") or {}

    n = safe_int(fetch_cfg.get("max_total", params.get("rows", 10)), 10)
    return max(1, n)


def get_resource_filter_formats(tpl: dict) -> list[str]:
    rf = tpl.get("resource_filter") or {}
    formats = rf.get("formats") or []
    if not isinstance(formats, list):
        return []
    out = []
    for x in formats:
        v = str(x).strip().lower()
        if v:
            out.append(v)
    return out


def get_resource_limit(tpl: dict) -> int:
    rf = tpl.get("resource_filter") or {}
    n = safe_int(rf.get("limit_resources", 0), 0)
    return max(0, n)


def _fetch_json(url: str, params: dict | None = None) -> Tuple[dict, str]:
    try:
        res = requests.get(url, params=params or {}, timeout=30)
        res.raise_for_status()
        return res.json(), res.url
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"opendata api error: {str(e)}")


def _fetch_binary(url: str) -> Tuple[bytes, str, str]:
    try:
        res = requests.get(url, timeout=60)
        res.raise_for_status()
        return res.content, res.url, (res.headers.get("Content-Type") or "")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"resource fetch error: {str(e)}")


def _normalize_ckan_result(payload: dict) -> object:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="invalid ckan response")

    if "success" not in payload or "result" not in payload:
        raise HTTPException(status_code=502, detail="invalid ckan response shape")

    if not payload.get("success", False):
        raise HTTPException(status_code=502, detail="ckan api error: success=false")

    return payload.get("result")


def _dataset_search_all_from_template(tpl: dict) -> Tuple[List[dict], str]:
    ds = tpl.get("dataset_search") or {}
    endpoint = ds.get("endpoint")
    params = dict(ds.get("params") or {})

    if not endpoint:
        raise HTTPException(status_code=500, detail="template missing dataset_search.endpoint")

    max_total = get_fetch_max_total(tpl)

    rows = safe_int(params.get("rows", 10), 10)
    rows = max(1, rows)

    start = safe_int(params.get("start", 0), 0)
    start = max(0, start)

    requested_url_last = endpoint
    collected: List[dict] = []

    max_scan_pages = 10
    scanned_pages = 0

    while len(collected) < max_total and scanned_pages < max_scan_pages:
        p = dict(params)
        p["rows"] = rows
        p["start"] = start

        payload, requested_url_last = _fetch_json(endpoint, p)
        result = _normalize_ckan_result(payload)

        if not isinstance(result, dict):
            raise HTTPException(status_code=502, detail="unexpected dataset_search result")

        items = result.get("results") or []
        if not isinstance(items, list) or len(items) == 0:
            break

        for item in items:
            if not isinstance(item, dict):
                continue

            allowed_resources = filter_allowed_resources(item, tpl)
            if len(allowed_resources) == 0:
                continue

            collected.append(item)
            if len(collected) >= max_total:
                break

        total_count = safe_int(result.get("count", 0), 0)

        if len(collected) >= max_total:
            break

        if total_count and start + rows >= total_count:
            break

        start += rows
        scanned_pages += 1

    return collected, requested_url_last


def dataset_id_of(ds: dict) -> str:
    return normalize_text(ds.get("id") or ds.get("name"))


def dataset_title_of(ds: dict) -> str:
    return normalize_text(ds.get("title") or ds.get("name") or ds.get("id") or "オープンデータ")


def resource_meta_ext(resource: dict) -> str:
    fmt = normalize_text(resource.get("format")).lower()
    if fmt:
        return fmt
    return guess_ext_from_url(normalize_text(resource.get("url")))


def resource_allowed_by_template(resource: dict, tpl: dict, source_path: str, content_type: str = "") -> bool:
    allowed_formats = set(get_resource_filter_formats(tpl))
    if not allowed_formats:
        return True

    fmt = normalize_text(resource.get("format")).lower()
    kind = detect_resource_kind(resource, source_path, content_type)
    meta_ext = resource_meta_ext(resource)

    return fmt in allowed_formats or kind in allowed_formats or meta_ext in allowed_formats


def filter_allowed_resources(ds: dict, tpl: dict) -> List[dict]:
    resources = ds.get("resources") or []
    if not isinstance(resources, list):
        return []

    allowed_formats = set(get_resource_filter_formats(tpl))
    limit_resources = get_resource_limit(tpl)

    matched: List[dict] = []

    for resource in resources:
        if not isinstance(resource, dict):
            continue

        ext = resource_meta_ext(resource)
        if allowed_formats and ext not in allowed_formats:
            continue

        matched.append(resource)

        if limit_resources > 0 and len(matched) >= limit_resources:
            break

    return matched


def infer_ext_from_resources(resources: List[dict], tpl: dict) -> Optional[str]:
    if not resources:
        return None

    preferred_order = get_resource_filter_formats(tpl) or ["csv", "json", "pdf"]
    found: List[str] = []
    for resource in resources:
        ext = resource_meta_ext(resource)
        if ext:
            found.append(ext)

    for preferred in preferred_order:
        if preferred in found:
            return preferred

    return found[0] if found else None


def find_dataset_by_id(datasets: List[dict], dataset_id: str) -> Optional[dict]:
    for ds in datasets:
        if dataset_id_of(ds) == dataset_id:
            return ds
    return None


def choose_resource_for_dataset(ds: dict, tpl: dict) -> tuple[Optional[dict], Optional[str]]:
    resources = filter_allowed_resources(ds, tpl)
    if not resources:
        return None, None

    preferred_order = get_resource_filter_formats(tpl) or ["csv", "json", "pdf"]

    for preferred in preferred_order:
        for resource in resources:
            ext = resource_meta_ext(resource)
            if ext == preferred:
                return resource, ext

    first = resources[0]
    return first, resource_meta_ext(first)


def guess_original_name(resource: dict, final_url: str, ext: str, logical_name: str) -> str:
    url_path = urlparse(final_url).path if final_url else ""
    candidate = os.path.basename(unquote(url_path)) if url_path else ""
    candidate = normalize_text(candidate)
    if candidate:
        return candidate

    name = normalize_text(resource.get("name") or resource.get("id") or logical_name or "opendata")
    if ext and not name.lower().endswith(f".{ext}"):
        return f"{name}.{ext}"
    return name or f"opendata.{ext or 'dat'}"


def summarize_dataset_from_api(ds: dict, tpl: dict) -> Optional[dict]:
    resource, ext = choose_resource_for_dataset(ds, tpl)
    if not resource or not ext:
        return None

    return {
        "dataset_id": dataset_id_of(ds),
        "title": dataset_title_of(ds),
        "ext": ext,
        "source_url": normalize_text(resource.get("url")),
        "original_name": guess_original_name(resource, normalize_text(resource.get("url")), ext, dataset_title_of(ds)),
    }


def upsert_dataset_headers(cur: sqlite3.Cursor, datasets: List[dict], tpl: dict, created_at: str) -> None:
    for ds in datasets:
        dataset_id = dataset_id_of(ds)
        if not dataset_id:
            continue

        summary = summarize_dataset_from_api(ds, tpl)
        if not summary:
            continue

        title = summary["title"]
        ext = summary["ext"]
        source_url = summary["source_url"]
        original_name = summary["original_name"]

        cur.execute(
            """
            SELECT source_id, status
            FROM opendata_documents
            WHERE source_item_id = ?
            LIMIT 1
            """,
            (dataset_id,),
        )
        row = cur.fetchone()

        if row:
            source_id = row[0]
            cur.execute(
                """
                UPDATE opendata_documents
                SET logical_name = ?,
                    original_name = ?,
                    source_key = ?,
                    source_url = ?,
                    ext = ?,
                    created_at = ?
                WHERE source_id = ?
                """,
                (title, original_name, dataset_id, source_url, ext, created_at, source_id),
            )
        else:
            cur.execute(
                """
                INSERT INTO opendata_documents
                  (source_id, status, logical_name, original_name, source_key, source_item_id, source_url, ext, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (str(ulid.new()), "new", title, original_name, dataset_id, dataset_id, source_url, ext, created_at),
            )


def list_registered_datasets(cur: sqlite3.Cursor) -> List[dict]:
    cur.execute(
        """
        SELECT
          source_id,
          status,
          logical_name,
          original_name,
          source_key,
          source_item_id,
          source_url,
          ext,
          created_at
        FROM opendata_documents
        ORDER BY created_at DESC
        """
    )

    rows = cur.fetchall()
    out = []
    for row in rows:
        out.append(
            {
                "source_id": row[0],
                "status": row[1],
                "title": row[2],
                "original_name": row[3],
                "dataset_id": row[5],
                "source_url": row[6],
                "ext": row[7],
                "created_at": row[8],
            }
        )
    return out


def download_selected_resource(ds: dict, tpl: dict) -> tuple[bytes, str, str, str]:
    resource, ext = choose_resource_for_dataset(ds, tpl)
    if not resource or not ext:
        raise HTTPException(status_code=400, detail="登録対象データがありません")

    source_url = normalize_text(resource.get("url"))
    if not source_url:
        raise HTTPException(status_code=400, detail="resource url not found")

    binary, final_url, content_type = _fetch_binary(source_url)
    if not resource_allowed_by_template(resource, tpl, final_url, content_type):
        raise HTTPException(status_code=400, detail="resource format not allowed")

    detected_ext = detect_resource_kind(resource, final_url, content_type) or ext or guess_ext_from_url(final_url)
    detected_ext = normalize_text(detected_ext).lower()
    if detected_ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"unsupported resource kind: {detected_ext}")

    original_name = guess_original_name(resource, final_url, detected_ext, dataset_title_of(ds))
    return binary, final_url, detected_ext, original_name


def _fetch_datasets_impl(authorization: str | None):
    uid = get_uid_from_auth_header(authorization)

    tpl = load_opendata_template()
    datasets, requested_url = _dataset_search_all_from_template(tpl)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}",
        )

    local_db_path = f"/tmp/ank_{uid}_opendata_fetch.db"
    db_blob.download_to_filename(local_db_path)

    created_at = now_iso()

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()
        upsert_dataset_headers(cur, datasets, tpl, created_at)
        conn.commit()
        items = list_registered_datasets(cur)
    finally:
        conn.close()

    db_blob.upload_from_filename(local_db_path)

    return {
        "mode": "fetch_datasets",
        "requested_url": requested_url,
        "dataset_count": len(items),
        "datasets": items,
    }


def _expand_dataset_impl(dataset_id: str, authorization: str | None):
    uid = get_uid_from_auth_header(authorization)

    tpl = load_opendata_template()
    datasets, _ = _dataset_search_all_from_template(tpl)

    ds = find_dataset_by_id(datasets, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="dataset not found")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}",
        )

    local_db_path = f"/tmp/ank_{uid}_opendata_expand_{dataset_id}.db"
    db_blob.download_to_filename(local_db_path)

    created_at = now_iso()
    binary, final_url, ext, original_name = download_selected_resource(ds, tpl)

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT source_id, status, logical_name
            FROM opendata_documents
            WHERE source_item_id = ?
            LIMIT 1
            """,
            (dataset_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="dataset not registered in opendata_documents")

        source_id, prev_status, logical_name = row
        object_path = opendata_object_path(uid, source_id, ext)
        blob = bucket.blob(object_path)
        blob.upload_from_string(binary)

        cur.execute(
            """
            UPDATE opendata_documents
            SET status = ?,
                original_name = ?,
                source_url = ?,
                ext = ?,
                created_at = ?
            WHERE source_id = ?
            """,
            ("done", original_name, final_url, ext, created_at, source_id),
        )
        conn.commit()

        cur.execute(
            """
            SELECT source_id, status, logical_name, original_name, source_url, ext, created_at
            FROM opendata_documents
            WHERE source_id = ?
            LIMIT 1
            """,
            (source_id,),
        )
        saved = cur.fetchone()
    finally:
        conn.close()

    db_blob.upload_from_filename(local_db_path)

    return {
        "mode": "expand_dataset",
        "dataset_id": dataset_id,
        "source_id": saved[0] if saved else None,
        "status": saved[1] if saved else None,
        "logical_name": saved[2] if saved else logical_name,
        "original_name": saved[3] if saved else original_name,
        "source_url": saved[4] if saved else final_url,
        "ext": saved[5] if saved else ext,
        "gcs_path": opendata_object_path(uid, source_id, ext),
        "previous_status": prev_status,
        "created_at": saved[6] if saved else created_at,
    }


def _list_documents_impl(authorization: str | None):
    uid = get_uid_from_auth_header(authorization)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}",
        )

    local_db_path = f"/tmp/ank_{uid}_opendata_documents.db"
    db_blob.download_to_filename(local_db_path)

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()
        rows = list_registered_datasets(cur)
    finally:
        conn.close()

    return {
        "mode": "documents",
        "dataset_count": len(rows),
        "datasets": rows,
    }


@router.get("/fetch_datasets")
def opendata_fetch_datasets_get(
    authorization: str | None = Header(default=None),
):
    return _fetch_datasets_impl(authorization)


@router.post("/fetch_datasets")
def opendata_fetch_datasets_post(
    authorization: str | None = Header(default=None),
):
    return _fetch_datasets_impl(authorization)


@router.post("/expand_dataset")
def opendata_expand_dataset_post(
    dataset_id: str,
    authorization: str | None = Header(default=None),
):
    return _expand_dataset_impl(dataset_id, authorization)


@router.get("/documents")
def opendata_documents_get(
    authorization: str | None = Header(default=None),
):
    return _list_documents_impl(authorization)
