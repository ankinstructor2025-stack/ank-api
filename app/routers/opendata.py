# app/routers/opendata.py

import os
import io
import csv
import json
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from fastapi import APIRouter, HTTPException, Header, Query
from google.cloud import storage
from pypdf import PdfReader

import firebase_admin
from firebase_admin import auth as fb_auth

import ulid

router = APIRouter(prefix="/opendata", tags=["opendata"])

JST = ZoneInfo("Asia/Tokyo")

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
OPENDATA_TEMPLATE_PATH = "template/opendata.json"


# =========================
# 共通
# =========================

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


def guess_ext_from_url(url: str) -> str:
    path = urlparse(url).path or ""
    if "." not in path:
        return ""
    return path.rsplit(".", 1)[-1].lower().strip()


def normalize_text(v: Any) -> str:
    return str(v or "").strip()


def safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


# =========================
# template 読み取り
# =========================

def get_fetch_max_total(tpl: dict) -> int:
    fetch_cfg = tpl.get("fetch") or {}
    ds = tpl.get("dataset_search") or {}
    params = ds.get("params") or {}

    n = safe_int(fetch_cfg.get("max_total", params.get("rows", 10)), 10)
    return max(1, n)


def get_resource_filter_formats(tpl: dict) -> set[str]:
    rf = tpl.get("resource_filter") or {}
    formats = rf.get("formats") or []
    if not isinstance(formats, list):
        return set()
    return {str(x).strip().lower() for x in formats if str(x).strip()}


def get_resource_limit(tpl: dict) -> int:
    rf = tpl.get("resource_filter") or {}
    n = safe_int(rf.get("limit_resources", 1), 1)
    return max(1, n)


def get_data_fetch_max_rows(tpl: dict) -> int:
    df = tpl.get("data_fetch") or {}
    n = safe_int(df.get("max_rows", 200), 200)
    return max(1, n)


def get_data_fetch_encoding(tpl: dict) -> str:
    df = tpl.get("data_fetch") or {}
    enc = str(df.get("encoding") or "utf-8").strip()
    return enc or "utf-8"


# =========================
# CKAN
# =========================

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

    while len(collected) < max_total:
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

        remain = max_total - len(collected)
        for item in items[:remain]:
            if isinstance(item, dict):
                collected.append(item)

        total_count = safe_int(result.get("count", 0), 0)

        if len(collected) >= max_total:
            break

        if total_count and start + rows >= total_count:
            break

        start += rows

    return collected, requested_url_last


# =========================
# dataset / resource
# =========================

def dataset_id_of(ds: dict) -> str:
    return normalize_text(ds.get("id") or ds.get("name"))


def dataset_title_of(ds: dict) -> str:
    return normalize_text(ds.get("title") or ds.get("name") or ds.get("id") or "オープンデータ")


def dataset_page_url_of(ds: dict, requested_url: str) -> str:
    dataset_name = normalize_text(ds.get("name") or ds.get("id"))
    if dataset_name:
        return f"https://data.e-gov.go.jp/data/dataset/{dataset_name}"
    return requested_url


def resource_id_of(r: dict, fallback_index: int) -> str:
    rid = normalize_text(r.get("id"))
    if rid:
        return rid
    return f"resource_{fallback_index}"


def resource_name_of(r: dict, fallback_index: int) -> str:
    name = normalize_text(r.get("name"))
    if name:
        return name
    return f"resource_{fallback_index}"


def detect_resource_kind(resource: dict, source_path: str, content_type: str = "") -> str:
    fmt = normalize_text(resource.get("format")).lower()
    mimetype = normalize_text(resource.get("mimetype")).lower()
    ext = guess_ext_from_url(source_path)
    ctype = content_type.lower()

    if fmt == "csv" or ext == "csv" or "csv" in mimetype or "csv" in ctype:
        return "csv"

    if fmt == "json" or ext == "json" or "json" in mimetype or "json" in ctype:
        return "json"

    if fmt == "pdf" or ext == "pdf" or "pdf" in mimetype or "pdf" in ctype:
        return "pdf"

    return ""


def resource_allowed_by_template(resource: dict, tpl: dict, source_path: str, content_type: str = "") -> bool:
    allowed_formats = get_resource_filter_formats(tpl)
    if not allowed_formats:
        return True

    fmt = normalize_text(resource.get("format")).lower()
    kind = detect_resource_kind(resource, source_path, content_type)

    if fmt in allowed_formats:
        return True

    if kind in allowed_formats:
        return True

    return False


def summarize_dataset(ds: dict, requested_url: str) -> dict:
    resources = ds.get("resources") or []
    if not isinstance(resources, list):
        resources = []

    return {
        "dataset_id": dataset_id_of(ds),
        "title": dataset_title_of(ds),
        "dataset_url": dataset_page_url_of(ds, requested_url),
        "resource_count": len(resources),
    }


def summarize_resource(resource: dict, tpl: dict, fallback_index: int) -> dict:
    resource_url = normalize_text(resource.get("url"))
    kind = detect_resource_kind(resource, resource_url)
    allowed = resource_allowed_by_template(resource, tpl, resource_url)

    return {
        "resource_id": resource_id_of(resource, fallback_index),
        "resource_name": resource_name_of(resource, fallback_index),
        "resource_url": resource_url,
        "format": normalize_text(resource.get("format")),
        "kind": kind,
        "allowed": allowed,
    }


def find_dataset_by_id(datasets: List[dict], dataset_id: str) -> Optional[dict]:
    for ds in datasets:
        if dataset_id_of(ds) == dataset_id:
            return ds
    return None


def find_resource_by_id(ds: dict, resource_id: str) -> Tuple[Optional[dict], Optional[int]]:
    resources = ds.get("resources") or []
    if not isinstance(resources, list):
        return None, None

    for idx, resource in enumerate(resources, start=1):
        if not isinstance(resource, dict):
            continue
        if resource_id_of(resource, idx) == resource_id:
            return resource, idx

    return None, None


# =========================
# row_data 分解
# =========================

def split_csv_records(binary: bytes, encoding: str, max_rows: int) -> List[dict]:
    text = binary.decode(encoding, errors="replace")
    f = io.StringIO(text)
    reader = csv.DictReader(f)

    rows = []
    for row in reader:
        rows.append(dict(row))
        if len(rows) >= max_rows:
            break
    return rows


def split_json_records(binary: bytes, encoding: str, max_rows: int) -> List[dict]:
    text = binary.decode(encoding, errors="replace")
    obj = json.loads(text)

    if isinstance(obj, list):
        out = []
        for x in obj[:max_rows]:
            out.append(x if isinstance(x, dict) else {"value": x})
        return out

    if isinstance(obj, dict):
        for key in ["results", "items", "data", "records"]:
            v = obj.get(key)
            if isinstance(v, list):
                out = []
                for x in v[:max_rows]:
                    out.append(x if isinstance(x, dict) else {"value": x})
                return out

        return [obj]

    return [{"value": obj}]


def split_pdf_records(binary: bytes, max_rows: int) -> List[dict]:
    reader = PdfReader(io.BytesIO(binary))
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = ""
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        pages.append({
            "page": i,
            "text": text,
            "char_count": len(text),
        })

        if len(pages) >= max_rows:
            break

    return pages


def build_row_records_for_resource(resource: dict, tpl: dict) -> Tuple[List[dict], str, str]:
    source_path = normalize_text(resource.get("url"))
    if not source_path:
        return [], "", ""

    binary, final_url, content_type = _fetch_binary(source_path)

    if not resource_allowed_by_template(resource, tpl, final_url, content_type):
        return [], final_url, ""

    kind = detect_resource_kind(resource, final_url, content_type)
    if not kind:
        return [], final_url, ""

    max_rows = get_data_fetch_max_rows(tpl)
    encoding = get_data_fetch_encoding(tpl)

    if kind == "csv":
        return split_csv_records(binary, encoding, max_rows), final_url, kind

    if kind == "json":
        return split_json_records(binary, encoding, max_rows), final_url, kind

    if kind == "pdf":
        return split_pdf_records(binary, max_rows), final_url, kind

    return [], final_url, ""


# =========================
# DB登録
# =========================

def register_opendata_resource(
    cur: sqlite3.Cursor,
    ds: dict,
    resource: dict,
    resource_index: int,
    requested_url: str,
    tpl: dict,
    created_at: str,
) -> Tuple[str, int, int, str, str]:
    dataset_id = dataset_id_of(ds)
    dataset_title = dataset_title_of(ds)
    dataset_url = dataset_page_url_of(ds, requested_url)

    resource_id = resource_id_of(resource, resource_index)
    resource_name = resource_name_of(resource, resource_index)
    resource_url = normalize_text(resource.get("url"))

    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id not found")

    if not resource_id:
        raise HTTPException(status_code=400, detail="resource_id not found")

    if not resource_url:
        raise HTTPException(status_code=400, detail="resource url not found")

    # 親重複チェック（resource単位）
    cur.execute("""
        SELECT source_id
        FROM opendata_documents
        WHERE source_item_id = ?
        LIMIT 1
    """, (resource_id,))
    exists = cur.fetchone()
    if exists:
        raise HTTPException(status_code=409, detail="同じオープンデータ resource は登録済みです")

    records, final_url, kind = build_row_records_for_resource(resource, tpl)

    if not final_url:
        raise HTTPException(status_code=400, detail="resource url not found")

    if not kind:
        raise HTTPException(status_code=400, detail="対象外の形式です")

    if len(records) == 0:
        raise HTTPException(status_code=400, detail="登録対象データがありません")

    source_id = str(ulid.new())
    logical_name = f"{dataset_title} / {resource_name}"
    source_key = dataset_id
    ext = kind

    cur.execute("""
        INSERT INTO opendata_documents
          (source_id, logical_name, source_key, source_item_id, source_url, ext, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        source_id,
        logical_name,
        source_key,
        resource_id,
        final_url,
        ext,
        created_at,
    ))

    inserted_rows = 0
    skipped_rows = 0
    row_index = 1

    for sub_idx, rec in enumerate(records, start=1):
        row_source_item_id = f"{dataset_id}:{resource_id}:{sub_idx}"
        row_id = str(ulid.new())

        content_obj = {
            "dataset_id": dataset_id,
            "dataset_title": dataset_title,
            "dataset_url": dataset_url,
            "resource_id": resource_id,
            "resource_name": resource_name,
            "resource_format": normalize_text(resource.get("format")),
            "source_path": final_url,
            "record_index": sub_idx,
            "data": rec,
        }

        content = json.dumps(content_obj, ensure_ascii=False)

        cur.execute("""
            INSERT OR IGNORE INTO row_data
              (row_id, file_id, source_type, source_key, source_item_id, row_index, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row_id,
            source_id,
            "opendata",
            dataset_id,
            row_source_item_id,
            row_index,
            content,
            created_at,
        ))

        if cur.rowcount == 1:
            inserted_rows += 1
        else:
            skipped_rows += 1

        row_index += 1

    return source_id, inserted_rows, skipped_rows, logical_name, kind


# =========================
# 3段階API
# =========================

def _fetch_datasets_impl(authorization: str | None):
    get_uid_from_auth_header(authorization)

    tpl = load_opendata_template()
    datasets, requested_url = _dataset_search_all_from_template(tpl)

    items = [summarize_dataset(ds, requested_url) for ds in datasets]

    return {
        "mode": "fetch_datasets",
        "requested_url": requested_url,
        "dataset_count": len(items),
        "datasets": items,
    }


def _fetch_resources_impl(dataset_id: str, authorization: str | None):
    get_uid_from_auth_header(authorization)

    tpl = load_opendata_template()
    datasets, requested_url = _dataset_search_all_from_template(tpl)

    ds = find_dataset_by_id(datasets, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="dataset not found")

    resources = ds.get("resources") or []
    if not isinstance(resources, list):
        resources = []

    items = []
    for idx, resource in enumerate(resources, start=1):
        if not isinstance(resource, dict):
            continue
        items.append(summarize_resource(resource, tpl, idx))

    return {
        "mode": "fetch_resources",
        "requested_url": requested_url,
        "dataset": summarize_dataset(ds, requested_url),
        "resource_count": len(items),
        "resources": items,
    }


def _register_resource_impl(
    dataset_id: str,
    resource_id: str,
    authorization: str | None,
):
    uid = get_uid_from_auth_header(authorization)

    tpl = load_opendata_template()
    datasets, requested_url = _dataset_search_all_from_template(tpl)

    ds = find_dataset_by_id(datasets, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="dataset not found")

    resource, resource_index = find_resource_by_id(ds, resource_id)
    if not resource or resource_index is None:
        raise HTTPException(status_code=404, detail="resource not found")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}"
        )

    local_db_path = f"/tmp/ank_{uid}_opendata_{resource_id}.db"
    db_blob.download_to_filename(local_db_path)

    created_at = now_iso()

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()

        source_id, inserted_rows, skipped_rows, logical_name, kind = register_opendata_resource(
            cur=cur,
            ds=ds,
            resource=resource,
            resource_index=resource_index,
            requested_url=requested_url,
            tpl=tpl,
            created_at=created_at,
        )

        conn.commit()

    finally:
        conn.close()

    db_blob.upload_from_filename(local_db_path)

    return {
        "mode": "register_resource",
        "requested_url": requested_url,
        "source_id": source_id,
        "dataset_id": dataset_id,
        "resource_id": resource_id,
        "logical_name": logical_name,
        "ext": kind,
        "row_inserted": inserted_rows,
        "row_skipped": skipped_rows,
    }


# =========================
# API
# =========================

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


@router.get("/fetch_resources")
def opendata_fetch_resources_get(
    dataset_id: str = Query(...),
    authorization: str | None = Header(default=None),
):
    return _fetch_resources_impl(dataset_id, authorization)


@router.post("/fetch_resources")
def opendata_fetch_resources_post(
    dataset_id: str = Query(...),
    authorization: str | None = Header(default=None),
):
    return _fetch_resources_impl(dataset_id, authorization)


@router.get("/register_resource")
def opendata_register_resource_get(
    dataset_id: str = Query(...),
    resource_id: str = Query(...),
    authorization: str | None = Header(default=None),
):
    return _register_resource_impl(dataset_id, resource_id, authorization)


@router.post("/register_resource")
def opendata_register_resource_post(
    dataset_id: str = Query(...),
    resource_id: str = Query(...),
    authorization: str | None = Header(default=None),
):
    return _register_resource_impl(dataset_id, resource_id, authorization)
