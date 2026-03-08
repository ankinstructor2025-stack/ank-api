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
    n = safe_int(rf.get("limit_resources", 0), 0)
    return max(0, n)


def get_data_fetch_max_rows(tpl: dict) -> int:
    df = tpl.get("data_fetch") or {}
    n = safe_int(df.get("max_rows", 200), 200)
    return max(1, n)


def get_data_fetch_encoding(tpl: dict) -> str:
    df = tpl.get("data_fetch") or {}
    enc = str(df.get("encoding") or "utf-8").strip()
    return enc or "utf-8"


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

    return collected, requested_url_last


def dataset_id_of(ds: dict) -> str:
    return normalize_text(ds.get("id") or ds.get("name"))


def dataset_title_of(ds: dict) -> str:
    return normalize_text(ds.get("title") or ds.get("name") or ds.get("id") or "オープンデータ")


def dataset_page_url_of(ds: dict, requested_url: str) -> str:
    dataset_name = normalize_text(ds.get("name") or ds.get("id"))
    if dataset_name:
        return f"https://data.e-gov.go.jp/data/dataset/{dataset_name}"
    return requested_url


def ckan_datastore_search_endpoint(dataset_search_endpoint: str) -> Optional[str]:
    if not dataset_search_endpoint:
        return None

    if dataset_search_endpoint.endswith("/package_search"):
        return dataset_search_endpoint[:-len("/package_search")] + "/datastore_search"

    if "/package_search" in dataset_search_endpoint:
        return dataset_search_endpoint.replace("/package_search", "/datastore_search")

    return None


def resource_meta_ext(resource: dict) -> str:
    fmt = normalize_text(resource.get("format")).lower()
    if fmt:
        return fmt
    return guess_ext_from_url(normalize_text(resource.get("url")))


def filter_allowed_resources(ds: dict, tpl: dict) -> List[dict]:
    resources = ds.get("resources") or []
    if not isinstance(resources, list):
        return []

    allowed_formats = get_resource_filter_formats(tpl)
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

    allowed_formats = list(get_resource_filter_formats(tpl))
    if not allowed_formats:
        allowed_formats = ["csv", "json", "pdf"]

    found: List[str] = []
    for resource in resources:
        ext = resource_meta_ext(resource)
        if ext:
            found.append(ext)

    for p in allowed_formats:
        if p in found:
            return p

    return found[0] if found else None


def try_get_row_count_from_metadata(resources: List[dict], tpl: dict) -> Optional[int]:
    ds_search = tpl.get("dataset_search") or {}
    dataset_search_endpoint = ds_search.get("endpoint") or ""
    datastore_endpoint = ckan_datastore_search_endpoint(dataset_search_endpoint)
    if not datastore_endpoint:
        return None

    for resource in resources:
        if not isinstance(resource, dict):
            continue

        if not resource.get("datastore_active", False):
            continue

        resource_id = normalize_text(resource.get("id"))
        if not resource_id:
            continue

        try:
            payload, _ = _fetch_json(datastore_endpoint, {"resource_id": resource_id, "limit": 0})
            result = _normalize_ckan_result(payload)
            if isinstance(result, dict):
                count = result.get("total") or result.get("count")
                if count is not None:
                    return safe_int(count, 0)
        except Exception:
            continue

    return None


def summarize_dataset_from_api(ds: dict, requested_url: str, tpl: dict) -> Optional[dict]:
    allowed_resources = filter_allowed_resources(ds, tpl)
    if len(allowed_resources) == 0:
        return None

    return {
        "dataset_id": dataset_id_of(ds),
        "title": dataset_title_of(ds),
        "dataset_url": dataset_page_url_of(ds, requested_url),
        "resource_count": len(allowed_resources),
        "ext": infer_ext_from_resources(allowed_resources, tpl),
        "row_count": try_get_row_count_from_metadata(allowed_resources, tpl),
    }


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
    meta_ext = resource_meta_ext(resource)

    if fmt in allowed_formats:
        return True

    if kind in allowed_formats:
        return True

    if meta_ext in allowed_formats:
        return True

    return False


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


def find_dataset_by_id(datasets: List[dict], dataset_id: str) -> Optional[dict]:
    for ds in datasets:
        if dataset_id_of(ds) == dataset_id:
            return ds
    return None


def upsert_dataset_headers(
    cur: sqlite3.Cursor,
    datasets: List[dict],
    requested_url: str,
    tpl: dict,
    created_at: str,
) -> None:
    for ds in datasets:
        dataset_id = dataset_id_of(ds)
        if not dataset_id:
            continue

        allowed_resources = filter_allowed_resources(ds, tpl)
        if len(allowed_resources) == 0:
            continue

        title = dataset_title_of(ds)
        dataset_url = dataset_page_url_of(ds, requested_url)
        ext = infer_ext_from_resources(allowed_resources, tpl)
        row_count = try_get_row_count_from_metadata(allowed_resources, tpl)

        cur.execute("""
            SELECT source_id, status, row_count, ext
            FROM opendata_documents
            WHERE source_item_id = ?
            LIMIT 1
        """, (dataset_id,))
        row = cur.fetchone()

        if row:
            source_id = row[0]
            old_row_count = row[2]
            old_ext = row[3]

            cur.execute("""
                UPDATE opendata_documents
                SET logical_name = ?,
                    source_key = ?,
                    source_url = ?,
                    ext = ?,
                    row_count = ?,
                    created_at = ?
                WHERE source_id = ?
            """, (
                title,
                dataset_id,
                dataset_url,
                ext or old_ext,
                row_count if row_count is not None else old_row_count,
                created_at,
                source_id,
            ))
        else:
            cur.execute("""
                INSERT INTO opendata_documents
                  (source_id, status, logical_name, source_key, source_item_id, source_url, ext, row_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(ulid.new()),
                "new",
                title,
                dataset_id,
                dataset_id,
                dataset_url,
                ext,
                row_count,
                created_at,
            ))


def list_registered_datasets(cur: sqlite3.Cursor) -> List[dict]:
    cur.execute("""
        SELECT
          source_id,
          status,
          logical_name,
          source_key,
          source_item_id,
          source_url,
          ext,
          row_count,
          created_at
        FROM opendata_documents
        ORDER BY created_at DESC
    """)

    rows = cur.fetchall()
    out = []
    for row in rows:
        out.append({
            "source_id": row[0],
            "status": row[1],
            "title": row[2],
            "dataset_id": row[4],
            "dataset_url": row[5],
            "ext": row[6],
            "row_count": row[7],
            "created_at": row[8],
        })
    return out


def expand_dataset_into_row_data(
    cur: sqlite3.Cursor,
    ds: dict,
    requested_url: str,
    tpl: dict,
    created_at: str,
) -> Tuple[int, int]:
    dataset_id = dataset_id_of(ds)
    dataset_title = dataset_title_of(ds)
    dataset_url = dataset_page_url_of(ds, requested_url)

    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id not found")

    cur.execute("""
        SELECT source_id, status
        FROM opendata_documents
        WHERE source_item_id = ?
        LIMIT 1
    """, (dataset_id,))
    row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="dataset not registered in opendata_documents")

    source_id, status = row

    if status == "done":
        raise HTTPException(status_code=409, detail="dataset already expanded")

    allowed_resources = filter_allowed_resources(ds, tpl)
    if len(allowed_resources) == 0:
        raise HTTPException(status_code=400, detail="登録対象データがありません")

    inserted_rows = 0
    skipped_rows = 0
    row_index = 1
    detected_ext = infer_ext_from_resources(allowed_resources, tpl)

    for resource in allowed_resources:
        resource_url = normalize_text(resource.get("url"))
        if not resource_url:
            continue

        records, final_url, kind = build_row_records_for_resource(resource, tpl)
        if not kind or len(records) == 0:
            continue

        resource_name = normalize_text(resource.get("name") or resource.get("id") or "resource")
        resource_id = normalize_text(resource.get("id") or resource_name)

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

    if inserted_rows == 0 and skipped_rows == 0:
        raise HTTPException(status_code=400, detail="登録対象データがありません")

    cur.execute("""
        UPDATE opendata_documents
        SET status = ?, row_count = ?, ext = ?
        WHERE source_id = ?
    """, (
        "done",
        inserted_rows,
        detected_ext,
        source_id,
    ))

    return inserted_rows, skipped_rows


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
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}"
        )

    local_db_path = f"/tmp/ank_{uid}_opendata_fetch.db"
    db_blob.download_to_filename(local_db_path)

    created_at = now_iso()

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()
        upsert_dataset_headers(cur, datasets, requested_url, tpl, created_at)
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
    datasets, requested_url = _dataset_search_all_from_template(tpl)

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
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}"
        )

    local_db_path = f"/tmp/ank_{uid}_opendata_expand_{dataset_id}.db"
    db_blob.download_to_filename(local_db_path)

    created_at = now_iso()

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()
        inserted_rows, skipped_rows = expand_dataset_into_row_data(
            cur=cur,
            ds=ds,
            requested_url=requested_url,
            tpl=tpl,
            created_at=created_at,
        )
        conn.commit()

        cur.execute("""
            SELECT source_id, status, row_count, ext
            FROM opendata_documents
            WHERE source_item_id = ?
            LIMIT 1
        """, (dataset_id,))
        row = cur.fetchone()
        source_id = row[0] if row else None
        status = row[1] if row else None
        row_count = row[2] if row else None
        ext = row[3] if row else None

    finally:
        conn.close()

    db_blob.upload_from_filename(local_db_path)

    return {
        "mode": "expand_dataset",
        "dataset_id": dataset_id,
        "source_id": source_id,
        "status": status,
        "row_count": row_count,
        "ext": ext,
        "row_inserted": inserted_rows,
        "row_skipped": skipped_rows,
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


@router.get("/expand_dataset")
def opendata_expand_dataset_get(
    dataset_id: str = Query(...),
    authorization: str | None = Header(default=None),
):
    return _expand_dataset_impl(dataset_id, authorization)


@router.post("/expand_dataset")
def opendata_expand_dataset_post(
    dataset_id: str = Query(...),
    authorization: str | None = Header(default=None),
):
    return _expand_dataset_impl(dataset_id, authorization)
