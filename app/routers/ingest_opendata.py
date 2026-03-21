# app/routers/ingest_opendata.py

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, List, Optional, Tuple
from urllib.parse import unquote, urlparse
from zoneinfo import ZoneInfo

import firebase_admin
import requests
import ulid
from fastapi import APIRouter, Header, HTTPException, Response
from firebase_admin import auth as fb_auth
from google.cloud import storage

from .content_detector import detect_resource_kind, guess_ext_from_url, normalize_text

router = APIRouter(prefix="/opendata", tags=["opendata"])

JST = ZoneInfo("Asia/Tokyo")

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
OPENDATA_TEMPLATE_PATH = "template/opendata.json"


def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


def get_local_db_path(uid: str) -> str:
    return f"/tmp/ank_{uid}.db"


def opendata_child_object_path(uid: str, source_id: str, file_id: str, ext: str) -> str:
    safe_ext = normalize_text(ext).lower().strip(".") or "dat"
    return f"users/{uid}/opendata/{source_id}/{file_id}.{safe_ext}"


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


def now_iso() -> str:
    return datetime.now(tz=JST).isoformat()


def safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _template_error(message: str) -> HTTPException:
    return HTTPException(status_code=500, detail=f"invalid opendata template: {message}")


def require_dict(obj: Any, path: str) -> dict:
    if not isinstance(obj, dict):
        raise _template_error(f"{path} must be an object")
    return obj


def require_non_empty_string(obj: Any, path: str) -> str:
    value = normalize_text(obj)
    if not value:
        raise _template_error(f"{path} is required")
    return value


def require_non_negative_int(obj: Any, path: str) -> int:
    try:
        value = int(obj)
    except Exception:
        raise _template_error(f"{path} must be an integer")
    if value < 0:
        raise _template_error(f"{path} must be >= 0")
    return value


def require_positive_int(obj: Any, path: str) -> int:
    try:
        value = int(obj)
    except Exception:
        raise _template_error(f"{path} must be an integer")
    if value <= 0:
        raise _template_error(f"{path} must be > 0")
    return value


def validate_opendata_template(tpl: dict) -> dict:
    root = require_dict(tpl, "template")

    dataset_search = require_dict(root.get("dataset_search"), "dataset_search")
    endpoint = require_non_empty_string(dataset_search.get("endpoint"), "dataset_search.endpoint")
    params = require_dict(dataset_search.get("params"), "dataset_search.params")
    rows = require_positive_int(params.get("rows"), "dataset_search.params.rows")
    start = require_non_negative_int(params.get("start"), "dataset_search.params.start")

    fetch = require_dict(root.get("fetch"), "fetch")
    max_total = require_positive_int(fetch.get("max_total"), "fetch.max_total")

    resource_filter = require_dict(root.get("resource_filter"), "resource_filter")
    formats = resource_filter.get("formats")
    if not isinstance(formats, list) or len(formats) == 0:
        raise _template_error("resource_filter.formats must be a non-empty array")

    normalized_formats: list[str] = []
    for idx, fmt in enumerate(formats):
        value = normalize_text(fmt).lower()
        if not value:
            raise _template_error(f"resource_filter.formats[{idx}] must be a non-empty string")
        normalized_formats.append(value)

    validated = dict(root)
    validated["dataset_search"] = {
        "endpoint": endpoint,
        "params": {
            **params,
            "rows": rows,
            "start": start,
        },
    }
    validated["fetch"] = {
        "max_total": max_total,
    }
    validated["resource_filter"] = {
        "formats": normalized_formats,
    }
    return validated


def load_opendata_template() -> dict:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(OPENDATA_TEMPLATE_PATH)

    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"{OPENDATA_TEMPLATE_PATH} not found")

    text = blob.download_as_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"invalid template json: {e}")

    return validate_opendata_template(raw)


def get_fetch_max_total(tpl: dict) -> int:
    fetch_cfg = require_dict(tpl.get("fetch"), "fetch")
    return require_positive_int(fetch_cfg.get("max_total"), "fetch.max_total")


def get_resource_filter_formats(tpl: dict) -> list[str]:
    resource_filter = require_dict(tpl.get("resource_filter"), "resource_filter")
    formats = resource_filter.get("formats")
    if not isinstance(formats, list) or len(formats) == 0:
        raise _template_error("resource_filter.formats must be a non-empty array")

    out: list[str] = []
    for idx, x in enumerate(formats):
        v = normalize_text(x).lower()
        if not v:
            raise _template_error(f"resource_filter.formats[{idx}] must be a non-empty string")
        out.append(v)
    return out


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
    ds = require_dict(tpl.get("dataset_search"), "dataset_search")
    endpoint = require_non_empty_string(ds.get("endpoint"), "dataset_search.endpoint")
    params = require_dict(ds.get("params"), "dataset_search.params")

    max_total = get_fetch_max_total(tpl)

    rows = require_positive_int(params.get("rows"), "dataset_search.params.rows")
    start = require_non_negative_int(params.get("start"), "dataset_search.params.start")

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


def resource_meta_ext(resource: dict) -> str:
    fmt = normalize_text(resource.get("format")).lower()
    if fmt:
        return fmt
    return normalize_text(guess_ext_from_url(normalize_text(resource.get("url")))).lower()


def resource_allowed_by_template(resource: dict, tpl: dict, source_path: str, content_type: str = "") -> bool:
    allowed_formats = set(get_resource_filter_formats(tpl))
    fmt = normalize_text(resource.get("format")).lower()
    kind = normalize_text(detect_resource_kind(resource, source_path, content_type)).lower()
    meta_ext = normalize_text(resource_meta_ext(resource)).lower()

    return fmt in allowed_formats or kind in allowed_formats or meta_ext in allowed_formats


def filter_allowed_resources(ds: dict, tpl: dict) -> List[dict]:
    resources = ds.get("resources") or []
    if not isinstance(resources, list):
        return []

    allowed_formats = set(get_resource_filter_formats(tpl))
    matched: List[dict] = []

    for resource in resources:
        if not isinstance(resource, dict):
            continue

        ext = normalize_text(resource_meta_ext(resource)).lower()
        fmt = normalize_text(resource.get("format")).lower()

        if ext in allowed_formats or fmt in allowed_formats:
            matched.append(resource)

    return matched


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
    resources = filter_allowed_resources(ds, tpl)
    if not resources:
        return None

    title = dataset_title_of(ds)
    first_resource = resources[0]
    source_url = normalize_text(first_resource.get("url"))

    return {
        "dataset_id": dataset_id_of(ds),
        "title": title,
        "child_count": len(resources),
        "source_url": source_url,
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
        child_count = summary["child_count"]
        source_url = summary["source_url"]

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
                SET source_key = ?,
                    source_url = ?,
                    child_count = ?
                WHERE source_id = ?
                """,
                (title, source_url, child_count, source_id),
            )
        else:
            cur.execute(
                """
                INSERT INTO opendata_documents
                  (source_id, status, child_count, source_key, source_item_id, source_url, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(ulid.new()),
                    "new",
                    child_count,
                    title,
                    dataset_id,
                    source_url,
                    created_at,
                ),
            )


def list_registered_datasets(cur: sqlite3.Cursor) -> List[dict]:
    cur.execute(
        """
        SELECT
          source_id,
          status,
          child_count,
          source_key,
          source_item_id,
          source_url,
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
                "child_count": row[2],
                "title": row[3],
                "dataset_id": row[4],
                "source_url": row[5],
                "created_at": row[6],
            }
        )
    return out


def list_document_files(cur: sqlite3.Cursor, source_id: str) -> List[dict]:
    cur.execute(
        """
        SELECT
          file_id,
          source_id,
          file_no,
          logical_name,
          original_name,
          source_url,
          gcs_path,
          ext,
          file_size,
          created_at
        FROM opendata_document_files
        WHERE source_id = ?
        ORDER BY file_no
        """,
        (source_id,),
    )

    rows = cur.fetchall()
    out = []
    for row in rows:
        out.append(
            {
                "file_id": row[0],
                "source_id": row[1],
                "file_no": row[2],
                "logical_name": row[3],
                "original_name": row[4],
                "source_url": row[5],
                "gcs_path": row[6],
                "ext": row[7],
                "file_size": row[8],
                "created_at": row[9],
            }
        )
    return out


def download_resource(resource: dict, tpl: dict, logical_name: str) -> tuple[bytes, str, str, str]:
    source_url = normalize_text(resource.get("url"))
    if not source_url:
        raise HTTPException(status_code=400, detail="resource url not found")

    binary, final_url, content_type = _fetch_binary(source_url)
    if not resource_allowed_by_template(resource, tpl, final_url, content_type):
        raise HTTPException(status_code=400, detail="resource format not allowed")

    allowed_formats = set(get_resource_filter_formats(tpl))

    detected_ext = (
        detect_resource_kind(resource, final_url, content_type)
        or resource_meta_ext(resource)
        or guess_ext_from_url(final_url)
    )
    detected_ext = normalize_text(detected_ext).lower()

    if detected_ext not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"許可されていない形式です: {detected_ext}"
        )

    original_name = guess_original_name(resource, final_url, detected_ext, logical_name)
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

    local_db_path = get_local_db_path(uid)
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

    resources = filter_allowed_resources(ds, tpl)
    if not resources:
        raise HTTPException(status_code=400, detail="登録対象データがありません")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}",
        )

    local_db_path = get_local_db_path(uid)
    db_blob.download_to_filename(local_db_path)

    created_at = now_iso()
    logical_name = dataset_title_of(ds)

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()

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
        if not row:
            raise HTTPException(status_code=404, detail="dataset not registered in opendata_documents")

        source_id, prev_status = row

        cur.execute("DELETE FROM opendata_document_files WHERE source_id = ?", (source_id,))
        conn.commit()

        saved_files = []
        file_no = 0

        for resource in resources:
            try:
                binary, final_url, ext, original_name = download_resource(resource, tpl, logical_name)
            except Exception:
                continue

            file_no += 1
            file_id = str(ulid.new())
            gcs_path = opendata_child_object_path(uid, source_id, file_id, ext)

            blob = bucket.blob(gcs_path)
            blob.upload_from_string(binary)

            cur.execute(
                """
                INSERT INTO opendata_document_files
                  (file_id, source_id, file_no, logical_name, original_name, source_url, gcs_path, ext, file_size, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    source_id,
                    file_no,
                    logical_name,
                    original_name,
                    final_url,
                    gcs_path,
                    ext,
                    len(binary),
                    created_at,
                ),
            )

            saved_files.append(
                {
                    "file_id": file_id,
                    "file_no": file_no,
                    "logical_name": logical_name,
                    "original_name": original_name,
                    "source_url": final_url,
                    "gcs_path": gcs_path,
                    "ext": ext,
                    "file_size": len(binary),
                    "created_at": created_at,
                }
            )

        new_status = "done" if saved_files else "error"
        top_source_url = saved_files[0]["source_url"] if saved_files else normalize_text(ds.get("url"))

        cur.execute(
            """
            UPDATE opendata_documents
            SET status = ?,
                child_count = ?,
                source_key = ?,
                source_url = ?,
                created_at = ?
            WHERE source_id = ?
            """,
            (new_status, len(saved_files), logical_name, top_source_url, created_at, source_id),
        )
        conn.commit()
    finally:
        conn.close()

    db_blob.upload_from_filename(local_db_path)

    return {
        "mode": "expand_dataset",
        "dataset_id": dataset_id,
        "source_id": source_id,
        "status": new_status,
        "title": logical_name,
        "child_count": len(saved_files),
        "previous_status": prev_status,
        "created_at": created_at,
        "files": saved_files,
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

    local_db_path = get_local_db_path(uid)
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


def _document_files_impl(source_id: str, authorization: str | None):
    uid = get_uid_from_auth_header(authorization)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(status_code=400, detail=f"ank.db not found. path={db_gcs_path}")

    local_db_path = get_local_db_path(uid)
    db_blob.download_to_filename(local_db_path)

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT source_id, status, child_count, source_key, source_item_id, source_url, created_at
            FROM opendata_documents
            WHERE source_id = ?
            LIMIT 1
            """,
            (source_id,),
        )
        doc = cur.fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail=f"document not found: source_id={source_id}")

        files = list_document_files(cur, source_id)
    finally:
        conn.close()

    return {
        "mode": "document_files",
        "document": {
            "source_id": doc[0],
            "status": doc[1],
            "child_count": doc[2],
            "title": doc[3],
            "dataset_id": doc[4],
            "source_url": doc[5],
            "created_at": doc[6],
        },
        "file_count": len(files),
        "files": files,
    }


def _download_url_impl(file_id: str, authorization: str | None):
    uid = get_uid_from_auth_header(authorization)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(status_code=400, detail=f"ank.db not found. path={db_gcs_path}")

    local_db_path = get_local_db_path(uid)
    db_blob.download_to_filename(local_db_path)

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT file_id, original_name, gcs_path, ext
            FROM opendata_document_files
            WHERE file_id = ?
            LIMIT 1
            """,
            (file_id,),
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"file not found: file_id={file_id}")

    saved_file_id, original_name, gcs_path, ext = row
    blob = bucket.blob(gcs_path)

    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"stored file not found: path={gcs_path}")

    data = blob.download_as_bytes()

    media_type_map = {
        "pdf": "application/pdf",
        "csv": "text/csv; charset=utf-8",
        "json": "application/json; charset=utf-8",
    }
    media_type = media_type_map.get((ext or "").lower(), "application/octet-stream")
    filename = original_name or f"{saved_file_id}.{ext}"

    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def find_dataset_by_id(datasets: List[dict], dataset_id: str) -> Optional[dict]:
    for ds in datasets:
        if dataset_id_of(ds) == dataset_id:
            return ds
    return None


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


@router.get("/document_files")
def opendata_document_files_get(
    source_id: str,
    authorization: str | None = Header(default=None),
):
    return _document_files_impl(source_id, authorization)


@router.get("/download_url")
def opendata_download_url_get(
    file_id: str,
    authorization: str | None = Header(default=None),
):
    return _download_url_impl(file_id, authorization)
