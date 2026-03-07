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
from fastapi import APIRouter, HTTPException, Header
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


# =========================
# DB
# =========================

def ensure_tables(cur: sqlite3.Cursor):
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

    fetch_cfg = tpl.get("fetch") or {}
    try:
        max_total = int(fetch_cfg.get("max_total", params.get("rows", 10)))
    except Exception:
        max_total = 10
    if max_total < 1:
        max_total = 1

    try:
        rows = int(params.get("rows", 10))
    except Exception:
        rows = 10
    if rows < 1:
        rows = 10

    try:
        start = int(params.get("start", 0))
    except Exception:
        start = 0
    if start < 0:
        start = 0

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

        try:
            total_count = int(result.get("count", 0))
        except Exception:
            total_count = 0

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


def dataset_ext_of(ds: dict) -> str:
    # 親は dataset なので固定で json 扱い
    return "json"


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

    if fmt == "csv" or ext == "csv" or "csv" in mimetype or "csv" in content_type.lower():
        return "csv"

    if fmt == "json" or ext == "json" or "json" in mimetype or "json" in content_type.lower():
        return "json"

    if fmt == "pdf" or ext == "pdf" or "pdf" in mimetype or "pdf" in content_type.lower():
        return "pdf"

    return ""


# =========================
# row_data 分解
# =========================

def split_csv_records(binary: bytes) -> List[dict]:
    text = binary.decode("utf-8-sig", errors="replace")
    f = io.StringIO(text)
    reader = csv.DictReader(f)

    rows = []
    for row in reader:
        rows.append(dict(row))
    return rows


def split_json_records(binary: bytes) -> List[dict]:
    text = binary.decode("utf-8-sig", errors="replace")
    obj = json.loads(text)

    if isinstance(obj, list):
        return [x if isinstance(x, dict) else {"value": x} for x in obj]

    if isinstance(obj, dict):
        # よくある配列キーを優先
        for key in ["results", "items", "data", "records"]:
            v = obj.get(key)
            if isinstance(v, list):
                return [x if isinstance(x, dict) else {"value": x} for x in v]

        # 配列がなければ全体を1件
        return [obj]

    return [{"value": obj}]


def split_pdf_records(binary: bytes) -> List[dict]:
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

    return pages


def build_row_records_for_resource(resource: dict) -> Tuple[List[dict], str]:
    source_path = normalize_text(resource.get("url"))
    if not source_path:
        return [], ""

    binary, final_url, content_type = _fetch_binary(source_path)
    kind = detect_resource_kind(resource, final_url, content_type)

    if kind == "csv":
        return split_csv_records(binary), final_url

    if kind == "json":
        return split_json_records(binary), final_url

    if kind == "pdf":
        return split_pdf_records(binary), final_url

    return [], final_url


# =========================
# 登録本体
# =========================

def register_dataset_parent(
    cur: sqlite3.Cursor,
    ds: dict,
    requested_url: str,
    created_at: str,
) -> Tuple[Optional[str], bool]:
    """
    戻り値:
      (source_id, inserted_new)
    """
    dataset_id = dataset_id_of(ds)
    if not dataset_id:
        return None, False

    source_type = "opendata"
    source_id = str(ulid.new())
    logical_name = dataset_title_of(ds)
    original_name = logical_name
    ext = dataset_ext_of(ds)
    source_key = OPENDATA_TEMPLATE_PATH
    source_item_id = dataset_id
    source_url = dataset_page_url_of(ds, requested_url)

    cur.execute("""
        SELECT source_id
        FROM source_documents
        WHERE source_type = ? AND source_item_id = ?
        LIMIT 1
    """, (source_type, source_item_id))
    exists = cur.fetchone()

    if exists:
        return str(exists[0]), False

    cur.execute("""
        INSERT INTO source_documents
          (source_id, source_type, logical_name, original_name, ext, source_key, source_item_id, source_url, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        source_id,
        source_type,
        logical_name,
        original_name,
        ext,
        source_key,
        source_item_id,
        source_url,
        created_at,
    ))

    return source_id, True


def register_row_data_for_dataset(
    cur: sqlite3.Cursor,
    source_id: str,
    ds: dict,
    created_at: str,
) -> Tuple[int, int, int]:
    """
    戻り値:
      (resource_count, inserted_rows, skipped_rows)
    """
    dataset_id = dataset_id_of(ds)
    resources = ds.get("resources") or []
    if not isinstance(resources, list):
        resources = []

    inserted_rows = 0
    skipped_rows = 0
    resource_count = 0

    row_index = 1

    for idx, resource in enumerate(resources, start=1):
        if not isinstance(resource, dict):
            continue

        resource_id = resource_id_of(resource, idx)
        resource_name = resource_name_of(resource, idx)

        try:
            records, source_path = build_row_records_for_resource(resource)
        except Exception:
            skipped_rows += 1
            continue

        if not source_path:
            skipped_rows += 1
            continue

        if len(records) == 0:
            # CSV/JSON/PDF以外、または分解不能
            skipped_rows += 1
            continue

        resource_count += 1

        for sub_idx, rec in enumerate(records, start=1):
            row_source_item_id = f"{dataset_id}:{resource_id}:{sub_idx}"
            row_id = str(ulid.new())

            content_obj = {
                "dataset_id": dataset_id,
                "dataset_title": dataset_title_of(ds),
                "resource_id": resource_id,
                "resource_name": resource_name,
                "resource_format": normalize_text(resource.get("format")),
                "source_path": source_path,
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
                resource_id,
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

    return resource_count, inserted_rows, skipped_rows


def _opendata_fetch_and_register_impl(authorization: str | None):
    uid = get_uid_from_auth_header(authorization)
    tpl = load_opendata_template()

    datasets, requested_url = _dataset_search_all_from_template(tpl)

    if len(datasets) == 0:
        return {
            "mode": "fetch_and_register",
            "requested_url": requested_url,
            "dataset_count": 0,
            "parent_inserted": 0,
            "parent_skipped": 0,
            "resource_used": 0,
            "row_inserted": 0,
            "row_skipped": 0,
        }

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}"
        )

    local_db_path = f"/tmp/ank_{uid}_opendata.db"
    db_blob.download_to_filename(local_db_path)

    parent_inserted = 0
    parent_skipped = 0
    resource_used = 0
    row_inserted = 0
    row_skipped = 0

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()
        ensure_tables(cur)

        for ds in datasets:
            created_at = now_iso()

            source_id, inserted_new = register_dataset_parent(
                cur=cur,
                ds=ds,
                requested_url=requested_url,
                created_at=created_at,
            )

            if not source_id:
                parent_skipped += 1
                continue

            if inserted_new:
                parent_inserted += 1
            else:
                # 既存 dataset は row_data 再登録まで進まない
                parent_skipped += 1
                continue

            used_count, ins_rows, skip_rows = register_row_data_for_dataset(
                cur=cur,
                source_id=source_id,
                ds=ds,
                created_at=created_at,
            )

            resource_used += used_count
            row_inserted += ins_rows
            row_skipped += skip_rows

        conn.commit()

    finally:
        conn.close()

    db_blob.upload_from_filename(local_db_path)

    return {
        "mode": "fetch_and_register",
        "requested_url": requested_url,
        "dataset_count": len(datasets),
        "parent_inserted": parent_inserted,
        "parent_skipped": parent_skipped,
        "resource_used": resource_used,
        "row_inserted": row_inserted,
        "row_skipped": row_skipped,
    }


# =========================
# API
# =========================

@router.get("/fetch_and_register")
def opendata_fetch_and_register_get(
    authorization: str | None = Header(default=None),
):
    return _opendata_fetch_and_register_impl(authorization)


@router.post("/fetch_and_register")
def opendata_fetch_and_register_post(
    authorization: str | None = Header(default=None),
):
    return _opendata_fetch_and_register_impl(authorization)
