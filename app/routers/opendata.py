# app/routers/opendata.py

import os
import io
import csv
import json
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, List, Optional, Tuple
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


def make_id() -> str:
    return str(ulid.new())


def normalize_text(v: Any) -> str:
    return str(v or "").strip()


def safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def guess_ext_from_url(url: str) -> str:
    path = urlparse(url).path or ""
    if "." not in path:
        return ""
    return path.rsplit(".", 1)[-1].lower().strip()


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
    n = safe_int(df.get("max_rows", 2000), 2000)
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


def dataset_page_url_of(ds: dict, requested_url: str) -> str:
    dataset_name = normalize_text(ds.get("name") or ds.get("id"))
    if dataset_name:
        return f"https://data.e-gov.go.jp/data/dataset/{dataset_name}"
    return requested_url


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


def ensure_opendata_documents_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS opendata_documents (
          source_id TEXT PRIMARY KEY,
          status TEXT NOT NULL,
          logical_name TEXT NOT NULL,
          source_key TEXT,
          source_item_id TEXT NOT NULL,
          source_url TEXT,
          ext TEXT,
          row_count INTEGER,
          created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_opendata_documents_dataset_id
        ON opendata_documents(source_item_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_opendata_documents_status
        ON opendata_documents(status)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_opendata_documents_created_at
        ON opendata_documents(created_at)
        """
    )


def upsert_opendata_document(
    conn: sqlite3.Connection,
    dataset_id: str,
    logical_name: str,
    source_url: str,
    ext: Optional[str],
    row_count: int,
) -> str:
    created_at = now_iso()

    cur = conn.execute(
        """
        SELECT source_id
        FROM opendata_documents
        WHERE source_item_id = ?
        LIMIT 1
        """,
        (dataset_id,),
    )
    row = cur.fetchone()

    if row:
        source_id = row[0]
        conn.execute(
            """
            UPDATE opendata_documents
               SET status = ?,
                   logical_name = ?,
                   source_key = ?,
                   source_url = ?,
                   ext = ?,
                   row_count = ?,
                   created_at = ?
             WHERE source_id = ?
            """,
            (
                "new",
                logical_name,
                dataset_id,
                source_url,
                ext,
                row_count,
                created_at,
                source_id,
            ),
        )
        return str(source_id)

    source_id = make_id()
    conn.execute(
        """
        INSERT INTO opendata_documents (
          source_id,
          status,
          logical_name,
          source_key,
          source_item_id,
          source_url,
          ext,
          row_count,
          created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_id,
            "new",
            logical_name,
            dataset_id,
            dataset_id,
            source_url,
            ext,
            row_count,
            created_at,
        ),
    )
    return source_id


def collect_dataset_rows(ds: dict, requested_url: str, tpl: dict) -> Tuple[List[dict], Optional[str]]:
    dataset_id = dataset_id_of(ds)
    dataset_title = dataset_title_of(ds)
    dataset_url = dataset_page_url_of(ds, requested_url)

    allowed_resources = filter_allowed_resources(ds, tpl)
    detected_ext = infer_ext_from_resources(allowed_resources, tpl)

    out_rows: List[dict] = []
    row_index = 1

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

            out_rows.append(
                {
                    "source_item_id": row_source_item_id,
                    "row_index": row_index,
                    "content": json.dumps(content_obj, ensure_ascii=False),
                }
            )
            row_index += 1

    return out_rows, detected_ext


def replace_row_data_for_opendata(
    conn: sqlite3.Connection,
    file_id: str,
    source_key: str,
    rows_data: List[dict],
) -> Tuple[int, int]:
    conn.execute(
        """
        DELETE FROM row_data
        WHERE file_id = ?
          AND source_type = 'opendata'
        """,
        (file_id,),
    )

    inserted = 0
    skipped = 0
    created_at = now_iso()

    for row in rows_data:
        row_id = make_id()

        conn.execute(
            """
            INSERT OR IGNORE INTO row_data
              (row_id, file_id, source_type, source_key, source_item_id, row_index, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row_id,
                file_id,
                "opendata",
                source_key,
                row["source_item_id"],
                row["row_index"],
                row["content"],
                created_at,
            ),
        )

        if conn.execute("SELECT changes()").fetchone()[0] == 1:
            inserted += 1
        else:
            skipped += 1

    return inserted, skipped


def _fetch_and_register_impl(authorization: str | None):
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

    local_db_path = f"/tmp/ank_{uid}_opendata.db"
    db_blob.download_to_filename(local_db_path)

    total_inserted = 0
    total_skipped = 0
    document_count = 0
    source_ids: List[str] = []

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()

        ensure_opendata_documents_table(conn)

        for ds in datasets:
            dataset_id = dataset_id_of(ds)
            if not dataset_id:
                continue

            logical_name = dataset_title_of(ds)
            source_url = dataset_page_url_of(ds, requested_url)

            rows_data, detected_ext = collect_dataset_rows(ds, requested_url, tpl)
            if len(rows_data) == 0:
                continue

            source_id = upsert_opendata_document(
                conn=conn,
                dataset_id=dataset_id,
                logical_name=logical_name,
                source_url=source_url,
                ext=detected_ext,
                row_count=len(rows_data),
            )
            source_ids.append(source_id)
            document_count += 1

            inserted, skipped = replace_row_data_for_opendata(
                conn=conn,
                file_id=source_id,
                source_key=dataset_id,
                rows_data=rows_data,
            )

            total_inserted += inserted
            total_skipped += skipped

            cur.execute(
                """
                SELECT COUNT(*)
                FROM row_data
                WHERE file_id = ?
                  AND source_type = 'opendata'
                """,
                (source_id,),
            )
            actual_row_count = cur.fetchone()[0]

            cur.execute(
                """
                UPDATE opendata_documents
                   SET status = ?,
                       row_count = ?,
                       ext = ?
                 WHERE source_id = ?
                """,
                ("done", actual_row_count, detected_ext, source_id),
            )

        conn.commit()

    finally:
        conn.close()

    db_blob.upload_from_filename(local_db_path)

    return {
        "mode": "fetch_and_register",
        "requested_url": requested_url,
        "document_count": document_count,
        "inserted": total_inserted,
        "skipped": total_skipped,
        "source_ids": source_ids,
    }


@router.post("/fetch_and_register")
def opendata_fetch_and_register(
    authorization: str | None = Header(default=None),
):
    """
    template/opendata.json の条件で dataset を取得し、
    対象 resource を分解して row_data.file_id = opendata_documents.source_id で入れ直す
    """
    return _fetch_and_register_impl(authorization)


@router.get("/fetch_datasets")
def opendata_fetch_datasets_get(
    authorization: str | None = Header(default=None),
):
    return _fetch_and_register_impl(authorization)


@router.post("/fetch_datasets")
def opendata_fetch_datasets_post(
    authorization: str | None = Header(default=None),
):
    return _fetch_and_register_impl(authorization)


@router.get("/documents")
def opendata_documents(
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)

    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}"
        )

    local_db_path = f"/tmp/ank_{uid}_opendata_documents.db"
    db_blob.download_to_filename(local_db_path)

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row

    try:
        ensure_opendata_documents_table(conn)

        cur = conn.execute(
            """
            SELECT
              d.source_id,
              d.status,
              d.logical_name AS title,
              d.source_key AS dataset_id,
              d.source_item_id,
              d.source_url AS dataset_url,
              d.ext,
              COUNT(r.row_id) AS row_count,
              d.created_at
            FROM opendata_documents d
            LEFT JOIN row_data r
              ON r.file_id = d.source_id
             AND r.source_type = 'opendata'
            GROUP BY
              d.source_id,
              d.status,
              d.logical_name,
              d.source_key,
              d.source_item_id,
              d.source_url,
              d.ext,
              d.created_at
            HAVING COUNT(r.row_id) > 0
            ORDER BY d.created_at DESC
            """
        )

        rows = [dict(r) for r in cur.fetchall()]

    finally:
        conn.close()

    return {
        "rows": rows,
        "count": len(rows),
    }


@router.get("/rows")
def opendata_rows(
    source_id: str,
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}"
        )

    local_db_path = f"/tmp/ank_{uid}_opendata_rows.db"
    db_blob.download_to_filename(local_db_path)

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row
    try:
        ensure_opendata_documents_table(conn)

        cur = conn.execute(
            """
            SELECT source_id
            FROM opendata_documents
            WHERE source_id = ?
            LIMIT 1
            """,
            (source_id,),
        )
        doc = cur.fetchone()

        if not doc:
            return {"rows": [], "count": 0}

        cur = conn.execute(
            """
            SELECT
              row_id,
              file_id,
              source_type,
              source_key,
              source_item_id,
              row_index,
              content,
              created_at
            FROM row_data
            WHERE file_id = ?
              AND source_type = 'opendata'
            ORDER BY row_index
            """,
            (source_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]

    finally:
        conn.close()

    return {
        "rows": rows,
        "count": len(rows),
    }
