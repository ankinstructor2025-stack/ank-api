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

# 既存の直書きは残す（/datasets/* が依存しているため）
CKAN_BASE = "https://data.e-gov.go.jp/data/api/action"  # e-Govデータポータル CKAN API base

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
OPENDATA_TEMPLATE_PATH = "template/opendata.json"


# kokkai.py と同じ
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


def _item_id(item: Any) -> str:
    # dictなら id / name、文字列ならそのまま
    if isinstance(item, dict):
        v = item.get("id") or item.get("name") or item.get("title")
        return str(v or "")
    return str(item or "")


# ----------------------------------------
# 取得 + row_data登録（kokkai と同じ思想）
# ----------------------------------------
@router.post("/fetch_and_register")
def opendata_fetch_and_register(authorization: str | None = Header(default=None)):
    uid = get_uid_from_auth_header(authorization)

    tpl = load_opendata_template()
    endpoint, params = _resolve_template_endpoint_and_params(tpl)

    payload, requested_url = _fetch_any_json(endpoint, params=params)
    result = _normalize_ckan_result(payload)
    count, items = _calc_count_and_items(result)

    if count == 0:
        return {
            "mode": "fetch_and_register",
            "requested_url": requested_url,
            "fetched": 0,
            "inserted": 0,
            "skipped": 0,
            "count": 0,
        }

    # --- ank.db を GCS から /tmp へ ---
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

    inserted = 0
    skipped = 0

    conn = sqlite3.connect(local_db_path)
    try:
        cur = conn.cursor()

        # uploaded_files に履歴を1件
        logical_name = f"opendata_{file_id[-6:]}"
        cur.execute(
            """
            INSERT INTO uploaded_files
              (file_id, logical_name, original_filename, ext, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (file_id, logical_name, OPENDATA_TEMPLATE_PATH, "api", created_at),
        )

        row_index = 1
        for item in items:
            iid = _item_id(item)
            if not iid:
                skipped += 1
                row_index += 1
                continue

            row_id = str(ulid.new())
            content = json.dumps(item, ensure_ascii=False)

            cur.execute(
                """
                INSERT OR IGNORE INTO row_data
                  (row_id, file_id, source_type, source_key, source_item_id, row_index, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (row_id, file_id, "opendata", OPENDATA_TEMPLATE_PATH, iid, row_index, content, created_at),
            )

            if cur.rowcount == 1:
                inserted += 1
            else:
                skipped += 1

            row_index += 1

        conn.commit()
    finally:
        conn.close()

    # --- GCSへ戻す ---
    db_blob.upload_from_filename(local_db_path)

    return {
        "mode": "fetch_and_register",
        "requested_url": requested_url,
        "file_id": file_id,
        "fetched": len(items),
        "count": len(items),
        "inserted": inserted,
        "skipped": skipped,
    }


# 互換：GETで呼ばれても同じ結果にする（UIの差分吸収）
@router.get("/fetch_and_register")
def opendata_fetch_and_register_get(authorization: str | None = Header(default=None)):
    return opendata_fetch_and_register(authorization)


# ---- 以下は既存のまま ----
@router.get("/datasets/search")
def search_datasets(
    q: str = Query(..., description="検索キーワード（CKAN package_search の q）"),
    rows: int = Query(10, ge=1, le=100),
):
    # 既存実装が続く想定
    return {"todo": "keep existing implementation here"}
