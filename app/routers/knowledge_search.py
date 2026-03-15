from __future__ import annotations

import os
import sqlite3
import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge_search"])

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")


class SearchRequest(BaseModel):
    db_name: str
    query: str
    mode: str = "plain_fts"   # qa / plain_fts / hybrid / hybrid_ai / ai_answer


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


def _sanitize_db_name(db_name: str) -> str:
    name = (db_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="db_name is required")

    if "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="invalid db_name")

    if not name.endswith(".sqlite"):
        raise HTTPException(status_code=400, detail="db_name must end with .sqlite")

    return name


def knowledge_db_path(uid: str, filename: str) -> str:
    return f"users/{uid}/{filename}"


def download_knowledge_db(uid: str, filename: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    gcs_path = knowledge_db_path(uid, filename)
    blob = bucket.blob(gcs_path)

    if not blob.exists():
        raise HTTPException(
            status_code=404,
            detail=f"knowledge db not found: {gcs_path}",
        )

    local_path = f"/tmp/knowledge_search_{uid}_{filename}"
    blob.download_to_filename(local_path)
    return local_path


def _open_knowledge_db(uid: str, db_name: str) -> sqlite3.Connection:
    safe_db_name = _sanitize_db_name(db_name)
    local_db_path = download_knowledge_db(uid, safe_db_name)

    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _normalize_fts_query(query: str) -> str:
    lines = [
        line.strip()
        for line in (query or "").splitlines()
        if line.strip()
    ]

    if not lines:
        raise HTTPException(status_code=400, detail="query is required")

    return " AND ".join(lines)


def _search_plain_fts(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict]:
    fts_query = _normalize_fts_query(query)

    sql = f"""
    SELECT
        ke.entry_id,
        ke.knowledge_type,
        ke.title,
        ke.question,
        ke.answer,
        ke.content,
        ke.source_type,
        ke.source_item_id,
        ke.source_label,
        bm25(knowledge_fts) AS score
    FROM knowledge_fts
    JOIN knowledge_entries ke
      ON ke.rowid = knowledge_fts.rowid
    WHERE knowledge_fts MATCH ?
      AND ke.knowledge_type = 'plain'
    ORDER BY bm25(knowledge_fts), ke.sort_no ASC, ke.created_at DESC
    LIMIT {int(limit)}
    """

    cur = conn.cursor()
    cur.execute(sql, (fts_query,))
    rows = cur.fetchall()

    items = []
    for row in rows:
        item = dict(row)
        content = (item.get("content") or "").strip()
        item["content_preview"] = content[:300]
        items.append(item)

    return items


@router.post("/search")
def search_knowledge(
    req: SearchRequest,
    authorization: str | None = Header(default=None)
):
    uid = get_uid_from_auth_header(authorization)
    conn = _open_knowledge_db(uid, req.db_name)

    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    try:
        mode = (req.mode or "").strip()

        if mode == "plain_fts":
            items = _search_plain_fts(conn, query, limit=20)

        elif mode == "qa":
            items = []

        elif mode == "hybrid":
            items = _search_plain_fts(conn, query, limit=20)

        elif mode == "hybrid_ai":
            items = _search_plain_fts(conn, query, limit=10)

        elif mode == "ai_answer":
            items = []

        else:
            raise HTTPException(status_code=400, detail="invalid mode")

        return {
            "ok": True,
            "mode": mode,
            "db_name": req.db_name,
            "query": req.query,
            "count": len(items),
            "items": items,
        }

    except sqlite3.Error as e:
        logger.exception("search_knowledge sqlite error")
        raise HTTPException(status_code=500, detail=f"sqlite error: {e}")

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("search_knowledge failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()
