from __future__ import annotations

import os
import sqlite3
import logging

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from google.cloud import storage

import firebase_admin
from firebase_admin import auth as fb_auth
from janome.tokenizer import Tokenizer


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge_search"])

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")

# janome は毎回生成せず、グローバルで 1 回だけ作る
_TOKENIZER = Tokenizer()


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


def _tokenize_for_fts(text: str) -> list[str]:
    """
    検索文字列を janome で分かち書きし、FTS 用のトークン配列にする。
    ノイズになりやすい助詞・助動詞・記号は除外する。
    """
    src = (text or "").strip()
    if not src:
        return []

    tokens: list[str] = []

    for token in _TOKENIZER.tokenize(src):
        surface = (token.surface or "").strip()
        if not surface:
            continue

        pos = token.part_of_speech.split(",")[0]
        if pos in {"助詞", "助動詞", "記号"}:
            continue

        base_form = getattr(token, "base_form", None) or surface
        base_form = base_form.strip()
        if not base_form or base_form == "*":
            base_form = surface

        # 1文字ノイズを少し抑制。ただし数字は残す
        if len(base_form) == 1 and not base_form.isdigit():
            if pos not in {"名詞"}:
                continue

        tokens.append(base_form)

    # 重複を除去しつつ順序維持
    unique_tokens: list[str] = []
    seen: set[str] = set()

    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        unique_tokens.append(t)

    return unique_tokens


def _normalize_fts_query(query: str) -> str:
    """
    入力文を分かち書きして、FTS MATCH 用の AND 連結文字列に変換する。
    例:
      戦略分野における民間投資の予見可能性を向上させることを目指す
      -> 戦略分野 AND 民間投資 AND 予見可能性 AND 向上 AND 目指す
    """
    lines = [
        line.strip()
        for line in (query or "").splitlines()
        if line.strip()
    ]

    if not lines:
        raise HTTPException(status_code=400, detail="query is required")

    all_terms: list[str] = []
    seen: set[str] = set()

    for line in lines:
        terms = _tokenize_for_fts(line)

        # janome で何も取れなかった時だけ生文を fallback
        if not terms:
            terms = [line]

        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            all_terms.append(term)

    if not all_terms:
        raise HTTPException(status_code=400, detail="query is required")

    return " AND ".join(all_terms)


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
        item["fts_query"] = fts_query
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


@router.get("/dbs")
def list_knowledge_dbs(
    authorization: str | None = Header(default=None)
):
    uid = get_uid_from_auth_header(authorization)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    prefix = f"users/{uid}/"

    blobs = client.list_blobs(bucket, prefix=prefix)

    items = []

    for blob in blobs:
        name = blob.name.replace(prefix, "")

        if not name.endswith(".sqlite"):
            continue

        if not name.startswith("knowledge_"):
            continue

        items.append({
            "db_name": name,
            "size": blob.size,
            "updated": blob.updated.isoformat() if blob.updated else None
        })

    items.sort(key=lambda x: x["db_name"], reverse=True)

    return {
        "ok": True,
        "count": len(items),
        "items": items
    }
