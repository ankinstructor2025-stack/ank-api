from __future__ import annotations

import os
import json
import math
import sqlite3
import logging

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from google.cloud import storage
from openai import OpenAI

import firebase_admin
from firebase_admin import auth as fb_auth
from janome.tokenizer import Tokenizer


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge_search"])

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

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


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def _embed_text(text: str) -> list[float]:
    src = (text or "").strip()
    if not src:
        raise HTTPException(status_code=400, detail="query is required")

    client = _get_openai_client()
    res = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=src
    )
    return list(res.data[0].embedding)


def _parse_embedding(raw_value) -> list[float] | None:
    if raw_value is None:
        return None

    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8", errors="ignore")

    if isinstance(raw_value, str):
        src = raw_value.strip()
        if not src:
            return None

        if src.startswith("[") and src.endswith("]"):
            try:
                arr = json.loads(src)
                return [float(x) for x in arr]
            except Exception:
                return None

        if "," in src:
            try:
                return [float(x.strip()) for x in src.split(",") if x.strip()]
            except Exception:
                return None

        return None

    if isinstance(raw_value, (list, tuple)):
        try:
            return [float(x) for x in raw_value]
        except Exception:
            return None

    return None


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2:
        return -1.0

    if len(vec1) != len(vec2):
        return -1.0

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0

    for a, b in zip(vec1, vec2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b

    if norm1 == 0.0 or norm2 == 0.0:
        return -1.0

    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


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

        if len(base_form) == 1 and not base_form.isdigit():
            if pos not in {"名詞"}:
                continue

        tokens.append(base_form)

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
    入力文を分かち書きして、FTS MATCH 用のスペース連結文字列に変換する。
    SQLite FTS5 ではスペース区切りで候補を広く拾い、bm25() で関連度順に並べる。
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

        if not terms:
            terms = [line]

        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            all_terms.append(term)

    if not all_terms:
        raise HTTPException(status_code=400, detail="query is required")

    return " ".join(all_terms)


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


def _search_qa_similarity(conn: sqlite3.Connection, query: str, limit: int = 5) -> list[dict]:
    query_embedding = _embed_text(query)

    sql = """
    SELECT
        entry_id,
        knowledge_type,
        title,
        question,
        answer,
        content,
        source_type,
        source_item_id,
        source_label,
        embedding
    FROM knowledge_entries
    WHERE knowledge_type = 'qa'
      AND question IS NOT NULL
      AND TRIM(question) <> ''
      AND embedding IS NOT NULL
      AND TRIM(embedding) <> ''
    """

    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()

    scored_items: list[dict] = []

    for row in rows:
        item = dict(row)

        qa_embedding = _parse_embedding(item.get("embedding"))
        if not qa_embedding:
            continue

        similarity = _cosine_similarity(query_embedding, qa_embedding)
        if similarity < -0.5:
            continue

        answer = (item.get("answer") or "").strip()
        content = (item.get("content") or "").strip()

        scored_items.append({
            "entry_id": item.get("entry_id"),
            "knowledge_type": item.get("knowledge_type"),
            "title": item.get("title"),
            "question": item.get("question"),
            "answer": answer,
            "content": content,
            "source_type": item.get("source_type"),
            "source_item_id": item.get("source_item_id"),
            "source_label": item.get("source_label"),
            "score": similarity,
            "content_preview": answer[:300] if answer else content[:300],
        })

    scored_items.sort(key=lambda x: x["score"], reverse=True)
    return scored_items[:limit]


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
            items = _search_qa_similarity(conn, query, limit=5)

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
