from __future__ import annotations

import os
import json
import math
import sqlite3
import logging

from fastapi import APIRouter, Header, HTTPException, Query
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
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
HYBRID_AI_PROMPT_FILE = os.getenv("HYBRID_AI_PROMPT_FILE", "knowledge_hybrid_ai_prompt.txt")
SEARCH_CONFIG_FILE = os.getenv("SEARCH_CONFIG_FILE", "search_config.json")

DEFAULT_SEARCH_CONFIG = {
    "qa_similarity": {
        "strong_threshold": 0.8,
        "weak_threshold": 0.4,
        "top_k": 5,
    },
    "fts": {
        "strong_bm25_max": -4.5,
        "weak_bm25_max": -2.0,
        "top_k": 10,
    },
    "hybrid": {
        "qa_top_k": 5,
        "fts_top_k": 5,
        "top_k": 10,
    },
}

_TOKENIZER = Tokenizer()


class SearchRequest(BaseModel):
    db_name: str
    query: str
    mode: str = "plain_fts"


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


def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


def download_knowledge_db(uid: str, filename: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    gcs_path = knowledge_db_path(uid, filename)
    blob = bucket.blob(gcs_path)

    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"knowledge db not found: {gcs_path}")

    local_path = f"/tmp/knowledge_search_{uid}_{filename}"
    blob.download_to_filename(local_path)
    return local_path


def download_user_db(uid: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    gcs_path = user_db_path(uid)
    blob = bucket.blob(gcs_path)

    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"user db not found: {gcs_path}")

    local_path = f"/tmp/knowledge_master_{uid}.sqlite"
    blob.download_to_filename(local_path)
    return local_path


def _open_knowledge_db(uid: str, db_name: str) -> sqlite3.Connection:
    safe_db_name = _sanitize_db_name(db_name)
    local_db_path = download_knowledge_db(uid, safe_db_name)
    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _open_user_db(uid: str) -> sqlite3.Connection:
    local_db_path = download_user_db(uid)
    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def _load_json_from_gcs(filename: str) -> dict:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    for path in [f"template/{filename}", filename]:
        blob = bucket.blob(path)
        if not blob.exists():
            continue
        try:
            text = blob.download_as_text(encoding="utf-8").strip()
            if not text:
                raise HTTPException(status_code=500, detail=f"config is empty: {path}")
            data = json.loads(text)
            if not isinstance(data, dict):
                raise HTTPException(status_code=500, detail=f"config is not object: {path}")
            return data
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to load config from GCS: {path}: {e}")

    raise HTTPException(status_code=500, detail=f"config not found in GCS: {filename}")


def _deep_merge_dict(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _load_search_config() -> dict:
    loaded = _load_json_from_gcs(SEARCH_CONFIG_FILE)
    config = _deep_merge_dict(DEFAULT_SEARCH_CONFIG, loaded)

    qa_cfg = config.get("qa_similarity", {})
    fts_cfg = config.get("fts", {})
    hybrid_cfg = config.get("hybrid", {})

    config["qa_similarity"] = {
        "strong_threshold": _to_float(qa_cfg.get("strong_threshold"), 0.8),
        "weak_threshold": _to_float(qa_cfg.get("weak_threshold"), 0.4),
        "top_k": _to_int(qa_cfg.get("top_k"), 5),
    }
    if config["qa_similarity"]["weak_threshold"] > config["qa_similarity"]["strong_threshold"]:
        config["qa_similarity"]["weak_threshold"] = config["qa_similarity"]["strong_threshold"]

    config["fts"] = {
        "strong_bm25_max": _to_float(fts_cfg.get("strong_bm25_max"), -4.5),
        "weak_bm25_max": _to_float(fts_cfg.get("weak_bm25_max"), -2.0),
        "top_k": _to_int(fts_cfg.get("top_k"), 10),
    }
    if config["fts"]["strong_bm25_max"] > config["fts"]["weak_bm25_max"]:
        strong = config["fts"]["strong_bm25_max"]
        weak = config["fts"]["weak_bm25_max"]
        config["fts"]["strong_bm25_max"] = weak
        config["fts"]["weak_bm25_max"] = strong

    config["hybrid"] = {
        "qa_top_k": _to_int(hybrid_cfg.get("qa_top_k"), 5),
        "fts_top_k": _to_int(hybrid_cfg.get("fts_top_k"), 5),
        "top_k": _to_int(hybrid_cfg.get("top_k"), 10),
    }
    return config


def _embed_text(text: str) -> list[float]:
    src = (text or "").strip()
    if not src:
        raise HTTPException(status_code=400, detail="query is required")
    client = _get_openai_client()
    res = client.embeddings.create(model=EMBEDDING_MODEL, input=src)
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
    lines = [line.strip() for line in (query or "").splitlines() if line.strip()]
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
    if len(all_terms) == 1:
        return all_terms[0]
    return " OR ".join(all_terms)


def _classify_qa_items(items: list[dict], strong_threshold: float, weak_threshold: float, limit: int) -> list[dict]:
    strong_items = [item for item in items if float(item.get("score", -1.0)) >= strong_threshold]
    if strong_items:
        return strong_items[:limit]

    weak_items = [
        item for item in items
        if weak_threshold <= float(item.get("score", -1.0)) < strong_threshold
    ]
    if weak_items:
        return weak_items[:limit]

    return []


def _classify_fts_items(items: list[dict], strong_bm25_max: float, weak_bm25_max: float, limit: int) -> list[dict]:
    strong_items = [item for item in items if float(item.get("score", 999999.0)) <= strong_bm25_max]
    if strong_items:
        return strong_items[:limit]

    weak_items = [
        item for item in items
        if strong_bm25_max < float(item.get("score", 999999.0)) <= weak_bm25_max
    ]
    if weak_items:
        return weak_items[:limit]

    return []


def _search_plain_fts_raw(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict]:
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
        item["result_kind"] = "plain"
        items.append(item)
    return items


def _search_plain_fts(conn: sqlite3.Connection, query: str, config: dict) -> list[dict]:
    raw_items = _search_plain_fts_raw(conn, query, limit=config["fts"]["top_k"])
    return _classify_fts_items(
        raw_items,
        strong_bm25_max=config["fts"]["strong_bm25_max"],
        weak_bm25_max=config["fts"]["weak_bm25_max"],
        limit=config["fts"]["top_k"],
    )


def _search_qa_similarity_raw(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[dict]:
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
        if similarity < 0.0:
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
            "result_kind": "qa",
        })

    scored_items.sort(key=lambda x: x["score"], reverse=True)
    return scored_items[:limit]


def _search_qa_similarity(conn: sqlite3.Connection, query: str, config: dict) -> list[dict]:
    raw_items = _search_qa_similarity_raw(conn, query, limit=max(config["qa_similarity"]["top_k"], 10))
    return _classify_qa_items(
        raw_items,
        strong_threshold=config["qa_similarity"]["strong_threshold"],
        weak_threshold=config["qa_similarity"]["weak_threshold"],
        limit=config["qa_similarity"]["top_k"],
    )


def _search_hybrid(conn: sqlite3.Connection, query: str, config: dict) -> dict:
    raw_qa_items = _search_qa_similarity_raw(
        conn,
        query,
        limit=max(config["hybrid"]["qa_top_k"], config["qa_similarity"]["top_k"], 10),
    )
    raw_plain_items = _search_plain_fts_raw(
        conn,
        query,
        limit=max(config["hybrid"]["fts_top_k"], config["fts"]["top_k"]),
    )

    qa_items = _classify_qa_items(
        raw_qa_items,
        strong_threshold=config["qa_similarity"]["strong_threshold"],
        weak_threshold=config["qa_similarity"]["weak_threshold"],
        limit=config["hybrid"]["qa_top_k"],
    )
    plain_items = _classify_fts_items(
        raw_plain_items,
        strong_bm25_max=config["fts"]["strong_bm25_max"],
        weak_bm25_max=config["fts"]["weak_bm25_max"],
        limit=config["hybrid"]["fts_top_k"],
    )

    combined_items: list[dict] = []
    combined_items.extend(qa_items)
    combined_items.extend(plain_items)

    top_k = int(config["hybrid"]["top_k"])
    if top_k > 0:
        combined_items = combined_items[:top_k]

    return {
        "qa_items": qa_items,
        "plain_items": plain_items,
        "items": combined_items,
        "count": len(combined_items),
    }


def _load_text_prompt_from_gcs(filename: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    for path in [f"template/{filename}", filename]:
        blob = bucket.blob(path)
        if blob.exists():
            text = blob.download_as_text(encoding="utf-8").strip()
            if not text:
                raise HTTPException(status_code=500, detail=f"prompt is empty: {path}")
            return text

    raise HTTPException(status_code=500, detail=f"prompt not found in GCS: {filename}")


def _build_hybrid_context(items: list[dict], max_chars_per_item: int = 1200) -> str:
    blocks: list[str] = []
    for idx, item in enumerate(items, start=1):
        result_kind = (item.get("result_kind") or "").strip()
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()
        content = (item.get("content_preview") or item.get("content") or "").strip()
        source_type = (item.get("source_type") or "").strip()
        source_label = (item.get("source_label") or "").strip()
        score = item.get("score")

        body = answer or content
        if max_chars_per_item and len(body) > max_chars_per_item:
            body = body[:max_chars_per_item]

        lines = [f"[候補{idx}]"]
        if result_kind:
            lines.append(f"種別: {result_kind}")
        if question:
            lines.append(f"質問: {question}")
        if body:
            lines.append(f"本文: {body}")
        if source_type or source_label:
            lines.append(f"情報源: {source_type} / {source_label}")
        if score is not None:
            try:
                if result_kind == "qa":
                    lines.append(f"類似度: {float(score):.4f}")
                else:
                    lines.append(f"bm25: {float(score):.6f}")
            except Exception:
                lines.append(f"score: {score}")

        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _generate_hybrid_ai_answer(conn: sqlite3.Connection, query: str, config: dict) -> dict:
    prompt_text = _load_text_prompt_from_gcs(HYBRID_AI_PROMPT_FILE)
    hybrid_result = _search_hybrid(conn, query, config)
    items = hybrid_result.get("items", [])[:8]

    if not items:
        no_answer = "検索結果が見つからなかったため、回答を整形できませんでした。"
        return {
            "title": "AI回答",
            "answer": no_answer,
            "content_preview": no_answer,
            "source_type": "openai",
            "source_label": CHAT_MODEL,
            "result_kind": "hybrid_ai_answer",
            "grounded_items": [],
        }

    context = _build_hybrid_context(items, max_chars_per_item=1200)
    user_prompt = f"""ユーザー質問:
{query}

ハイブリッド検索結果:
{context}

上記だけを根拠に、意味を変えず、読みやすい回答文を作成してください。"""

    client = _get_openai_client()
    response = client.responses.create(
        model=CHAT_MODEL,
        input=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer_text = ""
    if hasattr(response, "output_text") and response.output_text:
        answer_text = response.output_text.strip()
    if not answer_text:
        answer_text = "検索結果をもとに回答を整形できませんでした。"

    return {
        "title": "AI回答",
        "answer": answer_text,
        "content_preview": answer_text[:1000],
        "source_type": "openai",
        "source_label": CHAT_MODEL,
        "result_kind": "hybrid_ai_answer",
        "grounded_items": items,
    }


def _generate_ai_answer(query: str) -> dict:
    prompt = (query or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="query is required")

    client = _get_openai_client()
    system_message = (
        "あなたは日本語で回答するアシスタントです。"
        "質問に対して簡潔かつ自然な日本語で回答してください。"
        "不明な点は断定せず、その旨を明確にしてください。"
    )

    response = client.responses.create(
        model=CHAT_MODEL,
        input=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
    )

    answer_text = ""
    if hasattr(response, "output_text") and response.output_text:
        answer_text = response.output_text.strip()
    if not answer_text:
        answer_text = "回答を生成できませんでした。"

    return {
        "title": "AI回答",
        "answer": answer_text,
        "content_preview": answer_text[:1000],
        "source_type": "openai",
        "source_label": CHAT_MODEL,
        "result_kind": "ai_answer",
    }


@router.post("/search")
def search_knowledge(req: SearchRequest, authorization: str | None = Header(default=None)):
    uid = get_uid_from_auth_header(authorization)
    conn = _open_knowledge_db(uid, req.db_name)

    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    config = _load_search_config()

    try:
        mode = (req.mode or "").strip()

        if mode == "plain_fts":
            items = _search_plain_fts(conn, query, config)
            return {
                "ok": True,
                "mode": mode,
                "db_name": req.db_name,
                "query": req.query,
                "count": len(items),
                "items": items,
                "search_config": config,
            }

        if mode == "qa":
            items = _search_qa_similarity(conn, query, config)
            return {
                "ok": True,
                "mode": mode,
                "db_name": req.db_name,
                "query": req.query,
                "count": len(items),
                "items": items,
                "search_config": config,
            }

        if mode == "hybrid":
            hybrid_result = _search_hybrid(conn, query, config)
            return {
                "ok": True,
                "mode": mode,
                "db_name": req.db_name,
                "query": req.query,
                "count": hybrid_result["count"],
                "items": hybrid_result["items"],
                "qa_items": hybrid_result["qa_items"],
                "plain_items": hybrid_result["plain_items"],
                "search_config": config,
            }

        if mode == "hybrid_ai":
            item = _generate_hybrid_ai_answer(conn, query, config)
            return {
                "ok": True,
                "mode": mode,
                "db_name": req.db_name,
                "query": req.query,
                "count": 1,
                "items": [item],
                "grounded_count": len(item.get("grounded_items", [])),
                "grounded_items": item.get("grounded_items", []),
                "search_config": config,
            }

        if mode == "ai_answer":
            item = _generate_ai_answer(query)
            return {
                "ok": True,
                "mode": mode,
                "db_name": req.db_name,
                "query": req.query,
                "count": 1,
                "items": [item],
                "search_config": config,
            }

        raise HTTPException(status_code=400, detail="invalid mode")

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
    source_type: str | None = Query(default=None),
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)
    conn = _open_user_db(uid)

    try:
        params: list[str] = []
        sql = """
        SELECT
            database_name,
            source_type,
            created_at
        FROM knowledge_db
        """

        source_type_value = (source_type or "").strip()
        if source_type_value:
            sql += " WHERE source_type = ?"
            params.append(source_type_value)

        sql += " ORDER BY created_at DESC, database_name DESC"

        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()

        items = []
        for row in rows:
            database_name = (row["database_name"] or "").strip()
            if not database_name:
                continue

            items.append({
                "database_name": database_name,
                "source_type": row["source_type"],
                "created_at": row["created_at"],
            })

        return {
            "ok": True,
            "count": len(items),
            "items": items,
        }

    except sqlite3.Error as e:
        logger.exception("list_knowledge_dbs sqlite error")
        raise HTTPException(status_code=500, detail=f"sqlite error: {e}")

    finally:
        conn.close()
