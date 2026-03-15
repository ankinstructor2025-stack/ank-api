from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import sqlite3

router = APIRouter()


class SearchRequest(BaseModel):
    db_name: str
    query: str
    mode: str = "plain_fts"   # qa / plain_fts / hybrid / hybrid_ai / ai_answer


def _get_user_id_from_auth(authorization: Optional[str]) -> str:
    """
    既存の認証処理に合わせて差し替えること。
    いまは最低限、Authorization ヘッダがあることだけ確認する。
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="authorization header is required")

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="invalid authorization header")

    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="empty bearer token")

    # ここは既存の Firebase / session 検証に置き換える
    # いまはダミーで固定値を返す
    return "dummy_user"


def _sanitize_db_name(db_name: str) -> str:
    name = (db_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="db_name is required")

    if "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="invalid db_name")

    if not name.endswith(".sqlite"):
        raise HTTPException(status_code=400, detail="db_name must end with .sqlite")

    return name


def _open_knowledge_db(user_id: str, db_name: str) -> sqlite3.Connection:
    """
    既存の GCS ダウンロード処理に合わせて差し替えること。
    いまは /tmp に配置済みの knowledge DB を開く。
    """
    _ = user_id  # 将来のユーザー別パス制御用

    safe_db_name = _sanitize_db_name(db_name)
    db_path = f"/tmp/{safe_db_name}"

    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail=f"knowledge db not found: {safe_db_name}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _normalize_fts_query(query: str) -> str:
    """
    複数行入力を FTS 用の AND 検索に寄せる。
    必要に応じて OR に変えてもよい。
    """
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


@router.post("/knowledge/search")
def search_knowledge(
    req: SearchRequest,
    authorization: str | None = Header(default=None)
):
    user_id = _get_user_id_from_auth(authorization)
    conn = _open_knowledge_db(user_id, req.db_name)

    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    try:
        mode = (req.mode or "").strip()

        if mode == "plain_fts":
            items = _search_plain_fts(conn, query, limit=20)

        elif mode == "qa":
            # ここは今後、入力文を OpenAI でベクトル化して
            # knowledge_entries.embedding と類似度比較する想定
            items = []

        elif mode == "hybrid":
            # まずは plain_fts を流用
            items = _search_plain_fts(conn, query, limit=20)

        elif mode == "hybrid_ai":
            # まずは plain_fts を流用
            items = _search_plain_fts(conn, query, limit=10)

        elif mode == "ai_answer":
            # AI回答は別処理を追加予定
            items = []

        else:
            raise HTTPException(status_code=400, detail="invalid mode")

        return {
            "ok": True,
            "mode": mode,
            "db_name": req.db_name,
            "query": req.query,
            "count": len(items),
            "items": items
        }

    finally:
        conn.close()
