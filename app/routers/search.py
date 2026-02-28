from fastapi import APIRouter, Query

router = APIRouter()

@router.get("/search/topk")
def search_topk(
    q: str = Query(...),
    mode: str = Query("qa", pattern="^(qa|knowledge)$"),
    k: int = Query(5, ge=1, le=20),
):
    # デモ用：本来はSQLite＋embeddingでTopKを返す
    return {
        "query": q,
        "mode": mode,
        "k": k,
        "results": []
    }

@router.get("/search/generate")
def search_generate(
    q: str = Query(...),
    k: int = Query(5, ge=1, le=20),
):
    # デモ用：本来はTopK根拠を提示して生成（根拠ID付き）
    return {
        "query": q,
        "k": k,
        "answer": "",
        "evidence": []
    }
