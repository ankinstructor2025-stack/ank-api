from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

router = APIRouter()

class IngestApiRequest(BaseModel):
    preset_id: str | None = None
    params: dict | None = None

class IngestUrlRequest(BaseModel):
    url: str

@router.post("/ingest/api")
def ingest_api(req: IngestApiRequest):
    # デモ用：ここで公開APIから取得→正規化→SQLite保存へ繋ぐ
    return {"status": "queued", "source_type": "api", "preset_id": req.preset_id, "params": req.params or {}}

@router.post("/ingest/url")
def ingest_url(req: IngestUrlRequest):
    # デモ用：ここでURLから取得→本文抽出→SQLite保存へ繋ぐ
    return {"status": "queued", "source_type": "url", "url": req.url}

@router.post("/ingest/upload")
async def ingest_upload(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    # デモ用：受け取ったファイル情報だけ返す（実装は次段）
    return {
        "status": "received",
        "source_type": "upload",
        "user_id": user_id,
        "filename": file.filename,
        "content_type": file.content_type,
    }
