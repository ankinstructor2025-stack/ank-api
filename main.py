from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import storage
import json

app = FastAPI()

BUCKET_NAME = "anktest"

class SessionRequest(BaseModel):
    user_id: str

# CORS: ブラウザ(GitHub Pages) → Cloud Run のため
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SessionRequest(BaseModel):
    user_id: str

@app.get("/")
def root():
    return {"message": "anktest-api is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/session")
def create_session(req: SessionRequest):

    user_id = req.user_id

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blob_path = f"{user_id}/qa.json"
    blob = bucket.blob(blob_path)

    # 既に存在するか
    if not blob.exists():

        init_data = {
            "user_id": user_id,
            "records": []
        }

        blob.upload_from_string(
            json.dumps(init_data, ensure_ascii=False, indent=2),
            content_type="application/json"
        )

    return {
        "user_id": user_id,
        "status": "session ok"
    }

@app.post("/v1/qa_build")
async def qa_build(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    今回は「アップロードまで」：
    受け取った対話ファイルを GCS に保存して、保存先を返す。
    （次のステップで OpenAI→QA保存→qa.json更新 をここに追加する）
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # 保存先（設計図どおり uploads/ 配下）
    object_path = f"{user_id}/upload_files/{file.filename}"
    blob = bucket.blob(object_path)

    data = await file.read()
    blob.upload_from_string(
        data,
        content_type=file.content_type or "application/octet-stream"
    )

    # ブラウザ側が履歴表示などに使えるよう、相対パスも返す
    return {
        "status": "uploaded",
        "user_id": user_id,
        "upload_file": f"upload_files/{file.filename}",
        "gcs_object": object_path
    }
