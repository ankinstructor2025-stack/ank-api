from openai import OpenAI
from fastapi import FastAPI, Form, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import storage
import mimetypes
import json
import os
import io
import ulid  # pip: ulid-py
from urllib.parse import quote

app = FastAPI()

# できれば環境変数優先（無ければ教材の固定値）
BUCKET_NAME = os.environ.get("UPLOAD_BUCKET", "anktest")

def _safe_ascii_filename(name: str) -> str:
    # 日本語などは落ちるので、ヘッダ用はASCIIに退避
    return "download"

def _build_qa_prompt(dialogue_text: str) -> str:
    return (
        "次の対話データから、ユーザーの質問と回答のペア(QA)を抽出してください。\n"
        "出力は必ず JSON のみ。コードブロックは禁止。\n"
        "形式:\n"
        "[\n"
        "  {\"q\": \"質問\", \"a\": \"回答\"}\n"
        "]\n\n"
        "対話データ:\n"
        f"{dialogue_text}\n"
    )


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


def _ensure_user_qa_json(bucket, user_id: str) -> dict:
    """
    {user_id}/qa.json が無ければ作成し、内容(dict)を返す
    """
    blob_path = f"{user_id}/qa.json"
    blob = bucket.blob(blob_path)

    if not blob.exists():
        init_data = {"user_id": user_id, "records": []}
        blob.upload_from_string(
            json.dumps(init_data, ensure_ascii=False, indent=2),
            content_type="application/json",
        )
        return init_data

    text = blob.download_as_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except Exception:
        # 壊れていたら最低限復旧（教材用途で止めない）
        data = {"user_id": user_id, "records": []}
    if "user_id" not in data:
        data["user_id"] = user_id
    if "records" not in data or not isinstance(data["records"], list):
        data["records"] = []
    return data


def _save_user_qa_json(bucket, user_id: str, qa_json: dict) -> None:
    blob_path = f"{user_id}/qa.json"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(
        json.dumps(qa_json, ensure_ascii=False, indent=2),
        content_type="application/json",
    )


@app.post("/v1/session")
def create_session(req: SessionRequest):
    user_id = req.user_id

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    _ = _ensure_user_qa_json(bucket, user_id)

    return {"user_id": user_id, "status": "session ok"}


@app.post("/v1/qa_build")
async def qa_build(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    1) 対話ファイルを GCS に保存
    2) OpenAIでQA生成
    3) 成功した場合のみ
       - QAファイルをGCSに保存
       - qa.json(records) を追記して保存
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # ---- 1) 既存：GCSアップロード（維持） ----
    upload_rel_path = f"upload_files/{file.filename}"
    upload_object_path = f"{user_id}/{upload_rel_path}"
    upload_blob = bucket.blob(upload_object_path)

    data = await file.read()
    upload_blob.upload_from_string(
        data,
        content_type=file.content_type or "application/octet-stream",
    )

    # OPENAI_API_KEY が無ければ、アップロード成功だけ返す（教材の途中でも止めない）
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "status": "uploaded_but_qa_skipped",
            "reason": "OPENAI_API_KEY is not set",
            "user_id": user_id,
            "upload_file": upload_rel_path,
            "gcs_object": upload_object_path,
        }

    # ---- 2) QA生成 ----
    try:
        try:
            dialogue_text = data.decode("utf-8")
        except UnicodeDecodeError:
            dialogue_text = data.decode("utf-8", errors="replace")

        oa = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        prompt = _build_qa_prompt(dialogue_text)

        res = oa.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        content = res.choices[0].message.content or ""

        qa_list = json.loads(content)
        if not isinstance(qa_list, list):
            raise ValueError("QA JSON is not a list")

        # ---- 3) 成功時のみ：QAファイル保存 + qa.json更新 ----
        qa_id = str(ulid.new())

        qa_rel_path = f"qa_files/{qa_id}.json"
        qa_object_path = f"{user_id}/{qa_rel_path}"
        qa_blob = bucket.blob(qa_object_path)

        qa_blob.upload_from_string(
            json.dumps(qa_list, ensure_ascii=False, indent=2),
            content_type="application/json",
        )

        qa_json = _ensure_user_qa_json(bucket, user_id)

        qa_json["records"].append(
            {
                "qa_id": qa_id,
                "upload_file": upload_rel_path,
                "qa_file": qa_rel_path,
            }
        )

        _save_user_qa_json(bucket, user_id, qa_json)

        return {
            "status": "uploaded_and_qa_saved",
            "user_id": user_id,
            "qa_id": qa_id,
            "upload_file": upload_rel_path,
            "qa_file": qa_rel_path,
            "gcs_upload_object": upload_object_path,
            "gcs_qa_object": qa_object_path,
            "qa_count": len(qa_list),
            "qa": qa_list,  # 画面で確認できるように返す（必要なら後で削除可）
        }

    except Exception as e:
        # 失敗しても「アップロード済み」は維持して返す（qa.jsonは更新しない）
        return {
            "status": "uploaded_but_qa_failed",
            "user_id": user_id,
            "upload_file": upload_rel_path,
            "gcs_object": upload_object_path,
            "error": str(e),
        }

@app.get("/v1/files")
def list_files(user_id: str = Query(...)):
    """
    図の GET /v1/files
    - user_id の upload_files 一覧を返す
    - qa.json(records) から upload→qa の対応を返す
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    qa_json = _ensure_user_qa_json(bucket, user_id)
    records = qa_json.get("records", [])

    # upload_file -> qa_file の対応（同一uploadが複数回なら最新を優先）
    upload_to_qa = {}
    for r in records:
        uf = r.get("upload_file")
        qf = r.get("qa_file")
        if uf:
            upload_to_qa[uf] = qf  # 後勝ち

    # upload_files をGCSから列挙
    prefix = f"{user_id}/upload_files/"
    uploads = []
    for blob in client.list_blobs(bucket, prefix=prefix):
        # {user_id}/ を外して相対パスにする
        rel = blob.name[len(user_id) + 1:]  # "upload_files/xxx"
        uploads.append(rel)

    uploads = sorted(set(uploads))

    out = []
    for uf in uploads:
        qf = upload_to_qa.get(uf)  # "qa_files/xxxx.json" or None
        out.append({
            "upload_file": uf,
            "qa_file": qf,
            # DLは署名URLにせずAPI経由（確実に動く）
            "upload_url": f"/v1/file?user_id={user_id}&path={uf}",
            "qa_url": f"/v1/file?user_id={user_id}&path={qf}" if qf else None,
        })

    return {"user_id": user_id, "records": out}

@app.get("/v1/file")
def download_file(user_id: str = Query(...), path: str = Query(...)):
    """
    /v1/file?user_id=...&path=upload_files/xxx or qa_files/xxx.json
    - GCSのオブジェクトをAPI経由でダウンロードさせる
    """
    if not path:
        raise HTTPException(status_code=400, detail="path is required")

    object_path = f"{user_id}/{path}"

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(object_path)

    if not blob.exists():
        raise HTTPException(status_code=404, detail="file not found")

    data = blob.download_as_bytes()
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"

    filename = path.split("/")[-1]

    ascii_name = _safe_ascii_filename(filename)
    utf8_name = quote(filename, safe="")

    return StreamingResponse(
        io.BytesIO(data),
        media_type=mime,
        headers={
            "Content-Disposition": f"attachment; filename={ascii_name}; filename*=UTF-8''{utf8_name}"
        },
    )

@app.get("/v1/openai_echo")
def openai_echo():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "echo test"}],
    )

    return {"ok": True, "reply": res.choices[0].message.content}
