from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from google.cloud import storage
import sqlite3
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import ulid

router = APIRouter()
JST = ZoneInfo("Asia/Tokyo")
BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")

# ここは既存の認証ロジックに合わせて差し替え
def get_current_user_id() -> str:
    raise NotImplementedError("replace with your auth dependency")


@router.post("/upload_and_register")
async def upload_and_register(
    file: UploadFile = File(...),
    uid: str = Depends(get_current_user_id),
):
    """
    仕様:
    - GCS: users/{uid}/uploads/{file_id}_{original_filename} にアップロード
    - SQLite(ank.db): uploaded_files に1行INSERT
    - 同名(logical_name)は弾く（409）
    - ank.db は GCS: users/{uid}/ank.db を /tmp に落として更新→上書き戻し
    """

    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is empty")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    original_filename = file.filename
    logical_name = original_filename  # 同名判定の基準（画面表示名）
    ext = original_filename.rsplit(".", 1)[-1] if "." in original_filename else ""

    # ① ULID生成
    file_id = str(ulid.new())

    # ② ank.db を /tmp に用意（存在しない場合は新規作成）
    db_blob_path = f"users/{uid}/ank.db"
    db_blob = bucket.blob(db_blob_path)

    local_db_path = f"/tmp/ank_{uid}_{file_id}.db"

    try:
        if db_blob.exists():
            db_blob.download_to_filename(local_db_path)
        else:
            # 新規DB作成（空ファイルを作る）
            conn = sqlite3.connect(local_db_path)
            conn.close()

        # ③ DBオープン＆テーブル準備（logical_name UNIQUE）
        conn = sqlite3.connect(local_db_path)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_files (
                file_id TEXT PRIMARY KEY,
                logical_name TEXT NOT NULL UNIQUE,
                original_filename TEXT NOT NULL,
                ext TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # ④ 同名チェック（アップロード前に弾く）
        cur.execute(
            "SELECT 1 FROM uploaded_files WHERE logical_name = ? LIMIT 1",
            (logical_name,),
        )
        if cur.fetchone():
            conn.close()
            raise HTTPException(status_code=409, detail="同名ファイルはアップロードできません")

        # ⑤ GCS uploads へアップロード
        upload_blob_path = f"users/{uid}/uploads/{file_id}_{original_filename}"
        upload_blob = bucket.blob(upload_blob_path)

        # UploadFileはストリームとして扱える（メモリに全読みしない）
        file.file.seek(0)
        upload_blob.upload_from_file(file.file)

        # ⑥ DBへINSERT
        created_at = datetime.now(tz=JST).isoformat()

        try:
            cur.execute(
                """
                INSERT INTO uploaded_files
                  (file_id, logical_name, original_filename, ext, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (file_id, logical_name, original_filename, ext, created_at),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            # UNIQUE違反（同時実行など）
            conn.close()
            # ここまででファイルが上がってるので、孤児を消す
            try:
                upload_blob.delete()
            except Exception:
                pass
            raise HTTPException(status_code=409, detail="同名ファイルはアップロードできません")

        conn.close()

        # ⑦ DBをGCSへ上書きアップロード
        db_blob.upload_from_filename(local_db_path)

        return {
            "file_id": file_id,
            "logical_name": logical_name,
            "original_filename": original_filename,
            "ext": ext,
            "created_at": created_at,
            "gcs_path": upload_blob_path,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upload_and_register failed: {e}")
    finally:
        # /tmp の掃除（失敗しても無視）
        try:
            if os.path.exists(local_db_path):
                os.remove(local_db_path)
        except Exception:
            pass
