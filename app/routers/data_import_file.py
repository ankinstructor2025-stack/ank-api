import os
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from firebase_admin import firestore
from google.cloud import storage

from app.core.firebase import (
    get_firestore_client,
    verify_id_token,
)


router = APIRouter(
    prefix="/data-import",
    tags=["data-import"],
)

BUCKET_NAME = os.getenv(
    "UPLOAD_BUCKET",
    "qaplain",
)

DATA_SOURCE_COLLECTION = "data_sources"
UPLOADED_FILE_COLLECTION = "uploaded_files"


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_email(value: Any) -> str:
    return normalize_text(value).lower()


def normalize_extension(value: Any) -> str:
    return normalize_text(value).lower().lstrip(".")


def normalize_source_type(value: Any) -> str:
    source_type = (
        normalize_text(value)
        .lower()
        .replace("-", "_")
    )

    if source_type == "upload":
        return "file"

    return source_type


def get_file_extension(file_name: str) -> str:
    return normalize_extension(
        Path(file_name).suffix
    )


def sanitize_file_name(file_name: str) -> str:
    normalized = normalize_text(file_name)
    normalized = normalized.replace("\\", "/")
    normalized = normalized.split("/")[-1]
    normalized = re.sub(
        r"[\x00-\x1f\x7f]",
        "",
        normalized,
    )
    normalized = normalized.strip(" .")

    if not normalized:
        raise HTTPException(
            status_code=400,
            detail="ファイル名を確認できませんでした。",
        )

    return normalized


def authenticate_user(
    authorization: str,
) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header",
        )

    id_token = authorization.replace(
        "Bearer ",
        "",
        1,
    ).strip()

    try:
        decoded_token = verify_id_token(
            id_token
        )

    except Exception as error:
        print(
            "verify_id_token error: "
            f"{type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail="認証情報を確認できませんでした。",
        )

    email = normalize_email(
        decoded_token.get("email", "")
    )

    if not email:
        raise HTTPException(
            status_code=401,
            detail="メールアドレスを取得できませんでした。",
        )

    return {
        **decoded_token,
        "email": email,
    }


def normalize_extensions(
    values: Any,
) -> list[str]:
    if not isinstance(values, list):
        return []

    result = []

    for value in values:
        if isinstance(value, str):
            extension = normalize_extension(value)

        elif isinstance(value, dict):
            extension = normalize_extension(
                value.get(
                    "extension",
                    value.get("value", ""),
                )
            )

        else:
            extension = ""

        if extension and extension not in result:
            result.append(extension)

    return result


def get_data_source(
    data_source_id: str,
) -> dict:
    normalized_id = normalize_text(
        data_source_id
    )

    if not normalized_id:
        raise HTTPException(
            status_code=400,
            detail="データソースIDが指定されていません。",
        )

    db = get_firestore_client()

    document = (
        db.collection(DATA_SOURCE_COLLECTION)
        .document(normalized_id)
        .get()
    )

    if not document.exists:
        raise HTTPException(
            status_code=404,
            detail="データソースが見つかりません。",
        )

    data = document.to_dict() or {}

    extensions = (
        data.get("extensions")
        or data.get("file_extensions")
        or data.get("allowed_extensions")
        or []
    )

    return {
        "data_source_id": document.id,
        "data_source_name": data.get(
            "data_source_name",
            "",
        ),
        "source_type": normalize_source_type(
            data.get("source_type", "")
        ),
        "extensions": normalize_extensions(
            extensions
        ),
        "enabled": data.get(
            "enabled",
            True,
        ),
    }


def validate_data_source(
    data_source: dict,
) -> None:
    if not data_source.get("enabled", True):
        raise HTTPException(
            status_code=400,
            detail="無効なデータソースです。",
        )

    if data_source.get("source_type") != "file":
        raise HTTPException(
            status_code=400,
            detail="ファイル型のデータソースではありません。",
        )


def validate_file_extension(
    file_name: str,
    data_source: dict,
) -> str:
    extension = get_file_extension(
        file_name
    )

    if not extension:
        raise HTTPException(
            status_code=400,
            detail="拡張子のないファイルは取り込めません。",
        )

    allowed_extensions = data_source.get(
        "extensions",
        [],
    )

    if not allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "このデータソースには"
                "対象拡張子が設定されていません。"
            ),
        )

    if extension not in allowed_extensions:
        allowed_text = ", ".join(
            f".{item}"
            for item in allowed_extensions
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f".{extension}は対象外の拡張子です。"
                f" 使用可能: {allowed_text}"
            ),
        )

    return extension


def get_storage_bucket():
    return storage.Client().bucket(
        BUCKET_NAME
    )


def build_gcs_path(
    data_source_id: str,
    file_id: str,
    file_name: str,
) -> str:
    return (
        f"data-sources/{data_source_id}/"
        f"files/{file_id}/{file_name}"
    )


def upload_to_storage(
    upload_file: UploadFile,
    gcs_path: str,
) -> None:
    blob = get_storage_bucket().blob(
        gcs_path
    )

    try:
        upload_file.file.seek(0)

        blob.upload_from_file(
            upload_file.file,
            content_type=(
                upload_file.content_type
                or "application/octet-stream"
            ),
        )

    except Exception as error:
        print(
            "Cloud Storage upload error: "
            f"{type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Cloud Storageへの"
                "ファイル保存に失敗しました。"
            ),
        )


def delete_from_storage(
    gcs_path: str,
) -> None:
    if not gcs_path:
        return

    try:
        blob = get_storage_bucket().blob(
            gcs_path
        )

        if blob.exists():
            blob.delete()

    except Exception as error:
        print(
            "Cloud Storage delete error: "
            f"{type(error).__name__}: {error}"
        )


def find_same_name_file(
    data_source_id: str,
    file_name: str,
):
    db = get_firestore_client()

    documents = (
        db.collection(
            UPLOADED_FILE_COLLECTION
        )
        .where(
            "data_source_id",
            "==",
            data_source_id,
        )
        .where(
            "file_name_normalized",
            "==",
            file_name.lower(),
        )
        .where(
            "deleted",
            "==",
            False,
        )
        .limit(1)
        .stream()
    )

    return next(documents, None)


def save_file_document(
    *,
    file_id: str,
    data_source: dict,
    file_name: str,
    extension: str,
    upload_file: UploadFile,
    gcs_path: str,
    user: dict,
    is_update: bool,
) -> None:
    data = {
        "file_id": file_id,
        "data_source_id": data_source[
            "data_source_id"
        ],
        "data_source_name": data_source.get(
            "data_source_name",
            "",
        ),
        "file_name": file_name,
        "file_name_normalized": (
            file_name.lower()
        ),
        "extension": extension,
        "content_type": (
            upload_file.content_type
            or "application/octet-stream"
        ),
        "size_bytes": upload_file.size,
        "bucket_name": BUCKET_NAME,
        "gcs_path": gcs_path,
        "gcs_uri": (
            f"gs://{BUCKET_NAME}/{gcs_path}"
        ),
        "status": "uploaded",
        "deleted": False,
        "updated_at": (
            firestore.SERVER_TIMESTAMP
        ),
        "updated_by": user["email"],
    }

    if not is_update:
        data["created_at"] = (
            firestore.SERVER_TIMESTAMP
        )
        data["created_by"] = user["email"]

    (
        get_firestore_client()
        .collection(
            UPLOADED_FILE_COLLECTION
        )
        .document(file_id)
        .set(
            data,
            merge=True,
        )
    )


@router.post(
    "/file",
    status_code=201,
)
async def import_file(
    data_source_id: str = Form(...),
    overwrite: bool = Form(False),
    file: UploadFile = File(...),
    authorization: str = Header(...),
):
    user = authenticate_user(
        authorization
    )

    file_name = sanitize_file_name(
        file.filename or ""
    )

    data_source = get_data_source(
        data_source_id
    )

    validate_data_source(
        data_source
    )

    extension = validate_file_extension(
        file_name,
        data_source,
    )

    existing_document = find_same_name_file(
        data_source["data_source_id"],
        file_name,
    )

    if existing_document and not overwrite:
        existing_data = (
            existing_document.to_dict()
            or {}
        )

        raise HTTPException(
            status_code=409,
            detail={
                "code": "FILE_ALREADY_EXISTS",
                "message": (
                    "同名ファイルが既に"
                    "登録されています。"
                ),
                "existing_file_id": (
                    existing_data.get(
                        "file_id",
                        existing_document.id,
                    )
                ),
            },
        )

    if existing_document:
        existing_data = (
            existing_document.to_dict()
            or {}
        )

        file_id = (
            existing_data.get("file_id")
            or existing_document.id
        )

        old_gcs_path = existing_data.get(
            "gcs_path",
            "",
        )

        is_update = True

    else:
        file_id = uuid.uuid4().hex
        old_gcs_path = ""
        is_update = False

    gcs_path = build_gcs_path(
        data_source["data_source_id"],
        file_id,
        file_name,
    )

    upload_to_storage(
        file,
        gcs_path,
    )

    try:
        save_file_document(
            file_id=file_id,
            data_source=data_source,
            file_name=file_name,
            extension=extension,
            upload_file=file,
            gcs_path=gcs_path,
            user=user,
            is_update=is_update,
        )

    except Exception as error:
        delete_from_storage(
            gcs_path
        )

        print(
            "Firestore registration error: "
            f"{type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "ファイル管理情報の"
                "登録に失敗しました。"
            ),
        )

    if (
        is_update
        and old_gcs_path
        and old_gcs_path != gcs_path
    ):
        delete_from_storage(
            old_gcs_path
        )

    return {
        "message": (
            "ファイルを上書きしました。"
            if is_update
            else "ファイルを取り込みました。"
        ),
        "file_id": file_id,
        "file_name": file_name,
        "extension": extension,
        "data_source_id": (
            data_source["data_source_id"]
        ),
        "bucket_name": BUCKET_NAME,
        "gcs_path": gcs_path,
        "gcs_uri": (
            f"gs://{BUCKET_NAME}/{gcs_path}"
        ),
        "status": (
            "updated"
            if is_update
            else "created"
        ),
    }
