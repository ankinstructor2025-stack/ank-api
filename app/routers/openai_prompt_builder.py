from __future__ import annotations

from typing import Optional

from .openai_chunking import Chunk, render_chunk_text


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def build_kokkai_prompt_text(
    *,
    job_item_id: str,
    prompt_template: str,
    chunk: Chunk,
    name_of_house: str | None = None,
    name_of_meeting: str | None = None,
    logical_name: str | None = None,
    parent_label: str | None = None,
) -> str:
    target_name = f"{name_of_house or ''} / {name_of_meeting or ''}".strip(" /")
    input_text = render_chunk_text(chunk, include_source_item_id=False)

    return (
        f"対象: {target_name or '国会議事録'}\n"
        f"job_item_id: {job_item_id}\n"
        f"chunk_no: {chunk.chunk_no}\n"
        f"chunk_item_count: {chunk.item_count}\n\n"
        f"{prompt_template.strip()}\n\n"
        f"【会議情報】\n"
        f"院: {name_of_house or ''}\n"
        f"会議名: {name_of_meeting or ''}\n"
        f"名称: {logical_name or parent_label or ''}\n"
        f"job_item_id: {job_item_id}\n"
        f"chunk_no: {chunk.chunk_no}\n"
        f"chunk_range: {chunk.start_index + 1}-{chunk.end_index + 1}\n\n"
        f"【発言一覧】\n"
        f"{input_text}\n"
    )


def build_opendata_prompt_text(
    *,
    job_item_id: str,
    prompt_template: str,
    chunk: Chunk,
    parent_source_id: str | None = None,
    parent_key1: str | None = None,
    parent_key2: str | None = None,
    parent_label: str | None = None,
    row_count: int | None = None,
) -> str:
    input_text = render_chunk_text(chunk, include_source_item_id=True)

    return (
        f"対象: {parent_label or parent_source_id or 'オープンデータ'}\n"
        f"job_item_id: {job_item_id}\n"
        f"chunk_no: {chunk.chunk_no}\n"
        f"chunk_item_count: {chunk.item_count}\n\n"
        f"{prompt_template.strip()}\n\n"
        f"【データセット情報】\n"
        f"parent_source_id: {parent_source_id or ''}\n"
        f"parent_key1: {parent_key1 or ''}\n"
        f"parent_key2: {parent_key2 or ''}\n"
        f"parent_label: {parent_label or ''}\n"
        f"row_count: {row_count or 0}\n"
        f"job_item_id: {job_item_id}\n"
        f"chunk_no: {chunk.chunk_no}\n"
        f"chunk_range: {chunk.start_index + 1}-{chunk.end_index + 1}\n\n"
        f"【行データ一覧】\n"
        f"{input_text}\n"
    )


def build_upload_prompt_text(
    *,
    job_item_id: str,
    prompt_template: str,
    chunk: Chunk,
    file_name: str | None = None,
    file_type: str | None = None,
    source_label: str | None = None,
    row_count: int | None = None,
) -> str:
    input_text = render_chunk_text(chunk, include_source_item_id=True)

    return (
        f"対象: {source_label or file_name or 'アップロードファイル'}\n"
        f"job_item_id: {job_item_id}\n"
        f"chunk_no: {chunk.chunk_no}\n"
        f"chunk_item_count: {chunk.item_count}\n\n"
        f"{prompt_template.strip()}\n\n"
        f"【ファイル情報】\n"
        f"file_name: {file_name or ''}\n"
        f"file_type: {file_type or ''}\n"
        f"row_count: {row_count or 0}\n"
        f"job_item_id: {job_item_id}\n"
        f"chunk_no: {chunk.chunk_no}\n"
        f"chunk_range: {chunk.start_index + 1}-{chunk.end_index + 1}\n\n"
        f"【入力データ一覧】\n"
        f"{input_text}\n"
    )
