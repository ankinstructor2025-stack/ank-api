from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


@dataclass
class ChunkConfig:
    max_chars: int
    max_items: int
    overlap_items: int = 0


@dataclass
class ChunkItem:
    sort_no: int
    text: str
    source_item_id: Optional[str] = None
    content_type: Optional[str] = None
    row_id: Optional[str] = None


@dataclass
class Chunk:
    chunk_no: int
    start_index: int
    end_index: int
    items: list[ChunkItem]

    @property
    def item_count(self) -> int:
        return len(self.items)

    @property
    def char_count(self) -> int:
        return sum(len(x.text or "") for x in self.items)


def _to_chunk_item(row: Any) -> ChunkItem:
    if isinstance(row, ChunkItem):
        return row

    if isinstance(row, dict):
        return ChunkItem(
            sort_no=int(row.get("sort_no") or 0),
            text=normalize_text(row.get("content_text") or row.get("text")),
            source_item_id=row.get("source_item_id"),
            content_type=row.get("content_type"),
            row_id=row.get("row_id"),
        )

    return ChunkItem(
        sort_no=int(row["sort_no"] or 0),
        text=normalize_text(row["content_text"] if "content_text" in row.keys() else row["text"]),
        source_item_id=row["source_item_id"] if "source_item_id" in row.keys() else None,
        content_type=row["content_type"] if "content_type" in row.keys() else None,
        row_id=row["row_id"] if "row_id" in row.keys() else None,
    )


def _estimate_joined_chars(items: list[ChunkItem]) -> int:
    if not items:
        return 0
    return sum(len(x.text) for x in items) + (len(items) - 1) * 2


def _split_text_by_limit(text: str, max_chars: int) -> list[str]:
    s = normalize_text(text)
    if not s:
        return []

    if len(s) <= max_chars:
        return [s]

    parts: list[str] = []
    rest = s

    while rest:
        if len(rest) <= max_chars:
            parts.append(rest.strip())
            break

        window = rest[:max_chars]

        split_pos = -1

        # 句点・終端記号優先
        for m in re.finditer(r"[。．！？!?]\s*", window):
            split_pos = m.end()

        # なければ読点や区切り
        if split_pos <= 0:
            for m in re.finditer(r"[、，,；;:]\s*", window):
                split_pos = m.end()

        # それもなければ空白
        if split_pos <= 0:
            split_pos = window.rfind(" ")

        # 最後は強制分割
        if split_pos <= 0:
            split_pos = max_chars

        head = rest[:split_pos].strip()
        if head:
            parts.append(head)

        rest = rest[split_pos:].strip()

    return [x for x in parts if x]


def _split_item_if_needed(item: ChunkItem, max_chars: int) -> list[ChunkItem]:
    text = normalize_text(item.text)
    if not text:
        return []

    if len(text) <= max_chars:
        return [item]

    split_texts = _split_text_by_limit(text, max_chars)
    if not split_texts:
        return []

    split_items: list[ChunkItem] = []
    for idx, split_text in enumerate(split_texts, start=1):
        suffix = f":part_{idx}" if len(split_texts) > 1 else ""
        split_items.append(
            ChunkItem(
                sort_no=item.sort_no,
                text=split_text,
                source_item_id=(item.source_item_id or None) if not suffix else f"{item.source_item_id or ''}{suffix}".strip() or None,
                content_type=item.content_type,
                row_id=(item.row_id or None) if not suffix else f"{item.row_id or ''}{suffix}".strip() or None,
            )
        )

    return split_items


def build_chunks(
    rows: Iterable[Any],
    config: ChunkConfig,
    *,
    allowed_content_types: Optional[set[str]] = None,
) -> list[Chunk]:
    if int(config.max_chars) <= 0:
        raise RuntimeError("ChunkConfig.max_chars must be greater than 0")

    if int(config.max_items) <= 0:
        raise RuntimeError("ChunkConfig.max_items must be greater than 0")

    items: list[ChunkItem] = []

    for row in rows:
        item = _to_chunk_item(row)
        if not item.text:
            continue

        if allowed_content_types is not None:
            ct = (item.content_type or "").strip()
            if ct not in allowed_content_types:
                continue

        split_items = _split_item_if_needed(item, int(config.max_chars))
        items.extend(split_items)

    if not items:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_no = 1

    while start < len(items):
        current: list[ChunkItem] = []
        current_chars = 0
        i = start

        while i < len(items):
            candidate = items[i]
            candidate_len = len(candidate.text)
            sep_len = 2 if current else 0

            if current and len(current) >= config.max_items:
                break

            if current and (current_chars + sep_len + candidate_len) > config.max_chars:
                break

            # 念のため単体超過はここでも拒否
            if not current and candidate_len > config.max_chars:
                raise RuntimeError(
                    f"single chunk item exceeds max_chars: len={candidate_len}, max_chars={config.max_chars}, source_item_id={candidate.source_item_id}"
                )

            current.append(candidate)
            current_chars += sep_len + candidate_len
            i += 1

        if not current:
            current = [items[start]]
            i = start + 1

        chunk = Chunk(
            chunk_no=chunk_no,
            start_index=start,
            end_index=i - 1,
            items=current,
        )
        chunks.append(chunk)

        if i >= len(items):
            break

        overlap = max(0, config.overlap_items)
        next_start = i - overlap
        if next_start <= start:
            next_start = i

        start = next_start
        chunk_no += 1

    return chunks


def render_chunk_lines(
    chunk: Chunk,
    *,
    include_source_item_id: bool = True,
) -> list[str]:
    lines: list[str] = []

    for item in chunk.items:
        if item.text:
            lines.append(item.text)

    return lines


def render_chunk_text(
    chunk: Chunk,
    *,
    include_source_item_id: bool = True,
) -> str:
    return "\n\n".join(
        render_chunk_lines(chunk, include_source_item_id=include_source_item_id)
    ).strip()
