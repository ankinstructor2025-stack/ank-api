from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import urlparse


SUPPORTED_KINDS = {"csv", "pdf", "json", "html"}


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def guess_ext_from_url(url: str) -> str:
    path = urlparse(url).path or ""
    if "." not in path:
        return ""
    return path.rsplit(".", 1)[-1].lower().strip()


def detect_content_kind(
    *,
    filename: str = "",
    source_path: str = "",
    content_type: str = "",
    declared_format: str = "",
    mimetype: str = "",
) -> str:
    """
    汎用判定。
    upload / opendata / public_url のどこから呼んでもよいように、
    引数は全部 optional にしている。
    """
    fmt = normalize_text(declared_format).lower()
    mime = normalize_text(mimetype).lower()
    ctype = normalize_text(content_type).lower()

    ext_candidates = [
        guess_ext_from_url(filename),
        guess_ext_from_url(source_path),
        normalize_text(filename).rsplit(".", 1)[-1].lower().strip() if "." in normalize_text(filename) else "",
    ]

    if fmt == "csv" or "csv" in mime or "csv" in ctype or "csv" in ext_candidates:
        return "csv"

    if fmt == "pdf" or "pdf" in mime or "pdf" in ctype or "pdf" in ext_candidates:
        return "pdf"

    if fmt == "json" or "json" in mime or "json" in ctype or "json" in ext_candidates:
        return "json"

    if fmt == "html" or "html" in mime or "html" in ctype or "html" in ext_candidates or "htm" in ext_candidates:
        return "html"

    return ""


def detect_resource_kind(resource: Mapping[str, Any], source_path: str, content_type: str = "") -> str:
    """
    opendata.py からそのまま置き換えやすい形。
    resource dict の format / mimetype と、URL・Content-Type を見て判定する。
    """
    return detect_content_kind(
        filename=normalize_text(resource.get("name")),
        source_path=source_path,
        content_type=content_type,
        declared_format=normalize_text(resource.get("format")),
        mimetype=normalize_text(resource.get("mimetype")),
    )
