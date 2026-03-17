from __future__ import annotations

import io
import re
from typing import Any

from pypdf import PdfReader


def clean_pdf_text(text: str) -> str:
    """
    PDF専用の軽い整形だけを行う。
    CSVとは完全に分離しておく。
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = re.sub(r"[\u0001-\u0008\u000b\u000c\u000e-\u001f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_pdf_records(binary: bytes, max_rows: int = 2000) -> list[dict[str, Any]]:
    reader = PdfReader(io.BytesIO(binary))
    pages: list[dict[str, Any]] = []

    for index, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""

        text = clean_pdf_text(raw_text)
        pages.append(
            {
                "page": index,
                "text": text,
                "char_count": len(text),
            }
        )

        if len(pages) >= max_rows:
            break

    return pages
