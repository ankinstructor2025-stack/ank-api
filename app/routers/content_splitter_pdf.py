from __future__ import annotations

import io
import re
from typing import Any

from pypdf import PdfReader


CONTROL_CHARS_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
MULTI_SPACES_RE = re.compile(r"[ \t]+")
MULTI_NEWLINES_RE = re.compile(r"\n{3,}")
BROKEN_NUMBER_LINES_RE = re.compile(r"(?:\n?\d+){8,}")
LONG_DIGITS_RE = re.compile(r"\d{20,}")


def clean_pdf_text(text: str) -> str:
    """
    PDF専用の軽い整形だけを行う。
    意味は変えず、row_data に入れる前の最低限のクレンジングに留める。
    """
    if not text:
        return ""

    # 文字化け気味データも安全側で通す
    text = str(text).encode("utf-8", "ignore").decode("utf-8")

    # 改行統一
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 制御文字除去
    text = CONTROL_CHARS_RE.sub("", text)

    # PDF由来の壊れた数字列を軽く除去
    text = BROKEN_NUMBER_LINES_RE.sub(" ", text)
    text = LONG_DIGITS_RE.sub(" ", text)

    # 空白整理
    text = MULTI_SPACES_RE.sub(" ", text)
    text = MULTI_NEWLINES_RE.sub("\n\n", text)

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
