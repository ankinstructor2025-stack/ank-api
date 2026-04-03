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

LINE_NO_RE = re.compile(r"^\[\d+\]\s*")
PAGE_NO_ONLY_RE = re.compile(r"^\d{1,4}$")
INDD_LINE_RE = re.compile(r"^.*\.indd.*$", re.IGNORECASE)
DATETIME_RE = re.compile(r"\d{4}/\d{2}/\d{2}\s+\d{1,2}:\d{2}:\d{2}")


def clean_pdf_text(text: str) -> str:
    """
    PDF専用の軽い整形だけを行う。
    意味は変えず、row_data に入れる前の最低限のクレンジングに留める。
    """
    if not text:
        return ""

    text = str(text).encode("utf-8", "ignore").decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 制御文字除去
    text = CONTROL_CHARS_RE.sub("", text)

    # 行単位のノイズ除去
    cleaned_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        # 先頭の [142] などを除去
        line = LINE_NO_RE.sub("", line).strip()

        # 行中の日時除去
        line = DATETIME_RE.sub("", line).strip()

        # .indd 行は除去
        if INDD_LINE_RE.fullmatch(line):
            continue

        # 単独ページ番号は除去
        if PAGE_NO_ONLY_RE.fullmatch(line):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 明らかに壊れた数字列を軽く除去
    text = BROKEN_NUMBER_LINES_RE.sub(" ", text)
    text = LONG_DIGITS_RE.sub(" ", text)

    # 空白整理
    text = MULTI_SPACES_RE.sub(" ", text)

    # 単独改行はスペースに寄せる
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 連続改行整理
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
