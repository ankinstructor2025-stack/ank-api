from __future__ import annotations

import io
import re
from typing import Any

from pypdf import PdfReader


CONTROL_CHARS_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
MULTI_SPACES_RE = re.compile(r"[ \t]+")
BROKEN_NUMBER_LINES_RE = re.compile(r"(?:\n?\d+){8,}")
LONG_DIGITS_RE = re.compile(r"\d{20,}")

LINE_NO_RE = re.compile(r"^\[\d+\]\s*")
PAGE_NO_ONLY_RE = re.compile(r"^\d{1,4}$")
INDD_LINE_RE = re.compile(r"^.*\.indd.*$", re.IGNORECASE)
DATETIME_RE = re.compile(r"\d{4}/\d{2}/\d{2}\s+\d{1,2}:\d{2}:\d{2}")
URL_RE = re.compile(r"https?://\S+")
SECTION_HEADER_RE = re.compile(r"^第\s*[0-9０-９一二三四五六七八九十]+\s*[章節].*$")
BULLET_SECTION_RE = re.compile(r"^第\s*[0-9０-９一二三四五六七八九十]+\s*節\s*●.*$")
ONLY_SYMBOLS_RE = re.compile(r"^[\s\.\-–—・●○◯。､,，:：/／\(\)（）]+$")


def normalize_line_text(line: str) -> str:
    line = line.strip()
    if not line:
        return ""

    line = LINE_NO_RE.sub("", line).strip()
    line = DATETIME_RE.sub("", line).strip()
    line = URL_RE.sub("", line).strip()

    line = re.sub(r"[ ]{2,}", " ", line)
    line = re.sub(r"[。．]\s*[。．]+", "。", line)
    line = re.sub(r"\s+[。．、，]", lambda m: m.group(0).strip(), line)

    return line.strip()


def should_drop_line(line: str) -> bool:
    if not line:
        return True

    if INDD_LINE_RE.fullmatch(line):
        return True

    if PAGE_NO_ONLY_RE.fullmatch(line):
        return True

    if ONLY_SYMBOLS_RE.fullmatch(line):
        return True

    if (SECTION_HEADER_RE.fullmatch(line) or BULLET_SECTION_RE.fullmatch(line)) and len(line) <= 40:
        return True

    if len(line) <= 8 and re.fullmatch(r"[\wぁ-んァ-ヶ一-龥。．、，・ ]*", line):
        return True

    return False


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

    cleaned_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = normalize_line_text(raw_line)

        if should_drop_line(line):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 明らかに壊れた数字列を軽く除去
    text = BROKEN_NUMBER_LINES_RE.sub(" ", text)
    text = LONG_DIGITS_RE.sub(" ", text)

    # 空白整理
    text = MULTI_SPACES_RE.sub(" ", text)

    # 単独改行はスペースにする
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 連続改行は1つにする
    text = re.sub(r"\n{2,}", "\n", text)

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
