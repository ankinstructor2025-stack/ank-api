from __future__ import annotations

import csv
import io
import re
from typing import Any, Optional


CONTROL_CHARS_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
MULTI_SPACES_RE = re.compile(r"[ \t]+")


def clean_csv_cell(value: Any) -> str:
    """
    CSVセル専用の軽い整形。
    構造は壊さず、セル文字列だけ軽くクレンジングする。
    """
    if value is None:
        return ""

    text = str(value).encode("utf-8", "ignore").decode("utf-8")
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS_RE.sub("", text)
    text = MULTI_SPACES_RE.sub(" ", text)
    return text.strip()


def count_csv_rows_from_binary(binary: bytes, encoding: str = "utf-8") -> Optional[int]:
    try:
        text = binary.decode(encoding, errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        count = 0
        for _ in reader:
            count += 1
        return count
    except Exception:
        return None


def split_csv_records(
    binary: bytes,
    encoding: str = "utf-8",
    max_rows: int = 2000,
) -> list[dict[str, Any]]:
    text = binary.decode(encoding, errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    rows: list[dict[str, Any]] = []
    for row in reader:
        cleaned_row = {key: clean_csv_cell(value) for key, value in dict(row).items()}
        rows.append(cleaned_row)
        if len(rows) >= max_rows:
            break
    return rows
