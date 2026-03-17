from __future__ import annotations

import csv
import io
from typing import Any, Optional


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
        rows.append(dict(row))
        if len(rows) >= max_rows:
            break
    return rows
