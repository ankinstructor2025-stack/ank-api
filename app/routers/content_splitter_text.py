from __future__ import annotations

import re
from typing import Any


QUESTION_START_RE = re.compile(r"^\s*(\d+)[\.\)]\s+")
QA_QUESTION_RE = re.compile(r"^\s*(q|question|問い|質問)[:：]\s*", re.IGNORECASE)
QA_ANSWER_RE = re.compile(r"^\s*(a|answer|回答|答え)[:：]\s*", re.IGNORECASE)

CONTROL_CHARS_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
MULTI_SPACES_RE = re.compile(r"[ \t]+")
MULTI_NEWLINES_RE = re.compile(r"\n{3,}")
LONG_DIGITS_RE = re.compile(r"\d{20,}")


def safe_decode(binary: bytes) -> str:
    """
    日本語テキスト向けのデコード。
    優先順:
      1. utf-8
      2. shift_jis
      3. cp932
    どれでも読めない場合は補完せず例外にする。
    """
    last_error: Exception | None = None
    for encoding in ("utf-8", "shift_jis", "cp932"):
        try:
            return binary.decode(encoding)
        except Exception as e:
            last_error = e
    raise UnicodeDecodeError(
        "multi",
        binary,
        0,
        min(len(binary), 1),
        f"failed to decode with utf-8, shift_jis, cp932: {last_error}",
    )


def clean_text(text: str) -> str:
    """
    テキスト専用の軽い整形。
    意味は変えず、分解しやすい形にそろえる。
    """
    if not text:
        return ""

    text = str(text).encode("utf-8", "ignore").decode("utf-8")
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS_RE.sub("", text)
    text = LONG_DIGITS_RE.sub(" ", text)
    text = MULTI_SPACES_RE.sub(" ", text)
    text = MULTI_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]


def count_question_starts(lines: list[str]) -> int:
    return sum(1 for line in lines if QUESTION_START_RE.match(line))


def count_qa_markers(lines: list[str]) -> int:
    count = 0
    for line in lines:
        if QA_QUESTION_RE.match(line) or QA_ANSWER_RE.match(line):
            count += 1
    return count


def looks_like_quiz_text(lines: list[str]) -> bool:
    """
    連番問題っぽいかをざっくり判定。
    """
    return count_question_starts(lines) >= 2


def looks_like_qa_text(lines: list[str]) -> bool:
    """
    Q:/A: 形式っぽいかをざっくり判定。
    """
    return count_qa_markers(lines) >= 2


def build_record(
    *,
    record_type: str,
    index: int,
    text: str,
    title: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "record_type": record_type,
        "index": index,
        "title": title,
        "text": text.strip(),
        "char_count": len(text.strip()),
        "line_count": len([x for x in text.split("\n") if x.strip()]),
    }
    if extra:
        record.update(extra)
    return record


def split_quiz_blocks(text: str, max_rows: int = 2000) -> list[dict[str, Any]]:
    """
    1. / 2. / 3. のような連番問題を、設問単位でまとめる。
    """
    lines = non_empty_lines(text)
    records: list[dict[str, Any]] = []

    current_no = ""
    current_lines: list[str] = []

    def flush():
        nonlocal current_no, current_lines, records
        if not current_lines:
            return

        block_text = "\n".join(current_lines).strip()
        records.append(
            build_record(
                record_type="question_block",
                index=len(records),
                title=current_no,
                text=block_text,
                extra={"question_no": current_no},
            )
        )
        current_no = ""
        current_lines = []

    for line in lines:
        m = QUESTION_START_RE.match(line)
        if m:
            flush()
            current_no = m.group(1)
            current_lines = [line]
        else:
            current_lines.append(line)

        if len(records) >= max_rows:
            break

    if len(records) < max_rows:
        flush()

    return records[:max_rows]


def split_qa_blocks(text: str, max_rows: int = 2000) -> list[dict[str, Any]]:
    """
    Q:/A: 形式を1セット単位でまとめる。
    """
    lines = non_empty_lines(text)
    records: list[dict[str, Any]] = []

    current_q: list[str] = []
    current_a: list[str] = []
    mode = ""

    def flush():
        nonlocal current_q, current_a, records, mode
        if not current_q and not current_a:
            return

        q_text = "\n".join(current_q).strip()
        a_text = "\n".join(current_a).strip()
        merged = []
        if q_text:
            merged.append(f"Q: {q_text}")
        if a_text:
            merged.append(f"A: {a_text}")

        records.append(
            build_record(
                record_type="qa_block",
                index=len(records),
                title=f"qa_{len(records) + 1}",
                text="\n".join(merged).strip(),
                extra={
                    "question": q_text,
                    "answer": a_text,
                },
            )
        )
        current_q = []
        current_a = []
        mode = ""

    for line in lines:
        if QA_QUESTION_RE.match(line):
            if current_q or current_a:
                flush()
            mode = "q"
            current_q.append(QA_QUESTION_RE.sub("", line).strip())
            continue

        if QA_ANSWER_RE.match(line):
            mode = "a"
            current_a.append(QA_ANSWER_RE.sub("", line).strip())
            continue

        if mode == "q":
            current_q.append(line)
        elif mode == "a":
            current_a.append(line)
        else:
            current_q.append(line)

        if len(records) >= max_rows:
            break

    if len(records) < max_rows:
        flush()

    return records[:max_rows]


def split_paragraph_blocks(text: str, max_rows: int = 2000) -> list[dict[str, Any]]:
    """
    通常テキストは空行区切りで段落分割。
    """
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]

    records: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks[:max_rows]):
        first_line = chunk.split("\n", 1)[0].strip()
        title = first_line[:80]
        records.append(
            build_record(
                record_type="paragraph",
                index=idx,
                title=title,
                text=chunk,
            )
        )

    return records


def split_text_records(
    binary: bytes,
    encoding: str = "utf-8",
    max_rows: int = 2000,
) -> list[dict[str, Any]]:
    """
    テキスト系の共通 splitter。
    優先順:
      1. 連番問題
      2. Q/A 形式
      3. 段落分割
    """
    raw_text = safe_decode(binary)
    text = clean_text(raw_text)

    if not text:
        return []

    lines = non_empty_lines(text)

    if looks_like_quiz_text(lines):
        records = split_quiz_blocks(text, max_rows=max_rows)
        if records:
            return records

    if looks_like_qa_text(lines):
        records = split_qa_blocks(text, max_rows=max_rows)
        if records:
            return records

    return split_paragraph_blocks(text, max_rows=max_rows)
