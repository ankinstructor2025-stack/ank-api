from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

from openai import OpenAI


logger = logging.getLogger(__name__)


DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_RETRY_COUNT = int(os.environ.get("OPENAI_RETRY_COUNT", "3"))
DEFAULT_RETRY_SLEEP_SEC = float(os.environ.get("OPENAI_RETRY_SLEEP_SEC", "2"))
DEFAULT_CHUNK_SLEEP_SEC = float(os.environ.get("OPENAI_CHUNK_SLEEP_SEC", "1"))


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def strip_code_fence(text: str) -> str:
    s = (text or "").strip()

    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    if s.lower().startswith("json"):
        s = s[4:].strip()

    return s


def extract_json_candidate(text: str) -> str:
    s = strip_code_fence(text)

    start_obj = s.find("{")
    start_arr = s.find("[")
    candidates = [x for x in [start_obj, start_arr] if x >= 0]

    if not candidates:
        return s

    start = min(candidates)
    return s[start:].strip()


def parse_llm_json(text: str) -> dict:
    raw = extract_json_candidate(text)

    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    repaired = raw
    repaired = repaired.replace("\r\n", "\n").replace("\r", "\n")
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    repaired = re.sub(
        r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)',
        r'\1"\2"\3',
        repaired,
    )

    try:
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else {}
    except Exception as e:
        logger.error("parse_llm_json failed. raw=%s", raw[:2000])
        logger.error("parse_llm_json repaired=%s", repaired[:2000])
        raise e


def _create_chat_completion(
    *,
    client: OpenAI,
    prompt_text: str,
    model: str,
    temperature: float,
    response_format: Optional[dict[str, Any]],
):
    return client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format=response_format,
        messages=[
            {"role": "user", "content": prompt_text}
        ],
    )


def run_llm_json(
    prompt_text: str,
    log_prefix: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0,
    response_format: Optional[dict[str, Any]] = None,
    retry_count: int = DEFAULT_RETRY_COUNT,
    retry_sleep_sec: float = DEFAULT_RETRY_SLEEP_SEC,
) -> dict:
    client = get_openai_client()

    if response_format is None:
        response_format = {"type": "json_object"}

    last_error: Exception | None = None

    for attempt in range(1, retry_count + 1):
        try:
            logger.info("%s request start attempt=%s/%s", log_prefix, attempt, retry_count)

            res = _create_chat_completion(
                client=client,
                prompt_text=prompt_text,
                model=model,
                temperature=temperature,
                response_format=response_format,
            )

            logger.info("%s response received attempt=%s/%s", log_prefix, attempt, retry_count)

            content = res.choices[0].message.content or ""
            content = content.strip()

            logger.info("%s raw content: %s", log_prefix, content[:2000])

            result = parse_llm_json(content)

            logger.info("%s JSON parse success attempt=%s/%s", log_prefix, attempt, retry_count)
            return result

        except Exception as e:
            last_error = e
            logger.exception("%s generation failed attempt=%s/%s", log_prefix, attempt, retry_count)

            if attempt >= retry_count:
                break

            sleep_sec = retry_sleep_sec * attempt
            logger.warning(
                "%s retry sleep %.1f sec before next attempt",
                log_prefix,
                sleep_sec,
            )
            time.sleep(sleep_sec)

    if last_error is not None:
        raise last_error

    raise RuntimeError(f"{log_prefix} generation failed without explicit exception")


def run_chunked_llm_json(
    prompt_texts: list[str],
    log_prefix: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0,
    retry_count: int = DEFAULT_RETRY_COUNT,
    retry_sleep_sec: float = DEFAULT_RETRY_SLEEP_SEC,
    chunk_sleep_sec: float = DEFAULT_CHUNK_SLEEP_SEC,
) -> list[dict]:
    results: list[dict] = []

    total = len(prompt_texts)
    logger.info("%s chunk execution start total=%s", log_prefix, total)

    for idx, prompt_text in enumerate(prompt_texts, start=1):
        chunk_prefix = f"{log_prefix} chunk={idx}/{total}"

        logger.info("%s start", chunk_prefix)

        result = run_llm_json(
            prompt_text,
            chunk_prefix,
            model=model,
            temperature=temperature,
            retry_count=retry_count,
            retry_sleep_sec=retry_sleep_sec,
        )
        results.append(result)

        logger.info("%s done", chunk_prefix)

        if idx < total and chunk_sleep_sec > 0:
            logger.info("%s sleep %.1f sec before next chunk", chunk_prefix, chunk_sleep_sec)
            time.sleep(chunk_sleep_sec)

    logger.info("%s chunk execution finished total=%s", log_prefix, total)
    return results
