from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from openai import OpenAI


logger = logging.getLogger(__name__)


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


def run_llm_json(
    prompt_text: str,
    log_prefix: str,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0,
    response_format: Optional[dict[str, Any]] = None,
) -> dict:
    client = get_openai_client()

    if response_format is None:
        response_format = {"type": "json_object"}

    try:
        logger.info("%s request start", log_prefix)

        res = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format=response_format,
            messages=[
                {"role": "user", "content": prompt_text}
            ],
        )

        logger.info("%s response received", log_prefix)

        content = res.choices[0].message.content or ""
        content = content.strip()

        logger.info("%s raw content: %s", log_prefix, content[:2000])

        result = parse_llm_json(content)

        logger.info("%s JSON parse success", log_prefix)
        return result

    except Exception:
        logger.exception("%s generation failed", log_prefix)
        raise


def run_chunked_llm_json(
    prompt_texts: list[str],
    log_prefix: str,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0,
) -> list[dict]:
    results: list[dict] = []

    for idx, prompt_text in enumerate(prompt_texts, start=1):
        chunk_prefix = f"{log_prefix} chunk={idx}/{len(prompt_texts)}"
        result = run_llm_json(
            prompt_text,
            chunk_prefix,
            model=model,
            temperature=temperature,
        )
        results.append(result)

    return results
