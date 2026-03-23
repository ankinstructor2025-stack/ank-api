from __future__ import annotations

import json
import os
import re
import sqlite3
from copy import copy
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import firebase_admin
import requests
import ulid
from bs4 import BeautifulSoup
from bs4.element import Tag
from fastapi import APIRouter, Header, HTTPException
from firebase_admin import auth as fb_auth
from google.cloud import storage
from pydantic import BaseModel, Field


router = APIRouter(prefix="/public-url", tags=["public_url"])

JST = ZoneInfo("Asia/Tokyo")

BUCKET_NAME = os.getenv("UPLOAD_BUCKET", "ank-bucket")
TEMPLATE_PREFIX = os.getenv("TEMPLATE_PREFIX", "template")
PUBLIC_URL_CONFIG_NAME = "public_url.json"
DEFAULT_TIMEOUT_SEC = 15
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ank-bot/1.0)"
}

CONTENT_HINT_PATTERN = re.compile(
    r"main|content|contents|article|entry|post|pagebody|section|honbun|detail|wrapper",
    re.I,
)

NOISE_HINT_PATTERN = re.compile(
    r"breadcrumb|pankuzu|topicpath|sidebar|side|sidemenu|localnav|globalnav|gnav|subnav|"
    r"related|relation|share|sns|pagetop|search|utility|tool|header|footer|pickup|ranking|"
    r"banner|bnr|advert|ad-|ad_|menu|navi|navigation",
    re.I,
)

TOC_HINT_PATTERN = re.compile(
    r"目次|contents|index|ページ内リンク|このページ",
    re.I,
)


class PublicUrlRequest(BaseModel):
    source_key: str = Field(..., min_length=1)


class PublicUrlDecomposeRequest(BaseModel):
    page_url: str = Field(..., min_length=1)


@dataclass
class CrawlRule:
    same_domain_only: bool
    include_root_page: bool
    max_depth: int
    max_pages: int


@dataclass
class CrawlNode:
    url: str
    parent_url: str | None
    depth: int


def user_db_path(uid: str) -> str:
    return f"users/{uid}/ank.db"


def ensure_firebase_initialized():
    if firebase_admin._apps:
        return
    firebase_admin.initialize_app(options={"projectId": "ank-firebase"})


def get_uid_from_auth_header(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    token = authorization.replace("Bearer ", "", 1).strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")

    ensure_firebase_initialized()
    try:
        decoded = fb_auth.verify_id_token(token)
        uid = decoded.get("uid")
        if not uid:
            raise HTTPException(status_code=401, detail="uid not found in token")
        return uid
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid ID token: {e}")


def now_iso() -> str:
    return datetime.now(tz=JST).isoformat()


def make_id() -> str:
    return str(ulid.new())


def normalize_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{netloc}{path}{query}"


def same_domain(url1: str, url2: str) -> bool:
    return urlparse(url1).netloc.lower() == urlparse(url2).netloc.lower()


def is_html_like_url(url: str) -> bool:
    lower = (url or "").lower()
    blocked_exts = (
        ".pdf", ".zip", ".xls", ".xlsx", ".csv", ".json",
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
        ".doc", ".docx", ".ppt", ".pptx", ".xml",
        ".mp3", ".mp4", ".avi", ".mov",
    )
    return not lower.endswith(blocked_exts)


def load_public_url_config() -> dict[str, Any]:
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{TEMPLATE_PREFIX}/{PUBLIC_URL_CONFIG_NAME}")

        if not blob.exists():
            raise HTTPException(
                status_code=404,
                detail=f"config not found: gs://{BUCKET_NAME}/{TEMPLATE_PREFIX}/{PUBLIC_URL_CONFIG_NAME}",
            )

        text = blob.download_as_text(encoding="utf-8")
        data = json.loads(text)

        if not isinstance(data, dict):
            raise HTTPException(status_code=500, detail="invalid json structure")

        return data

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"invalid json: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to load config: {e}")


def get_public_url_source(config: dict[str, Any], source_key: str) -> dict[str, Any]:
    sources = config.get("sources")
    if not isinstance(sources, list):
        raise HTTPException(status_code=500, detail="sources not found in config")

    for s in sources:
        if isinstance(s, dict) and str(s.get("source_key") or "").strip() == source_key:
            return s

    raise HTTPException(status_code=404, detail=f"source_key not found: {source_key}")


def load_crawl_rule(config: dict[str, Any]) -> CrawlRule:
    crawl_conf = config.get("crawl")
    if not isinstance(crawl_conf, dict):
        raise HTTPException(status_code=500, detail="crawl not found in public_url.json")

    return CrawlRule(
        same_domain_only=bool(crawl_conf.get("same_domain_only", True)),
        include_root_page=bool(crawl_conf.get("include_root_page", False)),
        max_depth=int(crawl_conf.get("max_depth", 3)),
        max_pages=int(crawl_conf.get("max_pages", 200)),
    )


def load_page_scoring_config(config: dict[str, Any]) -> dict[str, Any]:
    page_scoring = config.get("page_scoring")
    if not isinstance(page_scoring, dict):
        raise HTTPException(status_code=500, detail="page_scoring not found in public_url.json")
    if page_scoring.get("enabled") is not True:
        raise HTTPException(status_code=500, detail="page_scoring.enabled must be true")
    return page_scoring


def get_source_url(source: dict[str, Any]) -> str:
    url = str(source.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required in public_url.json sources[]")
    return normalize_url(url)


def fetch_html_response(url: str) -> requests.Response:
    try:
        res = requests.get(url, headers=DEFAULT_HEADERS, timeout=DEFAULT_TIMEOUT_SEC)
        res.raise_for_status()
        res.encoding = res.apparent_encoding or res.encoding
        return res
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"public url fetch error: {e}")


def fetch_html(url: str) -> str:
    return fetch_html_response(url).text


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if href.startswith("#"):
            continue

        lower = href.lower()
        if lower.startswith(("javascript:", "mailto:", "tel:")):
            continue

        abs_url = urljoin(base_url, href)
        norm = normalize_url(abs_url)

        if norm not in seen:
            seen.add(norm)
            urls.append(norm)

    return urls


def filter_child_urls(
    urls: list[str],
    target_url: str,
    rule: CrawlRule,
    seen_urls: set[str],
) -> list[str]:
    result: list[str] = []

    for url in urls:
        if rule.same_domain_only and not same_domain(url, target_url):
            continue
        if url == target_url:
            continue
        if not is_html_like_url(url):
            continue
        if url in seen_urls:
            continue

        seen_urls.add(url)
        result.append(url)

    return result


def calc_short_line_ratio(text: str) -> float:
    if not text:
        return 0.0

    lines = []
    for x in re.split(r"[。\n\r]+", text):
        x = re.sub(r"\s+", " ", x).strip()
        if x:
            lines.append(x)

    if not lines:
        return 0.0

    short_count = sum(1 for line in lines if len(line) <= 20)
    return round(short_count / len(lines), 4)


def get_tag_hint_text(tag: Tag) -> str:
    class_names = " ".join(tag.get("class", [])) if tag.get("class") else ""
    tag_id = tag.get("id", "") or ""
    aria_label = tag.get("aria-label", "") or ""
    return f"{class_names} {tag_id} {aria_label}".strip()


def remove_global_noise(soup: BeautifulSoup) -> None:
    for tag in soup.select("script, style, noscript, svg, header, footer, nav, aside, form, iframe"):
        tag.decompose()

    for tag in list(soup.find_all(True)):
        hint = get_tag_hint_text(tag)
        if hint and NOISE_HINT_PATTERN.search(hint):
            tag.decompose()


def candidate_node_score(node: Tag) -> float:
    text = " ".join(node.stripped_strings)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return 0.0

    text_length = len(text)
    paragraph_count = len(
        [t for t in node.find_all(["p", "li", "dd", "dt"]) if len(" ".join(t.stripped_strings)) >= 30]
    )
    heading_count = len(node.find_all(["h1", "h2", "h3", "h4"]))
    link_count = len(node.find_all("a", href=True))

    score = float(text_length)
    score += paragraph_count * 180.0
    score += heading_count * 120.0
    score -= link_count * 20.0

    if text_length > 0:
        score -= (link_count / text_length) * 5000.0

    return score


def find_best_main_node(soup: BeautifulSoup) -> Tag:
    candidates: list[Tag] = []

    explicit = [
        soup.find("main"),
        soup.find("article"),
        soup.find(attrs={"role": "main"}),
    ]
    for node in explicit:
        if isinstance(node, Tag):
            candidates.append(node)

    for tag in soup.find_all(["div", "section", "article", "main"]):
        hint = get_tag_hint_text(tag)
        if hint and CONTENT_HINT_PATTERN.search(hint):
            candidates.append(tag)

    if soup.body:
        candidates.append(soup.body)

    unique_candidates: list[Tag] = []
    seen_ids: set[int] = set()
    for node in candidates:
        node_id = id(node)
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        unique_candidates.append(node)

    if not unique_candidates:
        return soup.body or soup

    return max(unique_candidates, key=candidate_node_score)


def remove_toc_like_nodes(main_node: Tag) -> None:
    for tag in list(main_node.find_all(["ul", "ol", "div", "section", "nav"])):
        text = " ".join(tag.stripped_strings)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue

        hint = get_tag_hint_text(tag)
        link_count = len(tag.find_all("a", href=True))
        li_count = len(tag.find_all("li"))
        text_length = len(text)

        if hint and TOC_HINT_PATTERN.search(hint) and link_count >= 3:
            tag.decompose()
            continue

        if text_length > 0 and link_count >= 5 and li_count >= 3:
            link_density = link_count / text_length
            if link_density >= 0.04 and text_length <= 800:
                tag.decompose()


def collect_content_blocks(main_node: Tag) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    for tag in main_node.find_all(["h1", "h2", "h3", "h4", "p", "li", "dd", "dt", "td", "th"]):
        text = " ".join(tag.stripped_strings)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue

        anchor_text = " ".join(a.get_text(" ", strip=True) for a in tag.find_all("a", href=True))
        anchor_text = re.sub(r"\s+", " ", anchor_text).strip()

        link_count = len(tag.find_all("a", href=True))
        is_heading = tag.name in {"h1", "h2", "h3", "h4"}
        is_short = len(text) <= 20
        is_link_only = bool(anchor_text) and anchor_text == text
        is_nav_like = is_link_only or (link_count >= 2 and is_short)

        blocks.append(
            {
                "tag": tag.name,
                "text": text,
                "length": len(text),
                "link_count": link_count,
                "is_heading": is_heading,
                "is_short": is_short,
                "is_link_only": is_link_only,
                "is_nav_like": is_nav_like,
            }
        )

    return blocks


def parse_content_features(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    remove_global_noise(soup)
    main_node = find_best_main_node(soup)
    remove_toc_like_nodes(main_node)

    blocks = collect_content_blocks(main_node)

    kept_blocks = [block["text"] for block in blocks if not block["is_link_only"]]

    text = "\n".join(kept_blocks).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    text_length = len(text)
    link_count = sum(block["link_count"] for block in blocks if not block["is_nav_like"])
    short_line_ratio = calc_short_line_ratio(text)

    heading_count = sum(1 for block in blocks if block["is_heading"])
    paragraph_count = sum(
        1
        for block in blocks
        if block["tag"] in {"p", "li", "dd", "dt", "td", "th"} and block["length"] >= 30 and not block["is_link_only"]
    )
    block_count = len(blocks)
    link_only_block_count = sum(1 for block in blocks if block["is_link_only"])
    nav_like_block_count = sum(1 for block in blocks if block["is_nav_like"])

    link_only_block_ratio = round(link_only_block_count / block_count, 4) if block_count > 0 else 0.0
    nav_like_block_ratio = round(nav_like_block_count / block_count, 4) if block_count > 0 else 0.0

    return {
        "title": title,
        "text": text.strip(),
        "text_length": text_length,
        "link_count": link_count,
        "short_line_ratio": short_line_ratio,
        "heading_count": heading_count,
        "paragraph_count": paragraph_count,
        "block_count": block_count,
        "link_only_block_ratio": link_only_block_ratio,
        "nav_like_block_ratio": nav_like_block_ratio,
    }


def extract_page_features(html: str) -> dict[str, Any]:
    return parse_content_features(html)


def extract_main_content_text(html: str) -> str:
    features = parse_content_features(html)
    return features["text"]


def apply_keyword_scores(
    target_text: str,
    keyword_scores: dict[str, Any],
    rule_name: str,
    reasons: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    score = 0
    matched_reasons: list[dict[str, Any]] = []

    if not target_text or not isinstance(keyword_scores, dict):
        return score, matched_reasons

    lower_text = target_text.lower()

    for keyword, delta in keyword_scores.items():
        try:
            delta_int = int(delta)
        except Exception:
            continue

        keyword_str = str(keyword)
        if keyword_str.lower() in lower_text:
            score += delta_int
            matched_reasons.append(
                {
                    "rule": rule_name,
                    "matched": keyword_str,
                    "score": delta_int,
                }
            )

    reasons.extend(matched_reasons)
    return score, matched_reasons


def apply_threshold_rule(
    value: float,
    rule_conf: dict[str, Any],
    rule_name: str,
    reasons: list[dict[str, Any]],
) -> int:
    if not isinstance(rule_conf, dict):
        return 0

    lt_value = rule_conf.get("lt")
    gte_value = rule_conf.get("gte")
    raw_score = rule_conf.get("score", 0)

    try:
        rule_score = int(raw_score)
    except Exception:
        return 0

    matched = True

    if lt_value is not None and not (value < float(lt_value)):
        matched = False

    if gte_value is not None and not (value >= float(gte_value)):
        matched = False

    if not matched:
        return 0

    reasons.append(
        {
            "rule": rule_name,
            "value": value,
            "score": rule_score,
        }
    )
    return rule_score


def apply_threshold_list(
    value: float,
    rule_list: list[dict[str, Any]],
    rule_name: str,
    reasons: list[dict[str, Any]],
) -> int:
    score = 0
    if not isinstance(rule_list, list):
        return score

    for item in rule_list:
        if isinstance(item, dict):
            score += apply_threshold_rule(value, item, rule_name, reasons)

    return score


def apply_structure_scores(
    text_length: int,
    heading_count: int,
    paragraph_count: int,
    short_line_ratio: float,
    link_only_block_ratio: float,
    nav_like_block_ratio: float,
    reasons: list[dict[str, Any]],
) -> int:
    score = 0

    def add(rule: str, delta: int, **extra):
        nonlocal score
        score += delta
        item = {"rule": rule, "score": delta}
        item.update(extra)
        reasons.append(item)

    if paragraph_count >= 10:
        add("structure_paragraph_rich", 18, paragraph_count=paragraph_count)
    elif paragraph_count >= 6:
        add("structure_paragraph_good", 12, paragraph_count=paragraph_count)
    elif paragraph_count >= 3:
        add("structure_paragraph_basic", 8, paragraph_count=paragraph_count)

    if heading_count >= 4:
        add("structure_heading_good", 10, heading_count=heading_count)
    elif heading_count >= 2:
        add("structure_heading_basic", 6, heading_count=heading_count)

    if text_length >= 1200 and paragraph_count >= 4:
        add("structure_dense_text", 8, text_length=text_length)

    if short_line_ratio <= 0.25 and text_length >= 600:
        add("structure_long_sentence", 8, short_line_ratio=short_line_ratio)
    elif short_line_ratio <= 0.40 and text_length >= 400:
        add("structure_sentence_ok", 4, short_line_ratio=short_line_ratio)

    if link_only_block_ratio >= 0.50:
        add("structure_link_only_blocks_high", -35, link_only_block_ratio=link_only_block_ratio)
    elif link_only_block_ratio >= 0.30:
        add("structure_link_only_blocks_mid", -20, link_only_block_ratio=link_only_block_ratio)
    elif link_only_block_ratio >= 0.15:
        add("structure_link_only_blocks_low", -10, link_only_block_ratio=link_only_block_ratio)

    if nav_like_block_ratio >= 0.50:
        add("structure_nav_like_high", -25, nav_like_block_ratio=nav_like_block_ratio)
    elif nav_like_block_ratio >= 0.30:
        add("structure_nav_like_mid", -15, nav_like_block_ratio=nav_like_block_ratio)

    if paragraph_count == 0 and heading_count <= 1 and text_length < 800:
        add("structure_content_thin", -15, text_length=text_length)

    return score


def match_decision(score_value: int, conf: dict[str, Any]) -> bool:
    if not isinstance(conf, dict):
        return False

    gte_value = conf.get("gte")
    lt_value = conf.get("lt")

    if gte_value is not None and score_value < int(gte_value):
        return False
    if lt_value is not None and score_value >= int(lt_value):
        return False

    return True


def judge_page(
    url: str,
    title: str,
    text: str,
    text_length: int,
    link_count: int,
    short_line_ratio: float,
    heading_count: int,
    paragraph_count: int,
    link_only_block_ratio: float,
    nav_like_block_ratio: float,
    page_scoring: dict[str, Any],
) -> dict[str, Any]:
    score = 0
    reasons: list[dict[str, Any]] = []

    rules = page_scoring.get("rules", {}) or {}
    thresholds = page_scoring.get("thresholds", {}) or {}
    decisions = page_scoring.get("decisions", {}) or {}

    keyword_total = 0

    title_keyword_score, _ = apply_keyword_scores(
        title or "",
        rules.get("title_keywords", {}) or {},
        "title_keywords",
        reasons,
    )
    body_keyword_score, _ = apply_keyword_scores(
        text or "",
        rules.get("body_keywords", {}) or {},
        "body_keywords",
        reasons,
    )
    url_keyword_score, _ = apply_keyword_scores(
        url or "",
        rules.get("url_keywords", {}) or {},
        "url_keywords",
        reasons,
    )

    keyword_total += title_keyword_score
    keyword_total += body_keyword_score
    keyword_total += url_keyword_score

    keyword_cap = 25
    if keyword_total > keyword_cap:
        reasons.append(
            {
                "rule": "keyword_cap",
                "score": keyword_cap - keyword_total,
                "raw_keyword_score": keyword_total,
                "applied_keyword_score": keyword_cap,
            }
        )
        score += keyword_cap
    else:
        score += keyword_total

    min_text_length = thresholds.get("min_text_length")
    if isinstance(min_text_length, dict):
        score += apply_threshold_rule(float(text_length), min_text_length, "min_text_length", reasons)

    score += apply_threshold_list(
        float(text_length),
        thresholds.get("text_length_bonus", []) or [],
        "text_length_bonus",
        reasons,
    )

    link_density = 0.0
    if text_length > 0:
        link_density = round(link_count / max(text_length, 1), 4)

    score += apply_threshold_list(
        link_density,
        thresholds.get("link_density", []) or [],
        "link_density",
        reasons,
    )

    score += apply_threshold_list(
        float(short_line_ratio),
        thresholds.get("short_line_ratio", []) or [],
        "short_line_ratio",
        reasons,
    )

    score += apply_structure_scores(
        text_length=text_length,
        heading_count=heading_count,
        paragraph_count=paragraph_count,
        short_line_ratio=short_line_ratio,
        link_only_block_ratio=link_only_block_ratio,
        nav_like_block_ratio=nav_like_block_ratio,
        reasons=reasons,
    )

    score = max(0, min(100, int(score)))

    decision = str(page_scoring.get("default_decision") or "pass").strip() or "pass"

    if match_decision(score, decisions.get("pass", {}) or {}):
        decision = "pass"
    elif match_decision(score, decisions.get("review", {}) or {}):
        decision = "review"
    elif match_decision(score, decisions.get("reject", {}) or {}):
        decision = "reject"

    filter_mode = str(page_scoring.get("filter_mode") or "score_only").strip()

    if filter_mode == "active_filter":
        is_usable = 1 if decision == "pass" else 0
    else:
        is_usable = 1 if decision in ("pass", "review") else 0

    return {
        "score": score,
        "decision": decision,
        "is_usable": is_usable,
        "decision_reason": json.dumps(
            {
                "filter_mode": filter_mode,
                "score": score,
                "link_density": link_density,
                "heading_count": heading_count,
                "paragraph_count": paragraph_count,
                "link_only_block_ratio": link_only_block_ratio,
                "nav_like_block_ratio": nav_like_block_ratio,
                "reasons": reasons if reasons else [{"rule": "no_rule_match", "score": 0}],
            },
            ensure_ascii=False,
        ),
    }


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table_name,),
    )
    return cur.fetchone() is not None


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    rows = cur.fetchall()
    return {str(row["name"]) for row in rows}


def require_table_columns(
    conn: sqlite3.Connection,
    table_name: str,
    required_columns: list[str],
) -> None:
    if not table_exists(conn, table_name):
        raise HTTPException(status_code=500, detail=f"required table not found: {table_name}")

    actual_columns = get_table_columns(conn, table_name)
    missing = [col for col in required_columns if col not in actual_columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"required columns not found in {table_name}: {', '.join(missing)}",
        )


def validate_public_url_schema(conn: sqlite3.Connection) -> None:
    require_table_columns(
        conn,
        "url_roots",
        [
            "root_id",
            "source_type",
            "root_url",
            "created_at",
        ],
    )

    require_table_columns(
        conn,
        "url_pages",
        [
            "page_id",
            "root_id",
            "parent_page_id",
            "page_url",
            "depth",
            "status",
            "child_count",
            "title",
            "content_type",
            "http_status",
            "fetched_at",
            "text_length",
            "link_count",
            "short_line_ratio",
            "score",
            "decision",
            "decision_reason",
            "is_usable",
            "created_at",
        ],
    )

    require_table_columns(
        conn,
        "url_page_contents",
        [
            "content_id",
            "page_id",
            "content_text",
            "created_at",
        ],
    )


def find_page_row(
    conn: sqlite3.Connection,
    root_id: str,
    page_url: str,
) -> sqlite3.Row | None:
    cur = conn.execute(
        """
        SELECT *
        FROM url_pages
        WHERE root_id = ? AND page_url = ?
        """,
        (root_id, page_url),
    )
    return cur.fetchone()


def find_page_with_root(conn: sqlite3.Connection, page_url: str) -> sqlite3.Row | None:
    cur = conn.execute(
        """
        SELECT
            p.page_id,
            p.root_id,
            p.page_url,
            p.depth,
            p.status,
            p.decision,
            p.is_usable,
            r.source_type AS source_key,
            r.root_url
        FROM url_pages p
        JOIN url_roots r
          ON p.root_id = r.root_id
        WHERE p.page_url = ?
        ORDER BY p.created_at DESC
        LIMIT 1
        """,
        (page_url,),
    )
    return cur.fetchone()


def insert_url_root_if_not_exists(
    conn: sqlite3.Connection,
    root_id: str,
    source_type: str,
    root_url: str,
) -> str:
    now = now_iso()

    cur = conn.execute(
        """
        SELECT root_id
        FROM url_roots
        WHERE root_url = ?
        """,
        (root_url,),
    )
    row = cur.fetchone()
    if row:
        return row["root_id"]

    conn.execute(
        """
        INSERT INTO url_roots (
            root_id,
            source_type,
            root_url,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (root_id, source_type, root_url, now),
    )
    return root_id


def upsert_url_page(
    conn: sqlite3.Connection,
    root_id: str,
    parent_page_id: str | None,
    page_url: str,
    depth: int,
    status: str,
    child_count: int,
    title: str | None,
    content_type: str | None,
    http_status: int | None,
    fetched_at: str | None,
    text_length: int,
    link_count: int,
    short_line_ratio: float,
    score: int,
    decision: str | None,
    decision_reason: str | None,
    is_usable: int,
) -> tuple[bool, str, str]:
    now = now_iso()
    existing = find_page_row(conn, root_id, page_url)

    if existing:
        page_id = existing["page_id"]

        current_status = str(existing["status"] or "new")
        if current_status == "done":
            next_status = "done"
        elif status == "fetch_error":
            next_status = "fetch_error"
        else:
            next_status = status

        conn.execute(
            """
            UPDATE url_pages
               SET parent_page_id = ?,
                   depth = ?,
                   status = ?,
                   child_count = ?,
                   title = ?,
                   content_type = ?,
                   http_status = ?,
                   fetched_at = ?,
                   text_length = ?,
                   link_count = ?,
                   short_line_ratio = ?,
                   score = ?,
                   decision = ?,
                   decision_reason = ?,
                   is_usable = ?
             WHERE page_id = ?
            """,
            (
                parent_page_id,
                depth,
                next_status,
                child_count,
                title,
                content_type,
                http_status,
                fetched_at,
                text_length,
                link_count,
                short_line_ratio,
                score,
                decision,
                decision_reason,
                is_usable,
                page_id,
            ),
        )
        return False, page_id, next_status

    page_id = make_id()
    conn.execute(
        """
        INSERT INTO url_pages (
            page_id,
            root_id,
            parent_page_id,
            page_url,
            depth,
            status,
            child_count,
            title,
            content_type,
            http_status,
            fetched_at,
            text_length,
            link_count,
            short_line_ratio,
            score,
            decision,
            decision_reason,
            is_usable,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            page_id,
            root_id,
            parent_page_id,
            page_url,
            depth,
            status,
            child_count,
            title,
            content_type,
            http_status,
            fetched_at,
            text_length,
            link_count,
            short_line_ratio,
            score,
            decision,
            decision_reason,
            is_usable,
            now,
        ),
    )
    return True, page_id, status


def replace_url_page_content(
    conn: sqlite3.Connection,
    page_id: str,
    content_text: str,
) -> tuple[str, bool]:
    now = now_iso()

    cur = conn.execute(
        """
        SELECT content_id
        FROM url_page_contents
        WHERE page_id = ?
        """,
        (page_id,),
    )
    row = cur.fetchone()

    if row:
        content_id = str(row["content_id"])
        conn.execute(
            """
            UPDATE url_page_contents
               SET content_text = ?,
                   created_at = ?
             WHERE content_id = ?
            """,
            (content_text, now, content_id),
        )
        return content_id, False

    content_id = make_id()
    conn.execute(
        """
        INSERT INTO url_page_contents (
            content_id,
            page_id,
            content_text,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (content_id, page_id, content_text, now),
    )
    return content_id, True


def build_page_results(
    target_url: str,
    rule: CrawlRule,
    page_scoring: dict[str, Any],
) -> list[dict[str, Any]]:
    root_res = fetch_html_response(target_url)
    root_html = root_res.text
    root_links_all = extract_links(root_html, target_url)

    seen_urls: set[str] = set()
    level1_urls = filter_child_urls(root_links_all, target_url, rule, seen_urls)

    stack: list[CrawlNode] = [
        CrawlNode(url=url, parent_url=None, depth=1)
        for url in reversed(level1_urls)
    ]

    page_results: list[dict[str, Any]] = []

    if rule.include_root_page:
        root_features = extract_page_features(root_html)
        root_judged = judge_page(
            url=target_url,
            title=root_features["title"],
            text=root_features["text"],
            text_length=root_features["text_length"],
            link_count=root_features["link_count"],
            short_line_ratio=root_features["short_line_ratio"],
            heading_count=root_features["heading_count"],
            paragraph_count=root_features["paragraph_count"],
            link_only_block_ratio=root_features["link_only_block_ratio"],
            nav_like_block_ratio=root_features["nav_like_block_ratio"],
            page_scoring=page_scoring,
        )
        page_results.append(
            {
                "page_id": None,
                "page_url": target_url,
                "parent_url": None,
                "parent_page_id": None,
                "depth": 0,
                "status": "new",
                "child_count": len(level1_urls),
                "title": root_features["title"],
                "content_type": root_res.headers.get("Content-Type", "text/html"),
                "http_status": root_res.status_code,
                "fetched_at": now_iso(),
                "text_length": root_features["text_length"],
                "link_count": root_features["link_count"],
                "short_line_ratio": root_features["short_line_ratio"],
                "score": root_judged["score"],
                "decision": root_judged["decision"],
                "decision_reason": root_judged["decision_reason"],
                "is_usable": root_judged["is_usable"],
                "created_at": now_iso(),
            }
        )

    while stack and len(page_results) < rule.max_pages:
        node = stack.pop()

        page_info = {
            "page_id": None,
            "page_url": node.url,
            "parent_url": node.parent_url,
            "parent_page_id": None,
            "depth": node.depth,
            "status": "new",
            "child_count": 0,
            "title": "",
            "content_type": None,
            "http_status": None,
            "fetched_at": None,
            "text_length": 0,
            "link_count": 0,
            "short_line_ratio": 0.0,
            "score": 0,
            "decision": "reject",
            "decision_reason": json.dumps(
                {
                    "filter_mode": str(page_scoring.get("filter_mode") or "score_only"),
                    "score": 0,
                    "reasons": [{"rule": "not_fetched", "score": 0}],
                },
                ensure_ascii=False,
            ),
            "is_usable": 0,
            "created_at": now_iso(),
        }

        try:
            res = fetch_html_response(node.url)
            html = res.text
            features = extract_page_features(html)

            page_info["title"] = features["title"]
            page_info["content_type"] = res.headers.get("Content-Type", "text/html")
            page_info["http_status"] = res.status_code
            page_info["fetched_at"] = now_iso()
            page_info["text_length"] = features["text_length"]
            page_info["link_count"] = features["link_count"]
            page_info["short_line_ratio"] = features["short_line_ratio"]

            child_urls: list[str] = []
            if node.depth < rule.max_depth:
                child_urls_all = extract_links(html, node.url)
                child_urls = filter_child_urls(child_urls_all, target_url, rule, seen_urls)

            page_info["child_count"] = len(child_urls)

            judged = judge_page(
                url=node.url,
                title=features["title"],
                text=features["text"],
                text_length=features["text_length"],
                link_count=features["link_count"],
                short_line_ratio=features["short_line_ratio"],
                heading_count=features["heading_count"],
                paragraph_count=features["paragraph_count"],
                link_only_block_ratio=features["link_only_block_ratio"],
                nav_like_block_ratio=features["nav_like_block_ratio"],
                page_scoring=page_scoring,
            )
            page_info["score"] = judged["score"]
            page_info["decision"] = judged["decision"]
            page_info["decision_reason"] = judged["decision_reason"]
            page_info["is_usable"] = judged["is_usable"]

            if node.depth < rule.max_depth:
                for child_url in reversed(child_urls):
                    if len(page_results) + len(stack) >= rule.max_pages:
                        break
                    stack.append(
                        CrawlNode(
                            url=child_url,
                            parent_url=node.url,
                            depth=node.depth + 1,
                        )
                    )

        except HTTPException:
            page_info["status"] = "fetch_error"
            page_info["decision"] = "reject"
            page_info["is_usable"] = 0
            page_info["decision_reason"] = json.dumps(
                {
                    "filter_mode": str(page_scoring.get("filter_mode") or "score_only"),
                    "score": 0,
                    "reasons": [{"rule": "fetch_error", "score": 0}],
                },
                ensure_ascii=False,
            )

        page_results.append(page_info)

    return page_results


@router.get("/sources")
def get_public_url_sources(
    authorization: str | None = Header(default=None),
):
    get_uid_from_auth_header(authorization)

    config = load_public_url_config()
    sources = config.get("sources") or []

    return {
        "source_type": "public_url",
        "sources": [
            {
                "source_key": str(s.get("source_key") or "").strip(),
                "label": str(s.get("label") or "").strip(),
                "url": str(s.get("url") or "").strip(),
            }
            for s in sources
            if isinstance(s, dict)
        ],
    }


@router.post("/register")
async def public_url_register(
    req: PublicUrlRequest,
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)

    config = load_public_url_config()
    source = get_public_url_source(config, req.source_key)
    rule = load_crawl_rule(config)
    page_scoring = load_page_scoring_config(config)
    target_url = get_source_url(source)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}",
        )

    local_db_path = f"/tmp/ank_{uid}_public_url.db"
    db_blob.download_to_filename(local_db_path)

    root_id = make_id()
    row_inserted = 0
    row_skipped = 0
    result_pages: list[dict[str, Any]] = []

    try:
        page_results = build_page_results(
            target_url=target_url,
            rule=rule,
            page_scoring=page_scoring,
        )

        conn = sqlite3.connect(local_db_path)
        conn.row_factory = sqlite3.Row
        try:
            validate_public_url_schema(conn)

            root_id = insert_url_root_if_not_exists(
                conn=conn,
                root_id=root_id,
                source_type=req.source_key,
                root_url=target_url,
            )

            url_to_page_id: dict[str, str] = {}

            for page in page_results:
                parent_page_id = None
                if page["parent_url"]:
                    parent_page_id = url_to_page_id.get(page["parent_url"])

                inserted, actual_page_id, actual_status = upsert_url_page(
                    conn=conn,
                    root_id=root_id,
                    parent_page_id=parent_page_id,
                    page_url=page["page_url"],
                    depth=page["depth"],
                    status=page["status"],
                    child_count=page["child_count"],
                    title=page["title"],
                    content_type=page["content_type"],
                    http_status=page["http_status"],
                    fetched_at=page["fetched_at"],
                    text_length=page["text_length"],
                    link_count=page["link_count"],
                    short_line_ratio=page["short_line_ratio"],
                    score=page["score"],
                    decision=page["decision"],
                    decision_reason=page["decision_reason"],
                    is_usable=page["is_usable"],
                )

                url_to_page_id[page["page_url"]] = actual_page_id

                if inserted:
                    row_inserted += 1
                else:
                    row_skipped += 1

                page["page_id"] = actual_page_id
                page["parent_page_id"] = parent_page_id
                page["status"] = actual_status
                result_pages.append(page)

            conn.commit()
        finally:
            conn.close()

        db_blob.upload_from_filename(local_db_path)

    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail=f"sqlite integrity error: {e}")
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"sqlite error: {e}")

    for page in result_pages:
        page.pop("parent_url", None)

    return {
        "mode": "public_url_register",
        "source_key": req.source_key,
        "root_url": target_url,
        "root_id": root_id,
        "max_depth": rule.max_depth,
        "max_pages": rule.max_pages,
        "page_count": len(result_pages),
        "row_inserted": row_inserted,
        "row_skipped": row_skipped,
        "pages": result_pages,
        "message": f"{req.source_key} public url registered",
    }


@router.post("/decompose")
async def public_url_decompose(
    req: PublicUrlDecomposeRequest,
    authorization: str | None = Header(default=None),
):
    uid = get_uid_from_auth_header(authorization)
    page_url = normalize_url(req.page_url)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    db_gcs_path = user_db_path(uid)
    db_blob = bucket.blob(db_gcs_path)
    if not db_blob.exists():
        raise HTTPException(
            status_code=400,
            detail=f"ank.db not found. call /v1/user/init first. path={db_gcs_path}",
        )

    local_db_path = f"/tmp/ank_{uid}_public_url.db"
    db_blob.download_to_filename(local_db_path)

    try:
        conn = sqlite3.connect(local_db_path)
        conn.row_factory = sqlite3.Row
        try:
            validate_public_url_schema(conn)

            page_row = find_page_with_root(conn, page_url)
            if not page_row:
                raise HTTPException(status_code=404, detail="page not found")

            html = fetch_html(page_url)
            content_text = extract_main_content_text(html)

            if not content_text:
                raise HTTPException(status_code=400, detail="no content extracted")

            content_id, inserted = replace_url_page_content(
                conn=conn,
                page_id=str(page_row["page_id"]),
                content_text=content_text,
            )

            conn.execute(
                """
                UPDATE url_pages
                   SET status = 'done'
                 WHERE page_id = ?
                """,
                (str(page_row["page_id"]),),
            )

            conn.commit()
        finally:
            conn.close()

        db_blob.upload_from_filename(local_db_path)

        return {
            "mode": "public_url_decompose",
            "page_url": page_url,
            "page_id": str(page_row["page_id"]),
            "content_id": content_id,
            "source_key": str(page_row["source_key"]),
            "content_length": len(content_text),
            "inserted": inserted,
            "message": "public url content saved",
        }

    except HTTPException:
        raise
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail=f"sqlite integrity error: {e}")
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"sqlite error: {e}")
