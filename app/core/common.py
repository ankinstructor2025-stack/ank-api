import base64
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from google.cloud import tasks_v2
from google.protobuf import duration_pb2  # ← 追加


PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("TASKS_LOCATION", "asia-northeast1")
CLOUD_RUN_BASE_URL = (os.getenv("CLOUD_RUN_BASE_URL") or "").rstrip("/")
WORKER_SHARED_TOKEN = os.getenv("WORKER_SHARED_TOKEN", "")

SOURCE_TYPE_TO_WORKER_PATH = {
    "upload": "/knowledge/upload/_worker/run",
    "public_url": "/knowledge/url/_worker/run",
    "opendata": "/knowledge/opendata/_worker/run",
    "kokkai": "/knowledge/kokkai/_worker/run",
}


def list_all_queues() -> list[dict]:
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID not found")

    client = tasks_v2.CloudTasksClient()
    parent = client.common_location_path(PROJECT_ID, LOCATION)

    queues = client.list_queues(parent=parent)

    results = []
    for queue in queues:
        queue_name = getattr(queue, "name", "") or ""
        queue_id = queue_name.split("/")[-1] if queue_name else ""

        stats = getattr(queue, "stats", None)

        tasks_count = int(getattr(stats, "tasks_count", 0) or 0)
        executed_last_minute_count = int(getattr(stats, "executed_last_minute_count", 0) or 0)
        concurrent_dispatches_count = int(getattr(stats, "concurrent_dispatches_count", 0) or 0)
        effective_execution_rate = float(getattr(stats, "effective_execution_rate", 0.0) or 0.0)

        rate_limits = getattr(queue, "rate_limits", None)
        max_dispatches_per_second = float(getattr(rate_limits, "max_dispatches_per_second", 0.0) or 0.0)
        max_concurrent_dispatches = int(getattr(rate_limits, "max_concurrent_dispatches", 0) or 0)

        results.append({
            "queue_id": queue_id,
            "queue_name": queue_name,
            "tasks_count": tasks_count,
            "executed_last_minute_count": executed_last_minute_count,
            "concurrent_dispatches_count": concurrent_dispatches_count,
            "effective_execution_rate": effective_execution_rate,
            "max_dispatches_per_second": max_dispatches_per_second,
            "max_concurrent_dispatches": max_concurrent_dispatches,
        })

    results.sort(key=lambda x: x["queue_id"])
    return results


def select_best_queue() -> dict:
    queues = list_all_queues()

    if not queues:
        raise RuntimeError("No Cloud Tasks queues found")

    best = min(
        queues,
        key=lambda q: (
            q["tasks_count"],
            q["concurrent_dispatches_count"],
            q["queue_id"],
        )
    )
    return best


def get_queue_path(queue_id: str) -> str:
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID not found")

    client = tasks_v2.CloudTasksClient()
    return client.queue_path(PROJECT_ID, LOCATION, queue_id)


def build_worker_url(worker_path: str) -> str:
    if not CLOUD_RUN_BASE_URL:
        raise RuntimeError("CLOUD_RUN_BASE_URL not found")

    if not worker_path:
        raise RuntimeError("worker_path is empty")

    if not worker_path.startswith("/"):
        worker_path = "/" + worker_path

    return f"{CLOUD_RUN_BASE_URL}{worker_path}"


def get_worker_path_by_source_type(source_type: str) -> str:
    source_type_norm = (source_type or "").strip().lower()
    worker_path = SOURCE_TYPE_TO_WORKER_PATH.get(source_type_norm)
    if not worker_path:
        raise RuntimeError(f"Unsupported source_type: {source_type}")
    return worker_path


def build_knowledge_task_payload(source_type: str, uid: str, job_id: str) -> dict[str, Any]:
    if not source_type:
        raise RuntimeError("source_type is empty")
    if not uid:
        raise RuntimeError("uid is empty")
    if not job_id:
        raise RuntimeError("job_id is empty")

    return {
        "source_type": source_type,
        "uid": uid,
        "job_id": job_id,
    }


def build_task_id(prefix: str, payload: dict[str, Any]) -> str:
    base_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(base_text.encode("utf-8")).hexdigest()[:24]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{timestamp}-{digest}"


def build_schedule_time_proto(schedule_seconds: int):
    if not schedule_seconds or schedule_seconds <= 0:
        return None

    dt = datetime.now(timezone.utc) + timedelta(seconds=int(schedule_seconds))
    ts = dt.isoformat().replace("+00:00", "Z")
    return ts


def create_http_task(
    queue_id: str,
    url: str,
    payload: dict[str, Any],
    headers: Optional[dict[str, str]] = None,
    task_id: Optional[str] = None,
    schedule_seconds: int = 0,
) -> dict[str, Any]:
    if not queue_id:
        raise RuntimeError("queue_id is empty")
    if not url:
        raise RuntimeError("url is empty")
    if not isinstance(payload, dict):
        raise RuntimeError("payload must be dict")

    client = tasks_v2.CloudTasksClient()
    parent = get_queue_path(queue_id)

    body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request_headers = {
        "Content-Type": "application/json; charset=utf-8",
    }
    if headers:
        request_headers.update(headers)

    task: dict[str, Any] = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": url,
            "headers": request_headers,
            "body": body_bytes,
        },
        "dispatch_deadline": duration_pb2.Duration(seconds=1800),  # ← 修正
    }

    if task_id:
        task_name = f"{parent}/tasks/{task_id}"
        task["name"] = task_name

    schedule_time = build_schedule_time_proto(schedule_seconds)
    if schedule_time:
        task["schedule_time"] = schedule_time

    created = client.create_task(parent=parent, task=task)

    created_name = getattr(created, "name", "") or ""
    created_queue_id = ""
    created_task_id = ""
    if created_name:
        parts = created_name.split("/")
        if len(parts) >= 2:
            created_queue_id = parts[-3] if len(parts) >= 3 else ""
            created_task_id = parts[-1]

    return {
        "queue_id": queue_id,
        "queue_path": parent,
        "task_name": created_name,
        "task_id": created_task_id,
        "created_queue_id": created_queue_id,
        "url": url,
        "payload": payload,
    }


def enqueue_http_task_to_best_queue(
    url: str,
    payload: dict[str, Any],
    headers: Optional[dict[str, str]] = None,
    task_prefix: str = "task",
    schedule_seconds: int = 0,
) -> dict[str, Any]:
    best_queue = select_best_queue()
    queue_id = best_queue["queue_id"]

    task_id = build_task_id(task_prefix, payload)

    created = create_http_task(
        queue_id=queue_id,
        url=url,
        payload=payload,
        headers=headers,
        task_id=task_id,
        schedule_seconds=schedule_seconds,
    )

    return {
        "selected_queue": best_queue,
        "created_task": created,
    }


def enqueue_knowledge_job(
    source_type: str,
    uid: str,
    job_id: str,
    schedule_seconds: int = 0,
) -> dict[str, Any]:
    worker_path = get_worker_path_by_source_type(source_type)
    worker_url = build_worker_url(worker_path)
    payload = build_knowledge_task_payload(source_type=source_type, uid=uid, job_id=job_id)

    headers: dict[str, str] = {}
    if WORKER_SHARED_TOKEN:
        headers["x-worker-token"] = WORKER_SHARED_TOKEN

    result = enqueue_http_task_to_best_queue(
        url=worker_url,
        payload=payload,
        headers=headers,
        task_prefix=f"knowledge-{source_type}",
        schedule_seconds=schedule_seconds,
    )

    return {
        "source_type": source_type,
        "uid": uid,
        "job_id": job_id,
        "worker_path": worker_path,
        "worker_url": worker_url,
        "queue_id": result["selected_queue"]["queue_id"],
        "queue_name": result["selected_queue"]["queue_name"],
        "task_name": result["created_task"]["task_name"],
        "task_id": result["created_task"]["task_id"],
        "selected_queue": result["selected_queue"],
        "created_task": result["created_task"],
    }
