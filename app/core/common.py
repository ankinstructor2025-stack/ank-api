import hashlib
import os

from google.cloud import tasks_v2


PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("TASKS_LOCATION", "asia-northeast1")


def list_all_queues() -> list[dict]:
    """
    Cloud Tasks の指定ロケーションに存在するキューを全件取得する。
    返却は扱いやすい dict の配列にしている。
    """
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
    """
    全キューの中から、もっとも待ちが少ないとみなせるキューを返す。
    まずは tasks_count 優先、同点なら実行中件数が少ないものを優先。
    """
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
