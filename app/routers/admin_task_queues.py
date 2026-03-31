from fastapi import APIRouter, HTTPException
from google.cloud import tasks_v2
import os

router = APIRouter(prefix="/admin/task-queues", tags=["admin-task-queues"])

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "asia-northeast1")


def get_client():
    return tasks_v2.CloudTasksClient()


def get_parent():
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID is not set")
    return f"projects/{PROJECT_ID}/locations/{LOCATION}"


def parse_queue_short_name(full_name: str) -> str:
    return full_name.split("/")[-1] if full_name else ""


def parse_task_short_name(full_name: str) -> str:
    return full_name.split("/")[-1] if full_name else ""


# =========================
# キュー一覧
# =========================
@router.get("")
def list_queues():
    try:
        client = get_client()
        parent = get_parent()

        queues = client.list_queues(parent=parent)

        result = []
        for q in queues:
            queue_name = q.name
            short_name = parse_queue_short_name(queue_name)

            # タスク数は別途取得（軽く1回だけ）
            task_count = 0
            try:
                tasks = client.list_tasks(parent=queue_name)
                task_count = sum(1 for _ in tasks)
            except Exception:
                task_count = -1  # 取得失敗

            result.append({
                "name": queue_name,
                "queue_name": short_name,
                "state": str(q.state).replace("State.", ""),
                "task_count": task_count
            })

        return {"queues": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"list_queues failed: {str(e)}")


# =========================
# タスク一覧
# =========================
@router.get("/{queue_name}/tasks")
def list_tasks(queue_name: str):
    try:
        client = get_client()
        parent = f"{get_parent()}/queues/{queue_name}"

        tasks = client.list_tasks(parent=parent)

        result = []
        for t in tasks:
            http_request = t.http_request

            url = ""
            if http_request:
                url = http_request.url

            result.append({
                "name": t.name,
                "short_name": parse_task_short_name(t.name),
                "schedule_time": t.schedule_time.isoformat() if t.schedule_time else "",
                "dispatch_count": t.dispatch_count,
                "response_count": t.response_count,
                "url": url
            })

        return {"tasks": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"list_tasks failed: {str(e)}")
