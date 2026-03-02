# app/routers/opendata_test.py
from fastapi import APIRouter, HTTPException, Query
import requests

router = APIRouter(
    prefix="/opendata",
    tags=["opendata"]
)

CKAN_BASE = "https://data.e-gov.go.jp/data/api/action"  # e-Govデータポータル CKAN API base

def _ckan_get(action: str, params: dict | None = None) -> dict:
    """
    CKAN action API を GET で呼び出して result を返す
    失敗時は HTTPException(502) にする（添付と同じ思想）
    """
    url = f"{CKAN_BASE}/{action}"
    try:
        res = requests.get(url, params=params or {}, timeout=20)
        res.raise_for_status()
        payload = res.json()

        # CKAN形式: {"success": true/false, "result": ...}
        if not payload.get("success", False):
            raise HTTPException(status_code=502, detail=f"ckan api error: success=false ({action})")

        return payload.get("result")

    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"ckan api error: {str(e)}")


# ----------------------------------------
# 取得テスト（まずはここだけ）
# ----------------------------------------
@router.get("/test")
def opendata_test():
    """
    e-Govデータポータル（CKAN）接続テスト
    - データセットIDを少数だけ取得して返す
    """
    result = _ckan_get("package_list", params={"limit": 5})
    return {
        "count": len(result),
        "dataset_ids": result
    }


@router.get("/datasets/search")
def search_datasets(
    q: str = Query(..., description="検索キーワード（CKAN package_search の q）"),
    rows: int = Query(10, ge=1, le=100, description="取得件数"),
    start: int = Query(0, ge=0, description="開始オフセット"),
):
    """
    データセット検索（package_search）
    - 返すのは「表示に使う最小限」を想定（id, title, notes, organization, resources）
    """
    result = _ckan_get("package_search", params={"q": q, "rows": rows, "start": start})

    items = []
    for ds in result.get("results", []):
        items.append({
            "id": ds.get("id") or ds.get("name"),
            "name": ds.get("name"),
            "title": ds.get("title"),
            "notes": ds.get("notes"),
            "organization": (ds.get("organization") or {}).get("title"),
            "num_resources": len(ds.get("resources", []) or []),
        })

    return {
        "count": result.get("count", 0),
        "rows": rows,
        "start": start,
        "items": items
    }


@router.get("/datasets/recent")
def recent_datasets(
    limit: int = Query(10, ge=1, le=100, description="取得件数"),
    offset: int = Query(0, ge=0, description="開始オフセット"),
):
    """
    最近変更されたデータセット（リソース込み）を取得
    - current_package_list_with_resources
    """
    result = _ckan_get("current_package_list_with_resources", params={"limit": limit, "offset": offset})

    items = []
    for ds in result or []:
        items.append({
            "id": ds.get("id") or ds.get("name"),
            "name": ds.get("name"),
            "title": ds.get("title"),
            "organization": (ds.get("organization") or {}).get("title"),
            "num_resources": len(ds.get("resources", []) or []),
        })

    return {
        "count": len(items),
        "limit": limit,
        "offset": offset,
        "items": items
    }


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    """
    データセット詳細（package_show）
    - resources の URL などもここで取れる
    """
    ds = _ckan_get("package_show", params={"id": dataset_id})

    resources = []
    for r in ds.get("resources", []) or []:
        resources.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "format": r.get("format"),
            "mimetype": r.get("mimetype"),
            "url": r.get("url"),
            "last_modified": r.get("last_modified"),
        })

    return {
        "id": ds.get("id") or ds.get("name"),
        "name": ds.get("name"),
        "title": ds.get("title"),
        "notes": ds.get("notes"),
        "organization": (ds.get("organization") or {}).get("title"),
        "license_id": ds.get("license_id"),
        "metadata_created": ds.get("metadata_created"),
        "metadata_modified": ds.get("metadata_modified"),
        "resources": resources,
    }


@router.get("/resources/{resource_id}")
def get_resource(resource_id: str):
    """
    リソース詳細（resource_show）
    - 実ファイルのURLも返る
    """
    r = _ckan_get("resource_show", params={"id": resource_id})
    return {
        "id": r.get("id"),
        "name": r.get("name"),
        "format": r.get("format"),
        "mimetype": r.get("mimetype"),
        "url": r.get("url"),
        "last_modified": r.get("last_modified"),
        "created": r.get("created"),
    }
