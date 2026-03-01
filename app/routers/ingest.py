# app/routers/ingest.py

from fastapi import APIRouter

from routers.ingest.kokkai_test import router as kokkai_router

router = APIRouter(prefix="/ingest", tags=["ingest"])

router.include_router(kokkai_router)
