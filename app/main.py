from fastapi import FastAPI
from app.core.cors import setup_cors
from app.routers.health import router as health_router
from app.routers.session import router as session_router
from app.routers.ingest import router as ingest_router
from app.routers.search import router as search_router

def create_app() -> FastAPI:
    app = FastAPI()
    setup_cors(app)

    app.include_router(health_router)
    app.include_router(session_router, prefix="/v1")
    app.include_router(ingest_router, prefix="/v1")
    app.include_router(search_router, prefix="/v1")
    return app

app = create_app()
