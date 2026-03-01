from fastapi import FastAPI
from app.core.cors import setup_cors
from app.routers.health import router as health_router
from app.routers.session import router as session_router
from app.routers.search import router as search_router
from app.routers.user_init import router as user_init_router
from app.routers.kokkai_test import router as kokkai_router

def create_app() -> FastAPI:
    app = FastAPI()
    setup_cors(app)

    app.include_router(health_router)
    app.include_router(session_router, prefix="/v1")
    app.include_router(search_router, prefix="/v1")
    app.include_router(user_init_router, prefix="/v1")
    app.include_router(kokkai_router, prefix="/v1")
    return app

app = create_app()
