from fastapi import FastAPI
from app.core.cors import setup_cors
from app.routers.health import router as health_router
from app.routers.session import router as session_router
from app.routers.search import router as search_router
from app.routers.user_init import router as user_init_router
from app.routers.kokkai import router as kokkai_router
from app.routers.opendata import router as opendata_router
from app.routers.egov import router as egov_router
from app.routers.caa import router as caa_router
from app.routers.upload_and_register import router as upload_and_register
from app.routers.row_data_ingest import router as row_data_ingest
from app.routers.row_data import router as row_data_router

def create_app() -> FastAPI:
    app = FastAPI()
    setup_cors(app)

    app.include_router(health_router)
    app.include_router(session_router, prefix="/v1")
    app.include_router(search_router, prefix="/v1")
    app.include_router(user_init_router, prefix="/v1")
    app.include_router(kokkai_router, prefix="/v1")
    app.include_router(opendata_router, prefix="/v1")
    app.include_router(egov_router, prefix="/v1")
    app.include_router(caa_router, prefix="/v1")
    app.include_router(upload_and_register, prefix="/v1")
    app.include_router(row_data_ingest, prefix="/v1")
    app.include_router(row_data_router, prefix="/v1")
    return app

app = create_app()
