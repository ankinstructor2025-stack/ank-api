from fastapi import FastAPI
from app.core.cors import setup_cors
from app.routers.health import router as health_router
from app.routers.session import router as session_router
from app.routers.search import router as search_router
from app.routers.user_init import router as user_init_router
from app.routers.kokkai import router as kokkai_router
from app.routers.opendata import router as opendata_router
from app.routers.public_url import router as public_url_router
from app.routers.upload_register import router as upload_register_router
from app.routers.upload_ingest import router as upload_ingest_router
from app.routers.upload_retrieve import router as upload_retrieve_router
from app.routers.knowledge_generate_kokkai import router as knowledge_generate_kokkai_router
from app.routers.knowledge_generate_opendata import router as knowledge_generate_opendata_router
from app.routers.knowledge_generate_upload import router as knowledge_generate_upload_router
from app.routers.knowledge_refine import router as knowledge_refine_router
from app.routers.knowledge_search import router as knowledge_search_router

def create_app() -> FastAPI:
    app = FastAPI()
    setup_cors(app)

    app.include_router(health_router)
    app.include_router(session_router, prefix="/v1")
    app.include_router(search_router, prefix="/v1")
    app.include_router(user_init_router, prefix="/v1")
    app.include_router(kokkai_router, prefix="/v1")
    app.include_router(opendata_router, prefix="/v1")
    app.include_router(public_url_router, prefix="/v1")
    app.include_router(upload_register_router, prefix="/v1")
    app.include_router(upload_ingest_router, prefix="/v1")
    app.include_router(upload_retrieve_router, prefix="/v1")
    app.include_router(knowledge_generate_kokkai_router, prefix="/v1")
    app.include_router(knowledge_generate_opendata_router, prefix="/v1")
    app.include_router(knowledge_generate_upload_router, prefix="/v1")
    app.include_router(knowledge_refine_router, prefix="/v1")
    app.include_router(knowledge_search_router, prefix="/v1")
    return app

app = create_app()
