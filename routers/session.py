# app/routers/session.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class SessionRequest(BaseModel):
    user_id: str

@router.post("/session")
def create_session(req: SessionRequest):
    return {"user_id": req.user_id, "status": "session ok"}
