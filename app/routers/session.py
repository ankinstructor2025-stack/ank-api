from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class SessionRequest(BaseModel):
    user_id: str

@router.post("/session")
def create_session():
    return {"status": "session ok"}
