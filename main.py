from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SessionRequest(BaseModel):
    user_id: str

@app.get("/")
def root():
    return {"message": "ank-api is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/session")
def create_session(req: SessionRequest):
    return {
        "user_id": req.user_id,
        "status": "session ok"
    }
