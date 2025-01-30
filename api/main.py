from fastapi import FastAPI, APIRouter
from api.config import settings
from api.models import TrainRequest
from api.services import train_model

app = FastAPI(title="AutoTrainerX API", version="1.0")
router = APIRouter()

@router.get("/")
def root():
    return {"message": "Welcome to AutoTrainerX API"}

@router.post("/train")
def train(request: TrainRequest):
    result = train_model(request)
    return {"status": "success", "details": result}

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)