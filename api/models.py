# models.py
from pydantic import BaseModel

class TrainRequest(BaseModel):
    model_type: str
    hyperparameters: dict
    dataset_path: str
