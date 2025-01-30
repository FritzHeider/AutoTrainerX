# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to AutoTrainerX API"}

def test_train():
    response = client.post("/train", json={
        "model_type": "Neural Network",
        "hyperparameters": {"lr": 0.01},
        "dataset_path": "data.csv"
    })
    assert response.status_code == 200
    assert "success" in response.json()["status"]