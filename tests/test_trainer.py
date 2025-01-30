# tests/test_trainer.py
from core.trainer import train_model

def test_train_model():
    result = train_model("Neural Network", "data.csv", {"lr": 0.01})
    assert "Training Neural Network" in result
