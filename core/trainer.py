# core/trainer.py
def train_model(model_type: str, dataset_path: str, hyperparameters: dict):
    trainer = AutoTrainer(model_type, dataset_path, hyperparameters)
    return trainer.train()
