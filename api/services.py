# services.py
def train_model(request):
    return f"Training {request.model_type} with hyperparameters {request.hyperparameters} using dataset {request.dataset_path}"
