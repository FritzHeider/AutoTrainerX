# utils/helper.py
def parse_hyperparameters(hyperparameters: str):
    """Parse hyperparameters from JSON string."""
    import json
    try:
        return json.loads(hyperparameters)
    except json.JSONDecodeError:
        return {}
