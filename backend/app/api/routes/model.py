from fastapi import APIRouter
import os

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")

import os
from datetime import datetime

@router.get("/model-status")
def model_status():
    if os.path.exists(MODEL_PATH):
        modified_time = os.path.getmtime(MODEL_PATH)

        return {
            "path":MODEL_PATH,
            "model_exists": True,
            "last_trained": datetime.fromtimestamp(modified_time)
        }

    return {"model_exists": False, "path":MODEL_PATH,}