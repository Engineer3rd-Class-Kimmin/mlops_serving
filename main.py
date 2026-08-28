from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import os
import time
import pymysql

app = FastAPI()

model = joblib.load("model.joblib")


class PredictRequest(BaseModel):
    data: List[float]


@app.post("/predict")
def predict(request: PredictRequest):
    prediction = model.predict([request.data])
    return {"class_index": int(prediction[0])}


@app.get("/health")
def health():
    db_host = os.getenv("DB_HOST", "db")
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "rootpassword")
    db_name = os.getenv("DB_NAME", "mlops")

    try:
        conn = pymysql.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name,
            connect_timeout=3,
        )
        conn.close()

        return {
            "status": "ok",
            "db": "mysql"
        }

    except Exception as e:
        return {
            "status": "error",
            "db": "mysql",
            "detail": str(e)
        }