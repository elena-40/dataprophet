import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from schemas import CustomerFeatures, PredictionResponse, HelpData
from metrics import PREDICTIONS_TOTAL, PREDICTION_DURATION

mlflow.set_tracking_uri("http://127.0.0.1:5000")

ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["model"] = mlflow.sklearn.load_model("models:/DataProphet@production")
    print("Modele charge depuis le Registry MLflow")
    yield
    ml.clear()


app = FastAPI(title="DataProphet API", version="3.0.0", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Le service DataProphet est actif"}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    start = time.time()
    donnees = pd.DataFrame([features.model_dump()])
    annee = float(ml["model"].predict(donnees)[0])
    PREDICTION_DURATION.observe(time.time() - start)
    if annee < 1980:
        tranche = "avant_1980"
    elif annee < 2000:
        tranche = "1980_2000"
    else:
        tranche = "apres_2000"
    PREDICTIONS_TOTAL.labels(tranche=tranche).inc()
    return PredictionResponse(
        annee_plantation_estimee=round(annee, 1),
        message=f"Arbre probablement plante vers {int(annee)}",
    )


@app.post("/api/helpdata")
def helpdata(data: HelpData):
    Path("help_data").mkdir(exist_ok=True)
    nom = f"help_data/feedback_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    with open(nom, "w", encoding="utf-8") as f:
        json.dump(data.model_dump(), f, ensure_ascii=False, indent=2)
    return {"status": "ok", "message": "Feedback enregistre", "fichier": nom}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
