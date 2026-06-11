import time
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from schemas import CustomerFeatures, PredictionResponse
from metrics import PREDICTIONS_TOTAL, PREDICTION_DURATION

mlflow.set_tracking_uri("http://127.0.0.1:5000")

ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["model"] = mlflow.sklearn.load_model("models:/DataProphet@production")
    print("✅ Modèle chargé depuis le Registry MLflow (DataProphet@production)")
    yield
    ml.clear()
    print("👋 Modèle déchargé")


app = FastAPI(
    title="DataProphet API",
    description="API de prédiction de l'année de plantation d'un arbre",
    version="3.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Le service DataProphet est actif"}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    start = time.time()
    donnees = pd.DataFrame([features.model_dump()])
    annee = float(ml["model"].predict(donnees)[0])
    PREDICTION_DURATION.observe(time.time() - start)   # ⏱️ durée

    # classer la prédiction dans une tranche (pour le Counter)
    if annee < 1980:
        tranche = "avant_1980"
    elif annee < 2000:
        tranche = "1980_2000"
    else:
        tranche = "apres_2000"
    PREDICTIONS_TOTAL.labels(tranche=tranche).inc()    # 🔢 +1

    return PredictionResponse(
        annee_plantation_estimee=round(annee, 1),
        message=f"Arbre probablement planté vers {int(annee)}",
    )


@app.get("/metrics")
def metrics():
    # expose les métriques au format que Prometheus comprend
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
