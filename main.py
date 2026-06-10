from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI

from schemas import CustomerFeatures, PredictionResponse

# On pointe vers le serveur MLflow (Docker)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # AU DÉMARRAGE : charger le modèle 'production' depuis le Registry
    ml["model"] = mlflow.sklearn.load_model("models:/DataProphet@production")
    print("✅ Modèle chargé depuis le Registry MLflow (DataProphet@production)")
    yield
    ml.clear()
    print("👋 Modèle déchargé")


app = FastAPI(
    title="DataProphet API",
    description="API de prédiction de l'année de plantation d'un arbre",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Le service DataProphet est actif"}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    donnees = pd.DataFrame([features.model_dump()])
    prediction = ml["model"].predict(donnees)
    annee = float(prediction[0])
    return PredictionResponse(
        annee_plantation_estimee=round(annee, 1),
        message=f"Arbre probablement planté vers {int(annee)}",
    )
