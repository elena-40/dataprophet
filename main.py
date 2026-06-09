from fastapi import FastAPI

from schemas import CustomerFeatures, PredictionResponse

app = FastAPI(
    title="DataProphet API",
    description="API de prédiction de l'année de plantation d'un arbre",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Le service DataProphet est actif"}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    # ⚠️ Prédiction FACTICE pour l'instant — le vrai modèle arrive à l'étape 1.5
    annee_factice = 1990.0
    return PredictionResponse(
        annee_plantation_estimee=annee_factice,
        message=f"(factice) Arbre probablement planté vers {int(annee_factice)}",
    )
