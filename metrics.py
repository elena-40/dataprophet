from prometheus_client import Counter, Histogram

# Counter : nombre total de prédictions, avec un label "tranche"
# (le kit utilise "churn"/"stable" ; toi tu fais de la régression,
#  donc on classe par tranche d'année prédite)
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Nombre total de predictions effectuees",
    ["tranche"],
)

# Histogram : durée de chaque prédiction (en secondes)
PREDICTION_DURATION = Histogram(
    "prediction_duration_seconds",
    "Duree d'une prediction en secondes",
)
