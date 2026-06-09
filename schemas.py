from pydantic import BaseModel, Field


# --- Données d'ENTRÉE : les 7 features de l'arbre ---
class CustomerFeatures(BaseModel):
    latitude: float = Field(..., ge=45.0, le=46.0, description="Latitude de l'arbre")
    longitude: float = Field(..., ge=5.0, le=6.0, description="Longitude de l'arbre")
    genre_bota: str = Field(..., description="Genre botanique, ex: Tilia")
    espece: str = Field(..., description="Espèce, ex: henryana")
    stadededeveloppement: str = Field(..., description="Arbre jeune / adulte / vieillissant")
    hauteurarbre: str = Field(..., description="Moins de 10 m / de 10 m à 20 m / Plus de 20 m")
    typenature: str = Field(..., description="Libre, Semi-libre, Architecturé...")

    # Exemple pré-rempli qui apparaîtra dans la Swagger UI
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "latitude": 45.167,
                    "longitude": 5.741,
                    "genre_bota": "Tilia",
                    "espece": "henryana",
                    "stadededeveloppement": "Arbre jeune",
                    "hauteurarbre": "Moins de 10 m",
                    "typenature": "Libre",
                }
            ]
        }
    }


# --- Données de SORTIE : la prédiction ---
class PredictionResponse(BaseModel):
    annee_plantation_estimee: float = Field(..., description="Année de plantation prédite")
    message: str = Field(..., description="Interprétation lisible de la prédiction")
