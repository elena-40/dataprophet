import glob
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("plantation-prediction-retrain")

MODEL_NAME = "DataProphet"
SEUIL_R2 = 0.65          # seuil de promotion en staging (à toi de le définir)

client = MlflowClient()

# 1. Données de base
df = pd.read_pickle("../Modele export (ml1)/data/data.pkl")

# 2. AJOUT : données de feedback si présentes dans help_data/ (vide pour l'instant)
feedback = glob.glob("help_data/*.csv")
if feedback:
    extra = pd.concat([pd.read_csv(f) for f in feedback], ignore_index=True)
    df = pd.concat([df, extra], ignore_index=True)
    print(f"{len(feedback)} fichier(s) de feedback ajouté(s)")
else:
    print("Aucune donnée de feedback (help_data/ vide) — normal pour l'instant")

target = "anneedeplantation"
df = df.dropna(subset=[target])
X = df.drop(columns=[target])
y = df[target]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Pipeline (mêmes hyperparamètres que le bon modèle)
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])
pipeline = Pipeline([("preprocessor", preprocessor),
                     ("model", RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42))])

# 4. Entraîner + loguer dans MLflow
with mlflow.start_run():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 15)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(pipeline, name="model", registered_model_name=MODEL_NAME)
    print(f"Nouveau modèle réentraîné -> R2={r2:.3f}")

# 5. AJOUT : comparer au modèle actuellement en production
try:
    prod = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")
    prod_r2 = r2_score(y_test, prod.predict(X_test))
    print(f"R2 production = {prod_r2:.3f}  |  R2 nouveau = {r2:.3f}")
except Exception as e:
    print("Pas de modèle en production pour comparer :", e)

# 6. AJOUT : promotion conditionnelle en staging
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest = max(versions, key=lambda v: int(v.version))
if r2 >= SEUIL_R2:
    client.set_registered_model_alias(MODEL_NAME, "staging", latest.version)
    print(f"✅ R2={r2:.3f} >= {SEUIL_R2} -> version {latest.version} promue en STAGING")
else:
    print(f"❌ R2={r2:.3f} < {SEUIL_R2} -> pas de promotion")
