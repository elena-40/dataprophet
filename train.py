import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# CONFIG MLflow : on pointe vers le serveur lancé dans le terminal 1
# ============================================================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("plantation-prediction")

# ============================================================
# 1. CHARGER LES DONNÉES
# ============================================================
df = pd.read_pickle("../Modele export (ml1)/data/data.pkl")

target = "anneedeplantation"
df = df.dropna(subset=[target])         # on enlève les lignes sans cible
X = df.drop(columns=[target])           # les features
y = df[target]                          # la cible (année de plantation)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 2. HYPERPARAMÈTRES  (👈 ce sont EUX qu'on fera varier entre 2 runs)
# ============================================================
n_estimators = 50
max_depth = 15

# ============================================================
# 3. LE PIPELINE (préprocessing + modèle), comme dans le notebook
# ============================================================
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )),
])

# ============================================================
# 4. ENTRAÎNEMENT INSTRUMENTÉ AVEC MLflow
# ============================================================
with mlflow.start_run():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # --- Métriques de RÉGRESSION (pas accuracy/f1 : ça c'est la classif) ---
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # --- On enregistre les PARAMÈTRES ---
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # --- On enregistre les MÉTRIQUES ---
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)

    # --- On enregistre LE MODÈLE et on l'inscrit au Registry "DataProphet" ---
    mlflow.sklearn.log_model(
        pipeline,
        name="model",
        registered_model_name="DataProphet",
    )

    print(f"Run terminé -> R2={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
