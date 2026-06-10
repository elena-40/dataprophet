import mlflow
from mlflow.tracking import MlflowClient

# Se connecter au serveur MLflow (Docker)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

MODEL_NAME = "DataProphet"

# 1. Lister toutes les versions du modèle + leur R²
print(f"Versions de '{MODEL_NAME}' :")
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
for v in versions:
    run = client.get_run(v.run_id)
    r2 = run.data.metrics.get("r2")
    print(f"   - version {v.version}  |  r2 = {r2}")

# 2. Trouver la MEILLEURE version (le plus haut R²)
def get_r2(v):
    return client.get_run(v.run_id).data.metrics.get("r2", float("-inf"))

best = max(versions, key=get_r2)
print(f"\nMeilleure version : v{best.version} (r2 = {get_r2(best):.3f})")

# 3. Poser l'alias 'production' sur cette version (= promotion en prod)
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="production",
    version=best.version,
)
print(f"✅ Alias 'production' posé sur la version {best.version}")
