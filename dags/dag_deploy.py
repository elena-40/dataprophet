from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "DataProphet"
SEUIL = 0.01  # le nouveau R2 doit depasser l'actuel d'au moins 0.01


def evaluate_metrics(**context):
    client = MlflowClient(tracking_uri=TRACKING_URI)
    try:
        staging = client.get_model_version_by_alias(MODEL_NAME, "staging")
        prod = client.get_model_version_by_alias(MODEL_NAME, "production")
    except Exception as e:
        print(f"Comparaison impossible (alias manquant ?): {e}")
        return "skip_promotion"

    r2_staging = client.get_run(staging.run_id).data.metrics.get("r2", float("-inf"))
    r2_prod = client.get_run(prod.run_id).data.metrics.get("r2", float("-inf"))
    print(f"R2 staging (v{staging.version}) = {r2_staging}")
    print(f"R2 production (v{prod.version}) = {r2_prod}")

    if r2_staging > r2_prod + SEUIL:
        print("-> Nouveau modele MEILLEUR : promotion")
        return "promote_to_production"
    print("-> Nouveau modele PAS meilleur : on garde la production")
    return "skip_promotion"


def promote_to_production(**context):
    client = MlflowClient(tracking_uri=TRACKING_URI)
    staging = client.get_model_version_by_alias(MODEL_NAME, "staging")
    client.set_registered_model_alias(MODEL_NAME, "production", staging.version)
    print(f"Version {staging.version} promue en PRODUCTION")


def skip_promotion(**context):
    print("Pas de promotion : le modele en production reste inchange")


with DAG(
    dag_id="dag_deploy",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["dataprophet"],
) as dag:
    evaluate = BranchPythonOperator(
        task_id="evaluate_metrics",
        python_callable=evaluate_metrics,
    )
    promote = PythonOperator(
        task_id="promote_to_production",
        python_callable=promote_to_production,
    )
    skip = PythonOperator(
        task_id="skip_promotion",
        python_callable=skip_promotion,
    )
    evaluate >> [promote, skip]
