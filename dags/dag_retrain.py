import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_DIR = Path(__file__).resolve().parent.parent


def run_retrain(**context):
    resultat = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "retrain.py")],
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
    )
    print("=== SORTIE retrain.py ===")
    print(resultat.stdout)
    if resultat.stderr:
        print("=== ERREURS ===")
        print(resultat.stderr)
    if resultat.returncode != 0:
        raise RuntimeError(f"retrain.py a echoue (code {resultat.returncode})")
    print("Reentrainement termine avec succes")


with DAG(
    dag_id="dag_retrain",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["dataprophet"],
) as dag:
    retrain_task = PythonOperator(
        task_id="run_retrain",
        python_callable=run_retrain,
    )
