import glob
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_DIR = Path(__file__).resolve().parent.parent
HELP_DATA = PROJECT_DIR / "help_data"
DATA_DIR = PROJECT_DIR / "data"

FEATURES = ["latitude", "longitude", "genre_bota", "espece",
            "stadededeveloppement", "hauteurarbre", "typenature"]


def load_and_validate(**context):
    fichiers = glob.glob(str(HELP_DATA / "*.json"))
    if not fichiers:
        print("help_data/ est vide - rien a traiter (normal au debut)")
        return []
    valides = []
    for f in fichiers:
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        if all(k in d for k in FEATURES) and "annee_reelle" in d:
            valides.append(d)
        else:
            print(f"Fichier rejete (champs manquants): {f}")
    print(f"{len(valides)} enregistrement(s) valides")
    return valides

def prepare_dataset(**context):
    valides = context["ti"].xcom_pull(task_ids="load_and_validate")
    if not valides:
        print("Aucune donnee validee - pas de dataset a preparer")
        return
    df = pd.DataFrame(valides)
    df = df.rename(columns={"annee_reelle": "anneedeplantation"})
    DATA_DIR.mkdir(exist_ok=True)
    sortie = DATA_DIR / "retrain_dataset.csv"
    df.to_csv(sortie, index=False)
    print(f"Dataset prepare: {sortie} ({len(df)} lignes)")


with DAG(
    dag_id="dag_preprocessing",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["dataprophet"],
) as dag:
    t1 = PythonOperator(task_id="load_and_validate", python_callable=load_and_validate)
    t2 = PythonOperator(task_id="prepare_dataset", python_callable=prepare_dataset)
    t1 >> t2
