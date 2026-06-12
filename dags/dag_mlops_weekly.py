from datetime import datetime

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

with DAG(
    dag_id="dag_mlops_weekly",
    start_date=datetime(2026, 1, 1),
    schedule="0 6 * * 1",
    catchup=False,
    tags=["dataprophet"],
) as dag:
    trigger_preprocessing = TriggerDagRunOperator(
        task_id="trigger_preprocessing",
        trigger_dag_id="dag_preprocessing",
        wait_for_completion=True,
        poke_interval=10,
        reset_dag_run=True,
        deferrable=True,
    )
    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="dag_retrain",
        wait_for_completion=True,
        poke_interval=10,
        reset_dag_run=True,
        deferrable=True,
    )
    trigger_deploy = TriggerDagRunOperator(
        task_id="trigger_deploy",
        trigger_dag_id="dag_deploy",
        wait_for_completion=True,
        poke_interval=10,
        reset_dag_run=True,
        deferrable=True,
    )

    trigger_preprocessing >> trigger_retrain >> trigger_deploy
