import logging
from airflow import DAG
from airflow.utils.db import provide_session
from airflow.operators.python_operator import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
from airflow.models import XCom
from airflow.utils.trigger_rule import TriggerRule
from scripts.preprocessing import (
    load_data,
    filter_records_by_word_count_and_date,
    filter_records_by_language,
    aggregate_filtered_task,
    data_cleaning,
    remove_abusive_data,
)

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# Define the function to clear XComs
@provide_session
def clear_xcom(context, session=None):
    dag_id = context["ti"]["dag"]
    execution_date = context["ti"]["execution_date"]
    session.query(XCom).filter(
        XCom.dag_id == dag_id, XCom.execution_date == execution_date
    ).delete()


MIN_WORD: int = 5

# Data Validation DAG
with DAG(
    "Data_validation_pipeline",
    default_args=default_args,
    description="DAG for Data Validation",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 10, 17),
    catchup=False,
) as dag:

    # Task 1: Load the data
    data_loading_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )

    # Task 2, 3, & 4: Parallel Data Processing
    # Task 2: Remove records which have words less than min count
    # Task 3: Remove records outside of range
    # Task 4: Remove un-recognised language
   
    filter_parallel_tasks = [
        PythonOperator(
            task_id="remove_records_with_minimum_words_and_outdated_records",
            python_callable=filter_records_by_word_count_and_date,
            op_args=[data_loading_task.output, MIN_WORD],
        ),
        PythonOperator(
            task_id="detect_language",
            python_callable=filter_records_by_language,
            op_args=[data_loading_task.output],
        ),
    ]

    # Task 4: Aggregate results from Task 2 & Task 3
    aggregate_parallel_tasks = PythonOperator(
        task_id="validation_aggregation",
        python_callable=aggregate_filtered_task,
        op_args=[filter_parallel_tasks[0].output, filter_parallel_tasks[1].output],
        provide_context=True,
    )
    # Task 5: Trigger Data Cleaning DAG
    trigger_data_cleaning_dag_task = TriggerDagRunOperator(
        task_id="data_cleaning_trigger",
        trigger_rule=TriggerRule.ALL_DONE,
        trigger_dag_id="Data_cleaning_pipeline",
        dag=dag,
    )

# Data Preprocessing DAG
with DAG(
    "Data_cleaning_pipeline",
    default_args=default_args,
    description="DAG for Data Preprocessing",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 10, 17),
    catchup=False,
    on_success_callback=clear_xcom,
) as dag:

    # Task 1: Data Cleaning
    data_cleaning_task = PythonOperator(
        task_id="datacleaning_process",
        python_callable=data_cleaning,
        provide_context=True,
    )

    # Task 2: Remove Abusive Data
    remove_abusive_task = PythonOperator(
        task_id="remove_abusive_data_task",
        python_callable=remove_abusive_data,
        op_args=[data_cleaning_task.output],
    )

(
    data_loading_task
    >> filter_parallel_tasks
    >> aggregate_parallel_tasks
    >> trigger_data_cleaning_dag_task
)
