import requests
from airflow import DAG
from airflow.decorators import dag, task
from airflow.models import TaskInstance
from airflow.providers.slack.hooks.slack_webhook import SlackWebhookHook
from airflow.utils.db import provide_session
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
from airflow.models.xcom import XCom
from airflow.utils.trigger_rule import TriggerRule
from scripts.preprocessing import (
    load_data,
    filter_records_by_word_count_and_date,
    filter_records_by_language,
    aggregate_filtered_task,
    data_cleaning,
    remove_abusive_data,
    insert_data_to_bigquery,
    standardise_product_class
)
from scripts.success_email import send_success_email
from scripts.failure_email import send_failure_email
from scripts.deidentification import anonymize_sensitive_data
from scripts.data_quality import validate_data_quality
from scripts.statistics_generation import schema_and_statistics_generation

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

MIN_WORD: int = 5
SLACK_WEBHOOK = (
    "https://hooks.slack.com/services/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"
)

def send_slack_notification(message: str):
    """Send a message to Slack via the defined webhook URL."""
    slack_msg = {"text": message}
    response = requests.post(
        SLACK_WEBHOOK, json=slack_msg, headers={"Content-Type": "application/json"}
    )
    if response.status_code != 200:
        raise ValueError(
            f"Request to Slack returned an error {response.status_code}, "
            f"the response is: {response.text}"
        )


def dag_success_alert(context):
    """Callback for successful DAG run."""
    dag_id = context.get("dag").dag_id
    execution_date = context.get("execution_date")
    log_url = context.get("task_instance").log_url

    # Format execution date and time
    exec_date_str = execution_date.strftime("%Y-%m-%d")
    exec_time_str = execution_date.strftime("%H:%M:%S")

    start_date = context.get("dag_run").start_date
    end_date = context.get("dag_run").end_date

    duration = (end_date - start_date).total_seconds() / 60
    duration = round(duration, 2)

    message = (
        f":large_green_circle: DAG Success Alert\n"
        f"*DAG Name*: {dag_id}\n"
        f"*Execution Date*: {exec_date_str}\n"
        f"*Execution Time*: {exec_time_str}\n"
        f"*Duration*: {duration} minutes\n"
        f"*Log URL*: {log_url}"
    )
    send_slack_notification(message)


def dag_failure_alert(context):
    """Callback for failed DAG run."""
    dag_id = context.get("dag").dag_id
    execution_date = context.get("execution_date")
    log_url = context.get("task_instance").log_url

    # Format execution date and time
    exec_date_str = execution_date.strftime("%Y-%m-%d")
    exec_time_str = execution_date.strftime("%H:%M:%S")

    start_date = context.get("dag_run").start_date
    end_date = context.get("dag_run").end_date

    duration = (end_date - start_date).total_seconds() / 60
    duration = round(duration, 2)

    message = (
        f":red_circle: DAG Failure Alert\n"
        f"*DAG Name*: {dag_id}\n"
        f"*Execution Date*: {exec_date_str}\n"
        f"*Execution Time*: {exec_time_str}\n"
        f"*Duration*: {duration} minutes\n"
        f"*Log URL*: {log_url}"
    )
    send_slack_notification(message)


# Define the function to clear XComs
@provide_session
def clear_xcom(context, session=None):
    dag_id = context["ti"]["dag"]
    execution_date = context["ti"]["execution_date"]
    session.query(XCom).filter(
        XCom.dag_id == dag_id, XCom.execution_date == execution_date
    ).delete()


# Data Preprocessing Pipeline Initialization DAG

with DAG(
    "Data_Preprocessing_INIT",
    default_args=default_args,
    description="DAG to start Data Preprocessing pipeline",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 10, 17),
    on_failure_callback=dag_failure_alert,
    on_success_callback=dag_success_alert,
    catchup=False,
) as dag:
    # Task: Run other DAGs
    trigger_data_validation_dag_task = TriggerDagRunOperator(
        task_id="data_validation_trigger",
        trigger_rule=TriggerRule.ALL_DONE,
        trigger_dag_id="Data_Validation_Pipeline",
        dag=dag,
    )

    trigger_data_validation_dag_task


# Data Validation DAG
with DAG(
    "Data_Validation_Pipeline",
    default_args=default_args,
    description="DAG for Data Validation",
    schedule_interval=None,
    on_failure_callback=dag_failure_alert,
    on_success_callback=dag_success_alert,
    catchup=False,
) as dag:

    # Task 1: Load the data
    data_loading_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
        on_failure_callback=send_failure_email,
    )

    # Task 2: Validate the data
    data_validation_task = PythonOperator(
        task_id="data_quality_checks",
        python_callable=validate_data_quality,
        op_args=[data_loading_task.output],
        on_failure_callback=send_failure_email,
    )

    # Task 3
    schema_and_statistics_generation_task = PythonOperator(
        task_id="schema_and_statistics_generation",
        python_callable=schema_and_statistics_generation,
        op_args=[data_validation_task.output],
        on_failure_callback=send_failure_email,
    )

    # Task 4, 5: Parallel Data Processing
    # Task 4: Filter out records based on word count and specified range criteria.
    # Task 5: Remove un-recognised language

    filter_parallel_tasks = [
        PythonOperator(
            task_id="remove_records_with_minimum_words_and_outdated_records",
            python_callable=filter_records_by_word_count_and_date,
            op_args=[schema_and_statistics_generation_task.output, MIN_WORD],
            on_failure_callback=send_failure_email,
        ),
        PythonOperator(
            task_id="language_checks",
            python_callable=filter_records_by_language,
            op_args=[schema_and_statistics_generation_task.output],
            on_failure_callback=send_failure_email,
        ),
    ]

    # Task 5: Aggregate results from Task 3, 4
    aggregate_parallel_tasks = PythonOperator(
        task_id="aggregation",
        python_callable=aggregate_filtered_task,
        op_args=[filter_parallel_tasks[0].output, filter_parallel_tasks[1].output],
        provide_context=True,
        on_failure_callback=send_failure_email,
    )

    # Task 6: Trigger Data Cleaning DAG
    trigger_data_cleaning_dag_task = TriggerDagRunOperator(
        task_id="data_cleaning_trigger",
        trigger_rule=TriggerRule.ALL_DONE,
        trigger_dag_id="Data_Cleaning_Pipeline",
        dag=dag,
        on_failure_callback=send_failure_email,
    )

    (
        data_loading_task
        >> data_validation_task
        >> schema_and_statistics_generation_task
        >> filter_parallel_tasks
        >> aggregate_parallel_tasks
        >> trigger_data_cleaning_dag_task
    )

# Data Cleaning DAG
with DAG(
    "Data_Cleaning_Pipeline",
    default_args=default_args,
    description="DAG for Data Preprocessing",
    schedule_interval=None,
    on_failure_callback=dag_failure_alert,
    on_success_callback=dag_success_alert,
    catchup=False,
) as dag:

    # Task 1: Data Cleaning
    data_cleaning_task = PythonOperator(
        task_id="datacleaning_process",
        python_callable=data_cleaning,
        provide_context=True,
        on_failure_callback=send_failure_email,
    )

    # Task 2: Anonymize sensitive data
    anonymization_task = PythonOperator(
        task_id="anonymize_sensitive_data_task",
        python_callable=anonymize_sensitive_data,
        op_args=[data_cleaning_task.output],
        on_failure_callback=send_failure_email,
    )
    # Task 3: Remove Abusive Data
    remove_abusive_task = PythonOperator(
        task_id="remove_abusive_data_task",
        python_callable=remove_abusive_data,
        op_args=[anonymization_task.output],
        on_failure_callback=send_failure_email,
    )

    # Task 4: Standardize Product Labels
    product_standardization_task = PythonOperator(
        task_id="standardize_product_class_task",
        python_callable=standardise_product_class,
        op_args=[remove_abusive_task.output],
        on_failure_callback=send_failure_email,
    )

    # Task 5: Send Success Email
    send_email_task = PythonOperator(
        task_id="send_success_email_task",
        python_callable=send_success_email,
        provide_context=True,
        on_failure_callback=send_failure_email,
    )

    # # Task 6: Insert data into BigQuery
    # insert_to_bigquery_task = PythonOperator(
    #     task_id="insert_to_bigquery_task",
    #     python_callable=insert_data_to_bigquery,
    #     op_args=[remove_abusive_task.output],
    #     on_failure_callback=send_failure_email,
    # )

    (
        data_cleaning_task
        >> anonymization_task
        >> remove_abusive_task
        >> product_standardization_task
        >> send_email_task
        # >> [insert_to_bigquery_task, send_email_task]
    )
