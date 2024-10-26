from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
from airflow.utils.trigger_rule import TriggerRule
from scripts.preprocessing import data_loading, minimum_word_check, detect_language, aggregate_filtered_task, data_cleaning, remove_special_characters, remove_abusive_data

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

MIN_WORD: int = 5

# Data Validation DAG
with DAG(
    'Data_validation_pipeline',
    default_args=default_args,
    description='DAG for Data Validation',
    schedule_interval=timedelta(days=1),  
    start_date=datetime(2024, 10, 17),
    catchup=False,
) as dag:
    
    # Task 1: Load the data
    data_loading_task = PythonOperator(
        task_id = 'data_loading',
        python_callable=data_loading,
    )
    # Task 2: Removing records with less than 5 words
    minimum_words_filter = PythonOperator(
        task_id = 'minimum_words_filter',
        python_callable=minimum_word_check,
        op_args=[data_loading_task.output, MIN_WORD],
    )
    # Task 3: Detect language and remove un-recognized language
    language_filter = PythonOperator(
        task_id = 'detect_language',
        python_callable=detect_language,
        op_args=[minimum_words_filter.output],
    )
    # Task 4: Aggregate results from Task 2 & Task 3
    aggregate_parallel_tasks = PythonOperator(
        task_id = 'validation_aggregation',
        python_callable=aggregate_filtered_task,
        op_args=[minimum_words_filter.output, language_filter.output],
        provide_context=True,
    )
    # Task 4: Trigger Data Cleaning DAG
    TriggerDagRunOperator = TriggerDagRunOperator(
        task_id='data_cleaning_trigger',
        trigger_rule=TriggerRule.ALL_DONE,
        trigger_dag_id='Data_cleaning_pipeline',
        dag=dag
        )

# Data Preprocessing DAG
with DAG(
    'Data_cleaning_pipeline',
    default_args=default_args,
    description='DAG for Data Preprocessing',
    schedule_interval=timedelta(days=1),  
    start_date=datetime(2024, 10, 17),
    catchup=False,
) as dag:

    # Task 1: Data Cleaning
    data_cleaning_task = PythonOperator(
        task_id='datacleaning_process',
        python_callable=data_cleaning,
        provide_context=True,
    )

    # Task 2: Remove Special Characters
    remove_special_characters_task = PythonOperator(
        task_id='remove_special_characters',
        python_callable=remove_special_characters,
        op_args=[data_cleaning_task.output]
    )

    # Task 3: Remove Abusive Data
    remove_abusive_task = PythonOperator(
        task_id='remove_abusive_data_task',
        python_callable=remove_abusive_data,
        op_args=[remove_special_characters_task.output]
    )

data_loading_task >> [language_filter, minimum_words_filter] >> aggregate_parallel_tasks >> TriggerDagRunOperator