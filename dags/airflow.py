from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from scripts.preprocessing import data_cleaning, remove_abusive_data, remove_special_characters  # Importing the functions from preprocessing.py

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'Data_cleaning_pipeline',
    default_args=default_args,
    description='DAG for Data cleaning preprocessing',
    schedule_interval=timedelta(days=1),  
    start_date=datetime(2024, 10, 17),
    catchup=False,
) as dag:

    # Task 1: Data Cleaning Task
    data_cleaning_task = PythonOperator(
        task_id='datacleaning_process',
        python_callable=data_cleaning,  # Reference the data_cleaning function
        provide_context=True,
    )

    # Task 2: Remove Special Characters Task
    remove_special_characters_task = PythonOperator(
        task_id='remove_special_characters',
        python_callable=remove_special_characters,
        provide_context=True,
    )

    # Task 3: Remove Abusive Data Task
    remove_abusive_task = PythonOperator(
        task_id='remove_abusive_data_task',
        python_callable=remove_abusive_data,  # Reference the remove_abusive_data function
        provide_context=True,
    )

    # Set task dependencies: remove_abusive_task runs after data_cleaning_task
    data_cleaning_task >> remove_abusive_task >> remove_abusive_task