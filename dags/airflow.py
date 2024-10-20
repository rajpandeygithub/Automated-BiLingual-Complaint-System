from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from scripts.preprocessing import data_cleaning  # Importing the function from preprocessing.py

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

    # Task to process data
    data_cleaning_task = PythonOperator(
        task_id='datacleaning_process',
        python_callable=data_cleaning,  # Reference the process_data function
        provide_context=True,
    )

    # Task dependencies 
    data_cleaning_task
