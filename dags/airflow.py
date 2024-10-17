from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os


def deduplicate_records(df):
    deduped_df = df.drop_duplicates(subset=['Product', 'Sub-product', 'Consumer complaint narrative'], keep='first')
    return deduped_df

# Task function to read the dataset, perform preprocessing, and save the result
def process_data(**kwargs):
    file_path = os.path.join(os.path.dirname(__file__), "data/JPMORGAN_CHASE_COMPLAINTS_v2 (1).csv")

    df = pd.read_csv(file_path)


    # Deduplicate the records
    deduped_df = deduplicate_records(df)

    # Save the deduplicated data to a new file
    output_path = os.path.join(os.path.dirname(__file__), "data/deduplicated_dataset.csv")
    deduped_df.to_csv(output_path, index=False)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


with DAG(
    'deduplication_pipeline',
    default_args=default_args,
    description='A simple DAG to deduplicate dataset',
    schedule_interval=timedelta(days=1),  
    start_date=datetime(2024, 10, 17),
    catchup=False,
) as dag:

    # Task to process data
    deduplicate_task = PythonOperator(
        task_id='deduplicate_data',
        python_callable=process_data,
        provide_context=True,  \
    )

    # Task dependencies 
    deduplicate_task
