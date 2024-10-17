from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os

def check_language_min_length(df, min_length=5):
    pass

# Function to hide PII information
def hide_pii(df):
    pass

# Function to replace abusive words
def replace_abusive_words(df):
    pass

def deduplicate_records(df):
    deduped_df = df.drop_duplicates(subset=['Product', 'Sub-product', 'Consumer complaint narrative'], keep='first')
    return deduped_df


# Task function for the first DAG (Language Check)
def process_language_check(**kwargs):
    file_path = os.path.join(os.path.dirname(__file__), "data/JPMORGAN_CHASE_COMPLAINTS_v2 (1).csv")
    df = pd.read_csv(file_path)
    checked_df = check_language_min_length(df)
    
    # Save checked data and push to XCom
    output_path = os.path.join(os.path.dirname(__file__), "data/language_checked_dataset.csv")
    checked_df.to_csv(output_path, index=False)
    
    # Push the output path to XCom
    kwargs['ti'].xcom_push(key='language_checked_data', value=output_path)

# Task function for the second DAG (PII Hiding)
def process_pii_hiding(**kwargs):
    # Pull the output path from the previous DAG's XCom
    prev_output_path = kwargs['ti'].xcom_pull(key='language_checked_data', task_ids='language_check_pipeline.check_language_min_length')
    
    df = pd.read_csv(prev_output_path)
    pii_hidden_df = hide_pii(df)
    
    output_path = os.path.join(os.path.dirname(__file__), "data/pii_hidden_dataset.csv")
    pii_hidden_df.to_csv(output_path, index=False)
    
    # Push the output path to XCom
    kwargs['ti'].xcom_push(key='pii_hidden_data', value=output_path)
    
# Task function to read the dataset, perform preprocessing, and save the result
def process_data(**kwargs):
    file_path = kwargs['ti'].xcom_pull(key='pii_hidden_data', task_ids='pii_hiding_pipeline.hide_pii_information')

    df = pd.read_csv(file_path)


    # Deduplicate the records
    deduped_df = deduplicate_records(df)

    # Save the clean data to a new file
    output_path = os.path.join(os.path.dirname(__file__), "data/deduplicated_dataset.csv")
    cleaned_df.to_csv(output_path, index=False)





# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_success': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    
}

# DAG for checking language and minimum length
with DAG(
    'language_check_pipeline',
    default_args=default_args,
    description='A DAG to check language and minimum word length',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 10, 17),
    catchup=False,
) as dag1:

    language_check_task = PythonOperator(
        task_id='check_language_min_length',
        python_callable=process_language_check,
        provide_context=True,
    )



# DAG for hiding PII information
with DAG(
    'pii_hiding_pipeline',
    default_args=default_args,
    description='A DAG to hide PII information',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 10, 17),
    catchup=False,
) as dag2:

    pii_hiding_task = PythonOperator(
        task_id='hide_pii_information',
        python_callable=process_pii_hiding,
        provide_context=True,
    )



# DAG for cleaning and preprocessing
with DAG(
    'cleaning_pipeline',
    default_args=default_args,
    description='DAG for cleaning and preprocessing data',
    schedule_interval=timedelta(days=1),  
    start_date=datetime(2024, 10, 17),
    catchup=False,
) as dag3:
        
    cleaning_task = PythonOperator(
        task_id='clean_and_deduplicate',
        python_callable=process_data,
        provide_context=True,  \
    )

    # Task dependencies 
    cleaning_task
