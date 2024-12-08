from typing import Dict
from kfp.dsl import Output, Dataset, component

@component(
    base_image="python:3.10-slim",
    packages_to_install = [
        'google-cloud-bigquery==3.26.0',
        'pandas==1.5.3',
        'numpy==1.26.4',
        'db-dtypes==1.3.0',
        'scikit-learn==1.5.2'
        ]
    )
def get_data_component(
    project_id: str,
    location: str,
    start_year: int, end_year: int,
    label_name: str,
    minimum_label_count: int,
    train_data: Output[Dataset],
    holdout_data: Output[Dataset],
    testset_size: float = 0.2,
    limit:int=None,
    slack_url: str = None
    ):
  
  import requests
  from google.cloud import bigquery
  from sklearn.model_selection import train_test_split
  from datetime import datetime

  def send_slack_message(
        webhook_url: str,
        message_str: str,
        execution_date: str, 
        execution_time: str, 
        duration: str,
        is_success: bool,
        ):
    
    if is_success:
        color = "#36a64f"
        pretext = f":large_green_circle: {message_str}"
    else:
        color = "FF0000"
        pretext = f":large_red_circle: {message_str}"

    message = {
        "attachments": [
            {
                "color": color,  # Green color for success
                "pretext": pretext,
                "fields": [
                    {
                        "title": "Component Name",
                        "value": "Get Data KubeFlow Component",
                        "short": True
                    },
                    {
                        "title": "Execution Date",
                        "value": str(execution_date),
                        "short": True
                    },
                    {
                        "title": "Execution Time",
                        "value": str(execution_time),
                        "short": True
                    },
                    {
                        "title": "Duration",
                        "value": f"{duration} minutes",
                        "short": True
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(webhook_url, json=message)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(e)

  start_time = datetime.now()
  
  if slack_url:
    send_slack_message(
       webhook_url=slack_url, message_str='KubeFlow Component: Get BigQuery Data | Component Started', 
       execution_date=start_time.date(), execution_time=start_time.time(), 
       duration=0, is_success=True
       )

  try:
    bqclient = bigquery.Client(project=project_id, location=location)

    if label_name == 'product':
       QUERY = f'''select complaint_english, complaint_hindi, {label_name} from `bilingualcomplaint-system.MLOps`.undersampled_product_dataset({minimum_label_count}, {start_year}, {end_year})'''
    elif label_name == 'department':
       QUERY = f'''select complaint_english, complaint_hindi, {label_name} from `bilingualcomplaint-system.MLOps`.undersampled_department_dataset({minimum_label_count}, {start_year}, {end_year})'''
    else:
       if slack_url:
          send_slack_message(
             webhook_url=slack_url, message_str='KubeFlow Component: Get BigQuery Data | Failed Data Pull from BigQuery', 
             execution_date=start_time.date(), execution_time=start_time.time(), 
             duration=(datetime.now() - start_time).total_seconds() / 60,
             is_success=False
             )
       raise ValueError(f'Unexpected label. Use only `product` or `department`.')

    if limit:
       QUERY = f'{QUERY} limit {limit}'
    
    query_job = bqclient.query(QUERY)  # API request
    rows = query_job.result()  # Waits for query to finish
    data = rows.to_dataframe()

    if slack_url:
        send_slack_message(
           webhook_url=slack_url, message_str='KubeFlow Component: Get BigQuery Data | Data Pull from BigQuery Complete', 
           execution_date=start_time.date(), execution_time=start_time.time(), 
           duration=(datetime.now() - start_time).total_seconds() / 60,
           is_success=True
           )
    
    # Melt the DataFrame
    data_features = data.melt(
        id_vars=[label_name],  # Columns to keep as-is
        value_vars=["complaint_english", "complaint_hindi"],  # Columns to melt
        value_name="complaints"  # New column name for melted values
    )

    train, holdout = train_test_split(data_features[['complaints', label_name]], test_size=testset_size, stratify=data_features[label_name], random_state=42)
    train.reset_index(drop=True, inplace=True)
    holdout.reset_index(drop=True, inplace=True)

    train.to_pickle(train_data.path)
    holdout.to_pickle(holdout_data.path)

    if slack_url:
        send_slack_message(
           webhook_url=slack_url, message_str='KubeFlow Component: Get BigQuery Data | Component Finished', 
           execution_date=start_time.date(), execution_time=start_time.time(), 
           duration=(datetime.now() - start_time).total_seconds() / 60,
           is_success=True
           )

  except Exception as e:
      # Send failure email if there's an error
      error_message = str(e)
      if slack_url:
        send_slack_message(
           webhook_url=slack_url, message_str='KubeFlow Component: Get BigQuery Data | Component Failed', 
           execution_date=start_time.date(), execution_time=start_time.time(), 
           duration=(datetime.now() - start_time).total_seconds() / 60,
           is_success=False
           )
      raise (e)