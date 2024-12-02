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
    train_data: Output[Dataset],
    holdout_data: Output[Dataset],
    testset_size: float = 0.2,
    limit:int=None):

  from google.cloud import bigquery
  from sklearn.model_selection import train_test_split
  from datetime import datetime

  # Track the start time of the component execution
  start_time = datetime.now()

  try:
    bqclient = bigquery.Client(project=project_id, location=location)

    QUERY = f'''select complaint_english, complaint_hindi, department from `bilingualcomplaint-system.MLOps`.get_dataset_by_complaint_year_interval({start_year}, {end_year})'''
    if limit:
       QUERY = f'{QUERY} limit {limit}'
    
    query_job = bqclient.query(QUERY)  # API request
    rows = query_job.result()  # Waits for query to finish
    data = rows.to_dataframe()
    
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
    # Track the end time and calculate duration
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60  # Duration in minutes

  except Exception as e:
      # Send failure email if there's an error
      error_message = str(e)