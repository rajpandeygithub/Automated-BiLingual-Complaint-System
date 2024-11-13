# Authenticate your Google Cloud account
import os

# Set the environment variable for Google application credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "pipeline/all_in_one_service_account_key.json"

from google.cloud import aiplatform


import kfp
import json
from datetime import datetime
from kfp import compiler, dsl
from typing import NamedTuple, List, Union, Dict, Any
from kfp.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics, Dataset
from google.cloud import aiplatform
from model_registry import ModelRegistry

print(f'KFB version: {kfp.__version__}')



PROJECT_ID = 'bilingualcomplaint-system'
LOCATION = 'us-east1'
# Bucket Name
GCS_artifacts_bucket_name = 'tfx-artifacts'
# Pipeline
pipeline_name = 'complaints-clf-vertex-training'
# Path to various pipeline artifact.
_pipeline_artifacts_dir = f'gs://{GCS_artifacts_bucket_name}/pipeline_artifacts/{pipeline_name}'

aiplatform.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=f'gs://{GCS_artifacts_bucket_name}',
    )



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
    feature_name: str,
    label_name: str,
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    testset_size: float = 0.2,
    limit:int=100):

  from google.cloud import bigquery
  from sklearn.model_selection import train_test_split

  bqclient = bigquery.Client(project=project_id, location=location)

  QUERY = f'''select * from `bilingualcomplaint-system.MLOps`.get_dataset_by_complaint_year_interval({start_year}, {end_year}) limit {limit}'''
  query_job = bqclient.query(QUERY)  # API request
  rows = query_job.result()  # Waits for query to finish
  data = rows.to_dataframe()
  data_features = data[[feature_name, label_name]]
  train, val = train_test_split(data_features, test_size=testset_size, random_state=42)
  train.reset_index(drop=True, inplace=True)
  val.reset_index(drop=True, inplace=True)
  train.to_pickle(train_data.path)
  val.to_pickle(val_data.path)


# Second Component - Training Script
#Training Component
@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        'google-cloud-storage==2.18.2',
        'pandas==1.5.3',
        'numpy==1.23.5',
        'scikit-learn==1.2.2',
        'xgboost==1.6.1'
    ]
)
def train_xgboost_model(
    train_data: Input[Dataset],
    feature_name: str,
    label_name: str,
    model: Output[Model],
    vectorizer_output: Output[Artifact]
):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb
    import os
    import pickle

    # Load dataset from the train_data input artifact
    data = pd.read_pickle(train_data.path)
    X = data[feature_name].fillna("")
    y = data[label_name].fillna("")

    # Encode the target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Initialize and fit TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    # Define and train XGBoost model
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_tfidf, y_encoded)

    # Save the model directly into a "model" folder
    model_directory = model.path
    os.makedirs(model_directory, exist_ok=True)
    model_output_path = os.path.join(model_directory, "model.bst")
    xgb_model.save_model(model_output_path)


    # Save the vectorizer as 'tfidf_vectorizer.pkl' directly to the vectorizer_output path
    vectorizer_output_path = f"{vectorizer_output.path}.pkl"
    with open(vectorizer_output_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)



# Testing Component
@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        'google-cloud-storage==2.18.2',
        'pandas==1.5.3',
        'numpy==1.23.5',
        'scikit-learn==1.2.2',
        'xgboost==1.6.1'
    ]
)
def test_xgboost_model(
    val_data: Input[Dataset],
    model_input: Input[Model],
    vectorizer_input: Input[Artifact],
    feature_name: str,
    label_name: str
) -> float:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder  # Import added
    from sklearn.metrics import f1_score, precision_score, recall_score
    import pickle
    import xgboost as xgb
    import os

    # Load the trained model
    model_path = os.path.join(model_input.path, "model.bst")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_path)

    # Load the TF-IDF vectorizer
    vectorizer_path = f"{vectorizer_input.path}.pkl"
    with open(vectorizer_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    # Load validation dataset from val_data input artifact
    data = pd.read_pickle(val_data.path)
    X_val = data[feature_name].fillna("")
    y_val = data[label_name].fillna("")

    # Transform validation data with vectorizer
    X_val_tfidf = tfidf_vectorizer.transform(X_val)

    # Make predictions
    y_pred_encoded = xgb_model.predict(X_val_tfidf)

    # Decode predictions to match the original label format
    label_encoder = LabelEncoder()
    label_encoder.fit(y_val)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # Calculate metrics
    f1 = f1_score(y_val, y_pred, average="macro")
    precision = precision_score(y_val, y_pred, average="macro")
    recall = recall_score(y_val, y_pred, average="macro")

    print(f"Validation F1 Score: {f1:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")

    return f1

# register
@component(
    packages_to_install=["google-cloud-aiplatform", "google-auth"]
)
def model_registration(
    model_output: Input[Model],
    project_id: str,
    location: str,
    model_display_name: str,
    model: Output[Model]
):
    from google.cloud import aiplatform

    # Initialize Vertex AI SDK
    aiplatform.init(project=project_id, location=location)

    # Use the URI directly without appending "/model"
    model_uri = model_output.uri

    # Register the model with the specified container for XGBoost
    registered_model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_uri,  # Direct path to the model artifact
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-5:latest",  # XGBoost serving container
    )

    # Output the model resource name
    model.uri = registered_model.resource_name


@component(
    packages_to_install=["google-cloud-aiplatform"]
)
def model_deployment(
    model: Input[Model],
    project_id: str,
    location: str,
    endpoint_display_name: str,
    deployed_model_display_name: str,
    endpoint: Output[Artifact]
):
    import time
    from google.cloud import aiplatform

    # Delay to allow for model registration completion
    time.sleep(30)  # Wait for 30 seconds

    # Initialize Vertex AI SDK
    aiplatform.init(project=project_id, location=location)

    # Retrieve the model using its resource name (model.uri should be in the format projects/PROJECT_ID/locations/LOCATION/models/MODEL_ID)
    deployed_model = aiplatform.Model(model.uri)

    # Create or get an existing endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        order_by='create_time desc',
        project=project_id,
        location=location
    )
    if endpoints:
        endpoint_obj = endpoints[0]
    else:
        endpoint_obj = aiplatform.Endpoint.create(display_name=endpoint_display_name)

    # Deploy the model to the endpoint
    endpoint_obj.deploy(
        model=deployed_model,
        deployed_model_display_name=deployed_model_display_name,
        machine_type="n1-standard-4",
    )

    # Output the endpoint resource name
    endpoint.uri = endpoint_obj.resource_name


#pipeline
@pipeline(
    name="model-data-pipeline",
    description="Model data pipeline - Training | Testing | Model Selection | Registration | Deployment",
    pipeline_root=_pipeline_artifacts_dir,
)
def model_data_pipeline(
    start_year: int = 2018,
    end_year: int = 2020,
    limit: int = 100,
    feature_name: str = 'complaint_english',
    label_name: str = 'product',
    model_display_name: str = "xgboost-complaints-model",
    endpoint_display_name: str = "xgboost-complaints-endpoint",
    deployed_model_display_name: str = "xgboost-complaints-deployment"
):
    # Fetch data
    get_data_component_task = get_data_component(
        project_id=PROJECT_ID,
        location=LOCATION,
        start_year=start_year,
        end_year=end_year,
        feature_name=feature_name,
        label_name=label_name,
        testset_size=0.2,
        limit=limit
    )

    # Train model
    train_xgboost_task = train_xgboost_model(
        train_data=get_data_component_task.outputs['train_data'],
        feature_name=feature_name,
        label_name=label_name
    )

    # Test model (runs after training)
    test_xgboost_task = test_xgboost_model(
        val_data=get_data_component_task.outputs['val_data'],
        model_input=train_xgboost_task.outputs["model"],
        vectorizer_input=train_xgboost_task.outputs["vectorizer_output"],
        feature_name=feature_name,
        label_name=label_name
    )

    # Register model (runs after testing)
    model_registration_task = model_registration(
        model_output=train_xgboost_task.outputs["model"],
        project_id=PROJECT_ID,
        location=LOCATION,
        model_display_name=model_display_name
    )
    model_registration_task.after(test_xgboost_task)

    # Deploy model (runs after registration)
    model_deployment_task = model_deployment(
        model=model_registration_task.outputs["model"],
        project_id=PROJECT_ID,
        location=LOCATION,
        endpoint_display_name=endpoint_display_name,
        deployed_model_display_name=deployed_model_display_name
    )

from kfp import compiler
from google.cloud import aiplatform
from datetime import datetime

# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=model_data_pipeline,
    package_path="model_data_pipeline_job.json"
)

# Generate a unique job ID using a timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

# Submit the job to Vertex AI
job = aiplatform.PipelineJob(
    display_name="model-data-pipeline",
    template_path="model_data_pipeline_job.json",
    job_id=f"model-data-pipeline-{TIMESTAMP}",
    enable_caching=True,
    pipeline_root=_pipeline_artifacts_dir,  # Make sure this path is set correctly
    parameter_values={
        "start_year": 2018,
        "end_year": 2020,
        "limit": 100,
        "feature_name": "complaint_english",
        "label_name": "product"
    }
)
job.submit()
