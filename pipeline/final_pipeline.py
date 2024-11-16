# Authenticate your Google Cloud account
import os
import json  # Added this line
from google.oauth2 import service_account


service_account_info = json.loads(os.getenv("GCP_SA_KEY"))
credentials = service_account.Credentials.from_service_account_info(service_account_info)

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
    limit:int=200):

  import os
  from google.cloud import bigquery
  from sklearn.model_selection import train_test_split
  import smtplib
  from email.mime.multipart import MIMEMultipart
  from email.mime.text import MIMEText
  import requests
  from datetime import datetime

  # Track the start time of the component execution
  start_time = datetime.now()

  # Function to send custom Slack message with Kubeflow component details
  def send_slack_message(component_name, execution_date, execution_time, duration, f1_score=None, precision=None, recall=None):
    # Get the Slack webhook URL from environment variables
    SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
    if not SLACK_WEBHOOK_URL:
        print("Error: SLACK_WEBHOOK_URL not found in environment variables.")  # Replace with your Slack webhook URL
        message = {
          "attachments": [
              {
                  "color": "#36a64f",  # Green color for success
                  "pretext": ":large_green_circle: Kubeflow Component Success Alert",
                  "fields": [
                      {
                          "title": "Component Name",
                          "value": component_name,
                          "short": True
                      },
                      {
                          "title": "Execution Date",
                          "value": execution_date,
                          "short": True
                      },
                      {
                          "title": "Execution Time",
                          "value": execution_time,
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
            response = requests.post(SLACK_WEBHOOK_URL, json=message)
            response.raise_for_status()  # Check for request errors
            pass
        except requests.exceptions.RequestException as e:
            pass

  # Function to send success email
  def send_success_email():
      sender_email = "sucessemailtrigger@gmail.com"
      password = "jomnpxbfunwjgitb"
      receiver_emails = ["hegde.anir@northeastern.edu",
                         "nenavath.r@northeastern.edu",
                         "pandey.raj@northeastern.edu",
                         "khatri.say@northeastern.edu",
                         "singh.arc@northeastern.edu",
                         "goparaju.v@northeastern.edu"]

      # Current time for logging purposes
      current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

      # Create the email content
      subject = '[Kubeflow Pipeline] - Started'
      body = f'''Hi team,

      Model training in the Kubeflow pipeline has started!

      Details:
      - Start Time: {current_time}
      - Dataset: {start_year}-{end_year}

      Please monitor the pipeline for further updates.
      '''

      try:
          # Set up the SMTP server
          server = smtplib.SMTP('smtp.gmail.com', 587)
          server.starttls()  # Secure the connection
          server.login(sender_email, password)

          # Send email to each receiver
          for receiver_email in receiver_emails:
              # Create a fresh message for each recipient
              message = MIMEMultipart()
              message['From'] = sender_email
              message['To'] = receiver_email
              message['Subject'] = subject
              message.attach(MIMEText(body, 'plain'))

              # Send the email
              server.sendmail(sender_email, receiver_email, message.as_string())

      except Exception as e:
          pass
      finally:
          server.quit()

  # Function to send failure email
  def send_failure_email(error_message):
      sender_email = "sucessemailtrigger@gmail.com"
      password = "jomnpxbfunwjgitb"
      receiver_emails = ["hegde.anir@northeastern.edu",
                         "nenavath.r@northeastern.edu",
                         "pandey.raj@northeastern.edu",
                         "khatri.say@northeastern.edu",
                         "singh.arc@northeastern.edu",
                         "goparaju.v@northeastern.edu"]

      # Create the email content
      subject = '[Kubeflow Pipeline]'
      body = f'Hi team,\nModel training has failed!.\nError Details: {error_message}'

      try:
          # Set up the SMTP server
          server = smtplib.SMTP('smtp.gmail.com', 587)
          server.starttls()  # Secure the connection
          server.login(sender_email, password)

          # Send email to each receiver
          for receiver_email in receiver_emails:
              # Create a fresh message for each recipient
              message = MIMEMultipart()
              message['From'] = sender_email
              message['To'] = receiver_email
              message['Subject'] = subject
              message.attach(MIMEText(body, 'plain'))

              # Send the email
              server.sendmail(sender_email, receiver_email, message.as_string())

      except Exception as e:
          pass
      finally:
          server.quit()
  try:
    bqclient = bigquery.Client(project=project_id, location=location)

    QUERY = f'''select * from `bilingualcomplaint-system.MLOps`.get_dataset_by_complaint_year_interval({start_year}, {end_year}) limit {limit}'''
    query_job = bqclient.query(QUERY)  # API request
    rows = query_job.result()  # Waits for query to finish
    data = rows.to_dataframe()
    # Selecting the necessary features and labels
    data_features = data[[feature_name, label_name]]

    # Initial split
    train, val = train_test_split(data_features, test_size=testset_size, random_state=42)

    # Identify labels in training set
    train_labels = set(train[label_name])

    # Filter validation set to remove rows with labels not in training set
    val = val[val[label_name].isin(train_labels)]

    # Reset indices and save
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    train.to_pickle(train_data.path)
    val.to_pickle(val_data.path)

    # Track the end time and calculate duration
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60  # Duration in minutes

    # Send success email once the data is processed
    send_success_email()
    # Send the Slack message with execution details
    send_slack_message(
        component_name="Getting Data Component",
        execution_date=end_time.strftime('%Y-%m-%d'),
        execution_time=end_time.strftime('%H:%M:%S'),
        duration=round(duration, 2)  # Round duration to 2 decimal places
    )

  except Exception as e:
      # Send failure email if there's an error
      error_message = str(e)
      send_failure_email(error_message)
      send_slack_message(
          component_name="Model Training Component Failed",
          execution_date=datetime.now().strftime('%Y-%m-%d'),
          execution_time=datetime.now().strftime('%H:%M:%S'),
          duration=0  # If failed, duration is 0
      )


#---------------------------------------------------------------------------------------------------------------------------------------

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
    vectorizer_output: Output[Artifact],
    label_encoder_output: Output[Artifact]  # New output for the LabelEncoder
):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb
    import os
    import pickle
    import requests
    from datetime import datetime

    # Track the start time of the component execution
    start_time = datetime.now()

    # Function to send custom Slack message with Kubeflow component details
    def send_slack_message(component_name, execution_date, execution_time, duration, f1_score=None, precision=None, recall=None):
    # Get the Slack webhook URL from environment variables
        SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
        if not SLACK_WEBHOOK_URL:
            print("Error: SLACK_WEBHOOK_URL not found in environment variables.") # Replace with your Slack webhook URL
            message = {
                "attachments": [
                    {
                        "color": "#36a64f",  # Green color for success
                        "pretext": ":large_green_circle: Kubeflow Component Success Alert",
                        "fields": [
                            {
                                "title": "Component Name",
                                "value": component_name,
                                "short": True
                            },
                            {
                                "title": "Execution Date",
                                "value": execution_date,
                                "short": True
                            },
                            {
                                "title": "Execution Time",
                                "value": execution_time,
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
                response = requests.post(SLACK_WEBHOOK_URL, json=message)
                response.raise_for_status()  # Check for request errors
                pass
            except requests.exceptions.RequestException as e:
                pass
    try:
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

      # Save the label encoder as 'label_encoder.pkl'
      label_encoder_output_path = f"{label_encoder_output.path}.pkl"
      with open(label_encoder_output_path, 'wb') as f:
          pickle.dump(label_encoder, f)

      # Track the end time and calculate duration
      end_time = datetime.now()
      duration = (end_time - start_time).total_seconds() / 60  # Duration in minutes

      # Send the Slack message with execution details
      send_slack_message(
          component_name="Model Training Component",
          execution_date=end_time.strftime('%Y-%m-%d'),
          execution_time=end_time.strftime('%H:%M:%S'),
          duration=round(duration, 2)  # Round duration to 2 decimal places
      )

    except Exception as e:
        error_message = str(e)
        send_slack_message(
            component_name="Model Training Component",
            execution_date=datetime.now().strftime('%Y-%m-%d'),
            execution_time=datetime.now().strftime('%H:%M:%S'),
            duration=0  # If failed, duration is 0
        )


#--------------------------------------------------------------------------------------------------------------------------

# Testing Component
@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        'google-cloud-storage==2.18.2',
        'pandas==1.5.3',
        'numpy==1.23.5',
        'scikit-learn==1.2.2',
        'xgboost==1.6.1',
        'google-cloud-aiplatform==1.18.3'
        ]
)
def test_xgboost_model(
    val_data: Input[Dataset],
    model_input: Input[Model],
    vectorizer_input: Input[Artifact],
    label_encoder_input: Input[Artifact],  # New input for the LabelEncoder
    feature_name: str,
    label_name: str
) -> float:
    import pandas as pd
    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score
    import pickle
    import xgboost as xgb
    import os
    import time
    import requests
    from datetime import datetime
    import google.cloud.aiplatform as aiplatform


    # Function to send custom Slack message with Kubeflow component details
    def send_slack_message(component_name, execution_date, execution_time, duration, f1_score=None, precision=None, recall=None):
    # Get the Slack webhook URL from environment variables
        SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
        if not SLACK_WEBHOOK_URL:
            print("Error: SLACK_WEBHOOK_URL not found in environment variables.")
            message = {
                "attachments": [
                    {
                        "color": "#36a64f",  # Green color for success
                        "pretext": ":large_green_circle: Kubeflow Component Success Alert",
                        "fields": [
                            {
                                "title": "Component Name",
                                "value": component_name,
                                "short": True
                            },
                            {
                                "title": "Execution Date",
                                "value": execution_date,
                                "short": True
                            },
                            {
                                "title": "Execution Time",
                                "value": execution_time,
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

            if f1_score is not None:
                message["attachments"][0]["fields"].append({
                    "title": "Validation F1 Score",
                    "value": f"{f1_score:.4f}",
                    "short": True
                })

            if precision is not None:
                message["attachments"][0]["fields"].append({
                    "title": "Validation Precision",
                    "value": f"{precision:.4f}",
                    "short": True
                })

            if recall is not None:
                message["attachments"][0]["fields"].append({
                    "title": "Validation Recall",
                    "value": f"{recall:.4f}",
                    "short": True
                })

            try:
                response = requests.post(SLACK_WEBHOOK_URL, json=message)
                response.raise_for_status()  # Check for request errors
            except requests.exceptions.RequestException as e:
                pass

    try:
      # Track the start time of the component execution
      start_time = datetime.now()

      # Load the trained model
      model_path = os.path.join(model_input.path, "model.bst")
      xgb_model = xgb.XGBClassifier()
      xgb_model.load_model(model_path)

      # Load the TF-IDF vectorizer
      vectorizer_path = f"{vectorizer_input.path}.pkl"
      with open(vectorizer_path, 'rb') as f:
          tfidf_vectorizer = pickle.load(f)

      # Load the LabelEncoder
      label_encoder_path = f"{label_encoder_input.path}.pkl"
      with open(label_encoder_path, 'rb') as f:
          label_encoder = pickle.load(f)

      # Load validation dataset from val_data input artifact
      data = pd.read_pickle(val_data.path)
      X_val = data[feature_name].fillna("")
      y_val = data[label_name].fillna("")

      # Transform validation data with vectorizer
      X_val_tfidf = tfidf_vectorizer.transform(X_val)

      # Encode validation labels using the loaded label encoder
      y_val_encoded = label_encoder.transform(y_val)

      # Make predictions
      y_pred_encoded = xgb_model.predict(X_val_tfidf)

      # Decode predictions back to the original label format
      y_pred = label_encoder.inverse_transform(y_pred_encoded)


      aiplatform.init(project="bilingualcomplaint-system", location="us-east1", experiment='Bilingial-Complaint-Experiment-Tracking')
      run = aiplatform.start_run("run-{}".format(int(time.time())))

      # Calculate metrics
      f1 = f1_score(y_val_encoded, y_pred_encoded, average="macro")
      precision = precision_score(y_val_encoded, y_pred_encoded, average="macro")
      recall = recall_score(y_val_encoded, y_pred_encoded, average="macro")
      metrics = {}
      metrics["f1"] = f1
      metrics["precision"] = precision
      metrics["recall"] = recall

      aiplatform.log_metrics(metrics)
      run.end_run()

      print(f"Validation F1 Score: {f1:.4f}")
      print(f"Validation Precision: {precision:.4f}")
      print(f"Validation Recall: {recall:.4f}")

      # Track the end time and calculate duration
      end_time = datetime.now()
      duration = (end_time - start_time).total_seconds() / 60  # Duration in minutes

      # Send the Slack message with execution details and metrics
      send_slack_message(
          component_name="Model Testing Component",
          execution_date=end_time.strftime('%Y-%m-%d'),
          execution_time=end_time.strftime('%H:%M:%S'),
          duration=round(duration, 2),
          f1_score=f1,
          precision=precision,
          recall=recall
      )

    except Exception as e:
      error_message = str(e)
      print(f"Error during model testing: {error_message}")
      send_slack_message(
          component_name="Model Testing Component Failed",
          execution_date=datetime.now().strftime('%Y-%m-%d'),
          execution_time=datetime.now().strftime('%H:%M:%S'),
          duration= 0  # If failed, duration is 0
      )
      raise e


    # # 2nd try-except block for **BigQuery Insertion**
    # try:
    #     project_id = "bilingualcomplaint-system"
    #     current_timestamp = datetime.utcnow().isoformat()  # Get the current timestamp

    #     # if not f1:
    #     #     f1_score = 0.0

    #     rows_to_insert = [
    #         {
    #             "last_training_timestamp": current_timestamp,
    #             "f1_score": float(f1)
    #         }
    #     ]

    #     # Initialize BigQuery client
    #     bqclient = bigquery.Client(project=project_id)

    #     # Define the BigQuery table name
    #     metadata_table = f"{project_id}.MLOps.metrics"

    #     # Insert metrics into BigQuery table
    #     errors = bqclient.insert_rows_json(metadata_table, rows_to_insert)

    #     if errors:
    #         print(f"Error inserting data: {errors}")
    #     else:
    #         print(f"Data inserted successfully into {metadata_table}")

    # except Exception as e:
    #     print(f"Error during BigQuery insertion: {str(e)}")
    #     raise e  # Re-raise to propagate the failure

    return f1

#--------------------------------------------------------------------------------------------------------------------------


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

    # Check if a model with the same display name already exists
    existing_models = aiplatform.Model.list(
        filter=f'display_name="{model_display_name}"',
        order_by='create_time desc',
        project=project_id,
        location=location
    )

    if existing_models:
        # Get the first existing model's resource name (ID) to add a new version under it
        parent_model = existing_models[0]
        model_id = parent_model.resource_name

        # Register the model under the existing model ID
        registered_model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=model_uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-5:latest",
            parent_model=model_id  # Use the existing model's ID to create a new version
        )
    else:
        # No existing model, create a new one
        registered_model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=model_uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-5:latest"
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
    import os
    import time
    from google.cloud import aiplatform, bigquery
    import requests
    from datetime import datetime
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    import smtplib

    # Function to send success email
    def send_success_email():
        sender_email = "sucessemailtrigger@gmail.com"
        password = "jomnpxbfunwjgitb"
        receiver_emails = ["hegde.anir@northeastern.edu",
                          "nenavath.r@northeastern.edu",
                          "pandey.raj@northeastern.edu",
                          "khatri.say@northeastern.edu",
                          "singh.arc@northeastern.edu",
                          "goparaju.v@northeastern.edu"]


        # Current time for logging purposes
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Create the email content
        subject = '[Kubeflow Pipeline] - Completed'
        body = f'''Hi team,

        Model has been deployed!

        Details:
        - Start Time: {current_time}


        '''

        try:
            # Set up the SMTP server
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()  # Secure the connection
            server.login(sender_email, password)

            # Send email to each receiver
            for receiver_email in receiver_emails:
                # Create a fresh message for each recipient
                message = MIMEMultipart()
                message['From'] = sender_email
                message['To'] = receiver_email
                message['Subject'] = subject
                message.attach(MIMEText(body, 'plain'))

                # Send the email
                server.sendmail(sender_email, receiver_email, message.as_string())

        except Exception as e:
            pass
        finally:
            server.quit()

    # Function to send failure email
    def send_failure_email(error_message):
        sender_email = "sucessemailtrigger@gmail.com"
        password = "jomnpxbfunwjgitb"
        receiver_emails = ["hegde.anir@northeastern.edu",
                          "nenavath.r@northeastern.edu",
                          "pandey.raj@northeastern.edu",
                          "khatri.say@northeastern.edu",
                          "singh.arc@northeastern.edu",
                          "goparaju.v@northeastern.edu"]

        # Create the email content
        subject = '[Kubeflow Pipeline]'
        body = f'Hi team,\nModel deployment has failed!.\nError Details: {error_message}'

        try:
            # Set up the SMTP server
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()  # Secure the connection
            server.login(sender_email, password)

            # Send email to each receiver
            for receiver_email in receiver_emails:
                # Create a fresh message for each recipient
                message = MIMEMultipart()
                message['From'] = sender_email
                message['To'] = receiver_email
                message['Subject'] = subject
                message.attach(MIMEText(body, 'plain'))

                # Send the email
                server.sendmail(sender_email, receiver_email, message.as_string())

        except Exception as e:
            pass
        finally:
            server.quit()

    # Function to send custom Slack message with Kubeflow component details
    def send_slack_message(component_name, execution_date, execution_time, duration, f1_score=None, precision=None, recall=None):
    # Get the Slack webhook URL from environment variables
        SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
        if not SLACK_WEBHOOK_URL:
            print("Error: SLACK_WEBHOOK_URL not found in environment variables.") # Replace with your Slack webhook URL
            message = {
                "attachments": [
                    {
                        "color": "#36a64f",  # Green color for success
                        "pretext": ":large_green_circle: Kubeflow Component Alert",
                        "fields": [
                            {
                                "title": "Component Name",
                                "value": component_name,
                                "short": True
                            },
                            {
                                "title": "Execution Date",
                                "value": execution_date,
                                "short": True
                            },
                            {
                                "title": "Execution Time",
                                "value": execution_time,
                                "short": True
                            },
                            {
                                "title": "Duration",
                                "value": f"{duration} minutes",
                                "short": True
                            },
                            {
                                "title": "Deployed Endpoint",
                                "value": endpoint_name,
                                "short": True
                            }
                        ]
                    }
                ]
            }

            try:
                response = requests.post(SLACK_WEBHOOK_URL, json=message)
                response.raise_for_status()  # Check for request errors
            except requests.exceptions.RequestException as e:
                pass

    # Track the start time of the component execution
    start_time = datetime.now()
    # Delay to allow for model registration completion
    time.sleep(35)

    try:
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

      # Deploy the model to the endpoint with 100% traffic
      deployed_model_resource = endpoint_obj.deploy(
          model=deployed_model,
          deployed_model_display_name=deployed_model_display_name,
          machine_type="n1-standard-4",
          traffic_split={"0": 100},  # Assign 100% traffic to the new deployment
      )

      # Ensure that deployed_model_resource is not None before accessing its ID
      if deployed_model_resource is not None and hasattr(deployed_model_resource, "id"):
          # Retrieve the current traffic allocation and set traffic of old versions to 0%
          traffic_split = {deployed_model_resource.id: 100}  # New model gets 100% traffic
          for deployed_model_id in endpoint_obj.traffic_split.keys():
              if deployed_model_id != deployed_model_resource.id:
                  traffic_split[deployed_model_id] = 0  # Set old versions to 0% traffic

          # Update the endpoint's traffic split
          endpoint_obj.update(traffic_split=traffic_split)
      else:
          print("Warning: Deployed model resource is None or lacks an ID attribute.")

      # Output the endpoint resource name
      endpoint.uri = endpoint_obj.resource_name

      # Track the end time and calculate duration
      end_time = datetime.now()
      duration = (end_time - start_time).total_seconds() / 60  # Duration in minutes

      send_success_email()
      # Send Slack and success email notifications
      send_slack_message(
          component_name="Model Deployment Component",
          execution_date=end_time.strftime('%Y-%m-%d'),
          execution_time=end_time.strftime('%H:%M:%S'),
          duration=round(duration, 2),
          endpoint_name=endpoint_display_name
      )

    except Exception as e:
      # Send failure Slack message and email in case of an error
      error_message = str(e)
      send_failure_email(error_message)
      send_slack_message(
          component_name="Model Deployment Component Failed",
          execution_date=datetime.now().strftime('%Y-%m-%d'),
          execution_time=datetime.now().strftime('%H:%M:%S'),
          duration=0,  # If failed, duration is 0
          endpoint_name=endpoint_display_name
      )



    # Separate block for BigQuery insertion (not part of the try block for deployment)
    try:
        project_id = "bilingualcomplaint-system"
        metadata_table = "bilingualcomplaint-system.MLOps.model_training_metadata"
        preprocessed_data_table = "bilingualcomplaint-system.MLOps.preprocessed_data"
        # Initialize BigQuery client
        bqclient = bigquery.Client(project=project_id)

        # Define metadata insertion details
        current_timestamp = datetime.utcnow().isoformat()

        # Query to get the record count from the preprocessed_data table
        query = f"SELECT COUNT(*) AS record_count FROM `{preprocessed_data_table}`"
        query_job = bqclient.query(query)

        # Fetch the query result and process it directly
        result = query_job.result()  # This returns a list of Row objects
        record_count = None

        # Iterate over the results and extract the record count
        for row in result:
            record_count = int(row["record_count"])

        if record_count is None:
            raise ValueError("No record count returned from BigQuery query.")

        # Prepare data for metadata insertion
        rows_to_insert = [
            {
                "last_training_timestamp": current_timestamp,
                "record_count": record_count
            }
        ]

        # Insert metadata into BigQuery table
        metadata_table = f"{project_id}.MLOps.model_training_metadata"
        errors = bqclient.insert_rows_json(metadata_table, rows_to_insert)
        print("hi")

        if errors:
            print(f"Failed to insert metadata: {errors}")
        else:
            print(f"Metadata inserted successfully into {metadata_table}")

    except Exception as e:
        print(f"Error inserting metadata into BigQuery: {str(e)}")
#-------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------
# Bias detection component


@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        'google-cloud-storage==2.18.2',
        'pandas==1.5.3',
        'numpy==1.23.5',
        'scikit-learn==1.2.2',
        'xgboost==1.6.1',
        'fairlearn==0.8.0'
    ]
)
def bias_detection(
    train_data: Input[Dataset],
    model_input: Input[Model],
    vectorizer_input: Input[Artifact],
    label_encoder_input: Input[Artifact],
    feature_name: str,
    label_name: str,
):
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from fairlearn.metrics import MetricFrame
    import xgboost as xgb
    import os
    import pickle
    import requests
    from datetime import datetime

    # Function to send custom Slack message with Kubeflow component details
    def send_slack_message(component_name, execution_date, execution_time, duration, f1_score=None, precision=None, recall=None):
    # Get the Slack webhook URL from environment variables
        SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
        if not SLACK_WEBHOOK_URL:
            print("Error: SLACK_WEBHOOK_URL not found in environment variables.")
            message = {
                "attachments": [
                    {
                        "color": "#36a64f",  # Green color for success
                        "pretext": ":large_green_circle: Kubeflow Bias Component Check Alert",
                        "fields": [
                            {
                                "title": "Component Name",
                                "value": component_name,
                                "short": True
                            },
                            {
                                "title": "Execution Date",
                                "value": execution_date,
                                "short": True
                            },
                            {
                                "title": "Execution Time",
                                "value": execution_time,
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

            if alerts:
                message["attachments"][0]["pretext"] = ":warning: Bias Detected"
                for alert in alerts:
                    message["attachments"][0]["fields"].append({
                        "title": "Bias Alert",
                        "value": alert,
                        "short": False
                    })

            elif no_bias_message:
                message["attachments"][0]["pretext"] = ":white_check_mark: No Bias Detected"
                message["attachments"][0]["fields"].append({
                    "title": "Bias Check Status",
                    "value": "Everything is fine. No bias detected across any slices.",
                    "short": False
                })

            if slice_results is not None:
                message["attachments"][0]["fields"].append({
                    "title": "Slice Accuracy Results",
                    "value": slice_results,
                    "short": False
                })

            try:
                response = requests.post(SLACK_WEBHOOK_URL, json=message)
                response.raise_for_status()  # Check for request errors
            except requests.exceptions.RequestException as e:
                pass

    try:
        sensitive_features = 'product'  # We are now only checking for the 'product' feature
        bias_threshold = 0.99  # Example threshold for bias detection
        # Track the start time of the component execution
        start_time = datetime.now()

        # Load the trained model
        model_path = os.path.join(model_input.path, "model.bst")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(model_path)

        # Load the TF-IDF vectorizer
        vectorizer_path = f"{vectorizer_input.path}.pkl"
        with open(vectorizer_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        # Load the LabelEncoder
        label_encoder_path = f"{label_encoder_input.path}.pkl"
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        # Load the dataset from train_data input artifact
        data = pd.read_pickle(train_data.path)
        X = data[feature_name].fillna("")  # Extract the features
        y = data[label_name].fillna("")   # Extract the label

        # Transform validation data with vectorizer
        X_tfidf = tfidf_vectorizer.transform(X)

        # Make predictions using the loaded model
        y_pred_encoded = xgb_model.predict(X_tfidf)

        # Decode predictions back to the original label format
        y_pred = label_encoder.inverse_transform(y_pred_encoded)

        # Bias detection using Fairlearn's MetricFrame
        metric_frame = MetricFrame(
            metrics=accuracy_score,
            y_true=y,
            y_pred=y_pred,
            sensitive_features=data[sensitive_features]  # We are checking for the 'product' feature
        )

        # Calculate slice-specific metrics
        slice_metrics = metric_frame.by_group
        valid_slices = slice_metrics.dropna()

        # Convert the slice metrics into a readable format for Slack
        slice_results = valid_slices.reset_index()
        slice_results.columns = ['Product', 'Accuracy']  # Ensuring column names are correct for Slack
        slice_results_str = slice_results.to_string(index=False)

        # Function to check for bias and trigger an alert if threshold is crossed
        def check_for_bias(slice_metrics, threshold):
            alert_flag = False
            alerts = []

            for slice_name, accuracy in slice_metrics.items():
                if accuracy < threshold:
                    alert_flag = True
                    alerts.append(f"Bias Alert: Accuracy for slice '{slice_name}' is below threshold: {accuracy:.2f}. We are removing these classes for mitigation.")

            if alert_flag:
                return alerts
            else:
                return None

        # Check for bias and trigger alerts
        bias_alerts = check_for_bias(valid_slices, bias_threshold)

        # Track the end time and calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60  # Duration in minutes

        # Send Slack message with execution details and bias alerts if any
        if bias_alerts:
            send_slack_message(
                component_name="Bias Detection Component",
                execution_date=end_time.strftime('%Y-%m-%d'),
                execution_time=end_time.strftime('%H:%M:%S'),
                duration=round(duration, 2),
                alerts=bias_alerts,
                slice_results=slice_results_str  # Include the slice results in the Slack message
            )
        else:
            send_slack_message(
                component_name="Bias Detection Component",
                execution_date=end_time.strftime('%Y-%m-%d'),
                execution_time=end_time.strftime('%H:%M:%S'),
                duration=round(duration, 2),
                no_bias_message=True
            )

    except Exception as e:
        error_message = str(e)
        print(f"Error during bias detection: {error_message}")
        send_slack_message(
            component_name="Bias Detection Component Failed",
            execution_date=datetime.now().strftime('%Y-%m-%d'),
            execution_time=datetime.now().strftime('%H:%M:%S'),
            duration=0  # If failed, duration is 0
        )
        raise e





# ---------------------------------------------------------------------------------------------------------------------

# Second Component - Training Script
#Training Component
@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        'google-cloud-storage==2.18.2',
        'pandas==1.5.3',
        'numpy==1.23.5',
        'scikit-learn==1.2.2',
    ]
)
def train_naive_bayes_model(
    train_data: Input[Dataset],
    feature_name: str,
    label_name: str,
    model: Output[Model],
    vectorizer_output: Output[Artifact],
    label_encoder_output: Output[Artifact]  # New output for the LabelEncoder
):
    import os
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import LabelEncoder
    import pickle
    import requests
    from datetime import datetime

    # Track the start time of the component execution
    start_time = datetime.now()

    # Function to send custom Slack message with Kubeflow component details
    def send_slack_message(component_name, execution_date, execution_time, duration, f1_score=None, precision=None, recall=None):
    # Get the Slack webhook URL from environment variables
        SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
        if not SLACK_WEBHOOK_URL:
            print("Error: SLACK_WEBHOOK_URL not found in environment variables.")  # Replace with your Slack webhook URL
            message = {
                "attachments": [
                    {
                        "color": "#36a64f",  # Green color for success
                        "pretext": ":large_green_circle: Kubeflow Component Success Alert",
                        "fields": [
                            {
                                "title": "Component Name",
                                "value": component_name,
                                "short": True
                            },
                            {
                                "title": "Execution Date",
                                "value": execution_date,
                                "short": True
                            },
                            {
                                "title": "Execution Time",
                                "value": execution_time,
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
                response = requests.post(SLACK_WEBHOOK_URL, json=message)
                response.raise_for_status()  # Check for request errors
                pass
            except requests.exceptions.RequestException as e:
                pass
    try:
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

      mnb = MultinomialNB(
          alpha=0.8,
          fit_prior=True,
          force_alpha=True
          )
      mnb.fit(X_tfidf, y_encoded)

      # Save the model as 'naive_bayes_model.pkl'
      model_path = f"{model.path}.pkl"
      with open(model_path, 'wb') as f:
          pickle.dump(mnb, f)

      # Save the vectorizer as 'tfidf_vectorizer.pkl' directly to the vectorizer_output path
      vectorizer_output_path = f"{vectorizer_output.path}.pkl"
      with open(vectorizer_output_path, 'wb') as f:
          pickle.dump(tfidf_vectorizer, f)

      # Save the label encoder as 'label_encoder.pkl'
      label_encoder_output_path = f"{label_encoder_output.path}.pkl"
      with open(label_encoder_output_path, 'wb') as f:
          pickle.dump(label_encoder, f)

      # Track the end time and calculate duration
      end_time = datetime.now()
      duration = (end_time - start_time).total_seconds() / 60  # Duration in minutes

      # Send the Slack message with execution details
      send_slack_message(
          component_name="Model Training Component (Naive Bayes Model Training)",
          execution_date=end_time.strftime('%Y-%m-%d'),
          execution_time=end_time.strftime('%H:%M:%S'),
          duration=round(duration, 2)  # Round duration to 2 decimal places
      )

    except Exception as e:
        error_message = str(e)
        send_slack_message(
            component_name="Model Training Component (Naive Bayes Model Training)",
            execution_date=datetime.now().strftime('%Y-%m-%d'),
            execution_time=datetime.now().strftime('%H:%M:%S'),
            duration=0  # If failed, duration is 0
        )

@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        'google-cloud-storage==2.18.2',
        'pandas==1.5.3',
        'numpy==1.23.5',
        'scikit-learn==1.2.2',
        'google-cloud-aiplatform==1.18.3'
    ]
)
def test_naive_bayes_model(
    val_data: Input[Dataset],
    model_input: Input[Model],
    vectorizer_input: Input[Artifact],
    label_encoder_input: Input[Artifact],
    feature_name: str,
    label_name: str
) -> float:
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score
    import pickle
    from datetime import datetime
    # from google.cloud import bigquery
    import requests
    import time
    import os
    import google.cloud.aiplatform as aiplatform

    # Function to send custom Slack message with Kubeflow component details
    def send_slack_message(component_name, execution_date, execution_time, duration, f1_score=None, precision=None, recall=None):
    # Get the Slack webhook URL from environment variables
        SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
        if not SLACK_WEBHOOK_URL:
            print("Error: SLACK_WEBHOOK_URL not found in environment variables.")
            message = {
                "attachments": [
                    {
                        "color": "#36a64f",  # Green color for success
                        "pretext": ":large_green_circle: Kubeflow Component Success Alert",
                        "fields": [
                            {"title": "Component Name", "value": component_name, "short": True},
                            {"title": "Execution Date", "value": execution_date, "short": True},
                            {"title": "Execution Time", "value": execution_time, "short": True},
                            {"title": "Duration", "value": f"{duration} minutes", "short": True}
                        ]
                    }
                ]
            }

            if f1_score is not None:
                message["attachments"][0]["fields"].append({"title": "Validation F1 Score", "value": f"{f1_score:.4f}", "short": True})
            if precision is not None:
                message["attachments"][0]["fields"].append({"title": "Validation Precision", "value": f"{precision:.4f}", "short": True})
            if recall is not None:
                message["attachments"][0]["fields"].append({"title": "Validation Recall", "value": f"{recall:.4f}", "short": True})

            try:
                response = requests.post(SLACK_WEBHOOK_URL, json=message)
                response.raise_for_status()  # Check for request errors
            except requests.exceptions.RequestException as e:
                print(f"Error sending Slack message: {e}")

    try:
        # Track the start time of the component execution
        start_time = datetime.now()

        # Load the trained model
        with open(f'{model_input.path}.pkl', 'rb') as f:
            mnb = pickle.load(f)

        # Load the TF-IDF vectorizer
        with open(f'{vectorizer_input.path}.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        # Load the LabelEncoder
        with open(f'{label_encoder_input.path}.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        # Load validation dataset from val_data input artifact
        data = pd.read_pickle(val_data.path)
        X_val = data[feature_name].fillna("")
        y_val = data[label_name].fillna("")

        # Transform validation data with vectorizer
        X_val_tfidf = tfidf_vectorizer.transform(X_val)

        # Encode validation labels using the loaded label encoder
        y_val_encoded = label_encoder.transform(y_val)

        # Make predictions
        y_pred_encoded = mnb.predict(X_val_tfidf)

        # Decode predictions back to the original label format
        y_pred = label_encoder.inverse_transform(y_pred_encoded)


        aiplatform.init(project="bilingualcomplaint-system", location="us-east1", experiment='Bilingial-Complaint-Experiment-Tracking')
        run = aiplatform.start_run("run-{}".format(int(time.time())))

        # Calculate metrics
        f1 = f1_score(y_val_encoded, y_pred_encoded, average="macro")
        precision = precision_score(y_val_encoded, y_pred_encoded, average="macro")
        recall = recall_score(y_val_encoded, y_pred_encoded, average="macro")
        metrics = {}
        metrics["f1"] = f1
        metrics["precision"] = precision
        metrics["recall"] = recall

        aiplatform.log_metrics(metrics)
        run.end_run()

        print(f"Validation F1 Score: {f1:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")

        # Track the end time and calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60  # Duration in minutes

        # Send the Slack message with execution details and metrics
        send_slack_message(
            component_name="Model Testing Component Testing (Naive Bayes Model Testing)",
            execution_date=end_time.strftime('%Y-%m-%d'),
            execution_time=end_time.strftime('%H:%M:%S'),
            duration=round(duration, 2),
            f1_score=f1,
            precision=precision,
            recall=recall
        )

    except Exception as e:
        error_message = str(e)
        print(f"Error during model testing: {error_message}")
        send_slack_message(
            component_name="Model Testing Component Failed (Naive Bayes Model Testing)",
            execution_date=datetime.now().strftime('%Y-%m-%d'),
            execution_time=datetime.now().strftime('%H:%M:%S'),
            duration=0  # If failed, duration is 0
        )
        raise e

    # # BigQuery Insertion
    # try:
    #     project_id = "bilingualcomplaint-system"
    #     current_timestamp = datetime.utcnow().isoformat()  # Get the current timestamp

    #     rows_to_insert = [
    #         {
    #             "last_training_timestamp": current_timestamp,
    #             "f1_score": float(f1)
    #         }
    #     ]

    #     # Initialize BigQuery client
    #     bqclient = bigquery.Client(project=project_id)

    #     # Define the BigQuery table name
    #     metadata_table = f"{project_id}.MLOps.metrics"

    #     # Insert metrics into BigQuery table
    #     errors = bqclient.insert_rows_json(metadata_table, rows_to_insert)

    #     if errors:
    #         print(f"Error inserting data: {errors}")
    #     else:
    #         print(f"Data inserted successfully into {metadata_table}")

    # except Exception as e:
    #     print(f"Error during BigQuery insertion: {str(e)}")
    #     raise e  # Re-raise to propagate the failure

    return f1

#----------------------------------------------------------------------------------------------------------------------
# Select Best Model
@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas==1.5.3", "numpy==1.23.5"]
)
def select_best_model(
    xgboost_f1: float,
    naive_bayes_f1: float,
    xgboost_model: Input[Model],
    naive_bayes_model: Input[Model],
    best_model: Output[Model]
) -> str:
    """
    Compare the F1 scores of XGBoost and Naive Bayes models and select the best model.
    """
    import shutil
    import os

    # Log the F1 scores for debugging
    print(f"XGBoost F1 Score: {xgboost_f1}")
    print(f"Naive Bayes F1 Score: {naive_bayes_f1}")

    # Select the best model
    if xgboost_f1 >= naive_bayes_f1:
        print("XGBoost model is selected as the best model.")
        shutil.copytree(xgboost_model.path, best_model.path)
        selected_model = "XGBoost"
    else:
        print("Naive Bayes model is selected as the best model.")
        shutil.copytree(naive_bayes_model.path, best_model.path)
        selected_model = "Naive Bayes"

    # Return the name of the selected model for tracking purposes
    return selected_model

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

@pipeline(
    name=f"model_pipeline_{TIMESTAMP}",
    description="Model data pipeline - Training | Testing | Model Selection | Registration | Deployment",
    pipeline_root=_pipeline_artifacts_dir,
)
def model_data_pipeline(
    start_year: int = 2018,
    end_year: int = 2020,
    limit: int = 100,
    feature_name: str = 'complaint_english',
    label_name: str = 'product',
    model_display_name: str = "best-complaints-model",
    endpoint_display_name: str = "best-complaints-endpoint",
    deployed_model_display_name: str = "best-complaints-deployment"
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

    # Train models
    train_xgboost_task = train_xgboost_model(
        train_data=get_data_component_task.outputs['train_data'],
        feature_name=feature_name,
        label_name=label_name
    )

    train_naive_bayes_task = train_naive_bayes_model(
        train_data=get_data_component_task.outputs['train_data'],
        feature_name=feature_name,
        label_name=label_name
    )

    # Test models
    test_xgboost_task = test_xgboost_model(
        val_data=get_data_component_task.outputs['val_data'],
        model_input=train_xgboost_task.outputs["model"],
        vectorizer_input=train_xgboost_task.outputs["vectorizer_output"],
        label_encoder_input=train_xgboost_task.outputs["label_encoder_output"],
        feature_name=feature_name,
        label_name=label_name
    )

    test_naive_bayes_task = test_naive_bayes_model(
        val_data=get_data_component_task.outputs['val_data'],
        model_input=train_naive_bayes_task.outputs["model"],
        vectorizer_input=train_naive_bayes_task.outputs["vectorizer_output"],
        label_encoder_input=train_naive_bayes_task.outputs["label_encoder_output"],
        feature_name=feature_name,
        label_name=label_name
    )

    # Select the best model based on F1 score
    select_best_model_task = select_best_model(
        xgboost_f1=test_xgboost_task.output,
        naive_bayes_f1=test_naive_bayes_task.output,
        xgboost_model=train_xgboost_task.outputs["model"],
        naive_bayes_model=train_naive_bayes_task.outputs["model"]
    )



    # Detect Bias
    bias_detection_task = bias_detection(
        train_data=get_data_component_task.outputs['train_data'],
        model_input=select_best_model_task.outputs["best_model"],
        vectorizer_input=train_xgboost_task.outputs["vectorizer_output"],
        label_encoder_input=train_xgboost_task.outputs["label_encoder_output"],
        feature_name=feature_name,
        label_name=label_name
    )

    bias_detection_task.after(select_best_model_task)


    # Register the selected model
    model_registration_task = model_registration(
        model_output=select_best_model_task.outputs["best_model"],
        project_id=PROJECT_ID,
        location=LOCATION,
        model_display_name=model_display_name
    )

    model_registration_task.after(bias_detection_task)
    # Deploy the registered model
    model_deployment_task = model_deployment(
        model=model_registration_task.outputs["model"],
        project_id=PROJECT_ID,
        location=LOCATION,
        endpoint_display_name=endpoint_display_name,
        deployed_model_display_name=deployed_model_display_name
    )
import os
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
        "start_year": 2017,
        "end_year": 2020,
        "limit": 200,
        "feature_name": "complaint_english",
        "label_name": "product"
    }
)
job.submit()
