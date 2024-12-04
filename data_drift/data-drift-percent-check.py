from google.cloud import bigquery, aiplatform
from datetime import datetime

# Constants for your project and dataset
PROJECT_ID = "bilingualcomplaint-system"
REGION = "us-east1"
THRESHOLD = 6
drift_table = "bilingualcomplaint-system.MLOps.drift_records"

def drift_trigger_retraining(request):
    # Initialize BigQuery client
    bqclient = bigquery.Client()

    # Query to count new records
    query = f"""
    SELECT COUNT(*) AS new_record_count
    FROM `{drift_table}`
    """
    query_job = bqclient.query(query)
    new_record_count = query_job.result().to_dataframe()['new_record_count'][0]

    if new_record_count >= THRESHOLD:
        # Initialize Vertex AI SDK
        aiplatform.init(project=PROJECT_ID, location=REGION)

        # Get current timestamp
        TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

        # Define the first pipeline job
        job1 = aiplatform.PipelineJob(
            display_name="model_deployment_pipeline_department",
            template_path="gs://tfx-artifacts/Template/model_deployment_pipeline_department_job.json",
            job_id="model-deployment-department-pipeline-{0}".format(TIMESTAMP),
            enable_caching=True
        )

        # Define the second pipeline job
        job2 = aiplatform.PipelineJob(
            display_name="model_deployment_pipeline_product",
            template_path="gs://tfx-artifacts/Template/model_deployment_pipeline_product_job.json",
            job_id="model-deployment-product-pipeline-{0}".format(TIMESTAMP),
            enable_caching=True
        )

        # Submit both pipeline jobs
        job1.submit()
        job2.submit()
        
        return "Both retraining pipelines triggered.", 200
    else:
        return "Threshold not met. Retraining not triggered.", 200
