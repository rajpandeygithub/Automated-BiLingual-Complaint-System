import os
import sys
import yaml
from datetime import datetime
from kfp import compiler, dsl
from google.cloud import aiplatform
from kfp.dsl import component, pipeline, Artifact, Input, Output, Model

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

@component(
    packages_to_install=["google-cloud-aiplatform", "google-auth"]
)
def model_registration(
    model_output_uri: str,
    project_id: str,
    location: str,
    model_display_name: str,
    model: Output[Model],
):
    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=location)

    existing_models = aiplatform.Model.list(
        filter=f'display_name="{model_display_name}"',
        order_by="create_time desc",
        project=project_id,
        location=location,
    )

    if existing_models:
        model_id = existing_models[0].resource_name
        registered_model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=model_output_uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",
            parent_model=model_id,
        )
    else:
        registered_model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=model_output_uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",
        )

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
    endpoint: Output[Artifact],
):
    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=location)

    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        order_by="create_time desc",
        project=project_id,
        location=location,
    )

    if endpoints:
        endpoint_obj = endpoints[0]
    else:
        endpoint_obj = aiplatform.Endpoint.create(display_name=endpoint_display_name)

    endpoint_obj.deploy(
        model=aiplatform.Model(model.uri),
        deployed_model_display_name=deployed_model_display_name,
        machine_type="n1-standard-4",
        traffic_split={"0": 100},
        min_replica_count=1,
        max_replica_count=3,
    )

    endpoint.uri = endpoint_obj.resource_name

@pipeline(
    name="dynamic_model_deployment_pipeline",
    description="Pipeline for dynamic model registration and deployment",
)
def deployment_pipeline(
    model_output_uri: str,
    project_id: str,
    location: str,
    model_display_name: str,
    endpoint_display_name: str,
    deployed_model_display_name: str,
):
    register_model_task = model_registration(
        model_output_uri=model_output_uri,
        project_id=project_id,
        location=location,
        model_display_name=model_display_name,
    )

    model_deployment(
        model=register_model_task.outputs["model"],
        project_id=project_id,
        location=location,
        endpoint_display_name=endpoint_display_name,
        deployed_model_display_name=deployed_model_display_name,
    )

def main():
    if len(sys.argv) != 2:
        print("Usage: python deployment_pipeline.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    aiplatform.init(
        project=config["project_id"],
        location=config["location"],
        staging_bucket=config["staging_bucket"],
    )

    # Generate pipeline JSON filename dynamically
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    pipeline_json_path = f"{config['pipeline_name']}_{TIMESTAMP}.json"

    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=deployment_pipeline,
        package_path=pipeline_json_path,
    )

    # Submit the pipeline job
    pipeline_job = aiplatform.PipelineJob(
        display_name=f"{config['pipeline_name']}_{TIMESTAMP}_model_deployment",
        template_path=pipeline_json_path,  # Use the compiled JSON file path
        job_id=f"dynamic-model-deployment-{TIMESTAMP}",
        enable_caching=True,
        parameter_values={
            "model_output_uri": config["model_output_uri"],
            "project_id": config["project_id"],
            "location": config["location"],
            "model_display_name": config["model_display_name"],
            "endpoint_display_name": config["endpoint_display_name"],
            "deployed_model_display_name": config["deployed_model_display_name"],
        },
    )

    # Submit the job
    pipeline_job.submit()

if __name__ == "__main__":
    main()
