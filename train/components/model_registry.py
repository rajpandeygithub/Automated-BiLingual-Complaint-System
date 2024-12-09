from kfp.dsl import component, Input, Output, Model

@component(
    packages_to_install=["google-cloud-aiplatform", "google-auth"]
)
def register_model_component(
    model_artifact: Input[Model],
    project_id: str,
    location: str,
    model_display_name: str,
    registered_model_artifact: Output[Model],
    slack_url: str = None,
):
    import requests
    from datetime import datetime
    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=location)

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
               webhook_url=slack_url, message_str=f'KubeFlow Component: Model Registeration | Registration Started',
               execution_date=start_time.date(), execution_time=start_time.time(), 
               duration=(datetime.now() - start_time).total_seconds() / 60, is_success=True
               )

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
            artifact_uri=model_artifact.uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",
            parent_model=model_id,
        )
        if slack_url:
             send_slack_message(
                  webhook_url=slack_url, message_str=f'KubeFlow Component: Model Registeration | Registering a New Version of {model_display_name}',
                  execution_date=start_time.date(), execution_time=start_time.time(),
                  duration=(datetime.now() - start_time).total_seconds() / 60, is_success=True
                  )
    else:
        registered_model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=model_artifact.uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",
        )
        if slack_url:
            send_slack_message(
                webhook_url=slack_url, message_str=f'KubeFlow Component: Model Registeration | Registering a New Model: {model_display_name}',
                execution_date=start_time.date(), execution_time=start_time.time(), 
                duration=(datetime.now() - start_time).total_seconds() / 60, is_success=True
                )

    registered_model_artifact.uri = registered_model.resource_name