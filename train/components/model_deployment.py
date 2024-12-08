from kfp.dsl import component, Input, Output, Model, Artifact

@component(
    packages_to_install=["google-cloud-aiplatform"]
)
def deploy_model_component(
    model: Input[Model],
    project_id: str,
    location: str,
    endpoint_display_name: str,
    deployed_model_display_name: str,
    endpoint: Output[Artifact],
    endpoint_machine_type: str,
    minimum_replica_count: int,
    maximum_replica_count: int,
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

    if slack_url:
        send_slack_message(
            webhook_url=slack_url, message_str=f'KubeFlow Component: Model Deployment | Started Deploying: {endpoint_display_name}',
            execution_date=start_time.date(), execution_time=start_time.time(),
            duration=(datetime.now() - start_time).total_seconds() / 60, is_success=True
                  )

    endpoint_obj.deploy(
        model=aiplatform.Model(model.uri),
        deployed_model_display_name=deployed_model_display_name,
        machine_type=endpoint_machine_type,
        traffic_split={"0": 100},
        min_replica_count=minimum_replica_count,
        max_replica_count=maximum_replica_count,
    )

    if slack_url:
        send_slack_message(
            webhook_url=slack_url, message_str=f'KubeFlow Component: Model Deployment | Deployment Successful for {endpoint_display_name}',
            execution_date=start_time.date(), execution_time=start_time.time(),
            duration=(datetime.now() - start_time).total_seconds() / 60, is_success=True
            )

    endpoint.uri = endpoint_obj.resource_name