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