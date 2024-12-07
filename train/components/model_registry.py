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
            artifact_uri=model_artifact.uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",
            parent_model=model_id,
        )
    else:
        registered_model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=model_artifact.uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",
        )

    registered_model_artifact.uri = registered_model.resource_name