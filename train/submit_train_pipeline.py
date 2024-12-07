import yaml
import argparse
from typing import Callable, Dict
from datetime import datetime
from kfp import dsl, compiler
from google.cloud import aiplatform
from components.get_data import get_data_component
from components.prepare_data import prepare_data_component
from components.hf_model_train import train_huggingface_model_component
from components.hf_model_test import test_huggingface_model_component
from components.bias_detection import detect_bias_component
from components.model_registry import register_model_component
from components.model_deployment import deploy_model_component


def get_training_pipeline(
    pipeline_name: str, 
    description: str, 
    pipeline_root: str,
    project_params: Dict,
    data_params: Dict,
    model_params: Dict,
    label_2_idx_map: Dict,
    training_params: Dict, 
    bias_detection_params: Dict,
    deploy_params: Dict,
    ) -> Callable:

    @dsl.pipeline(
        name=pipeline_name,
        description=description,
        pipeline_root=pipeline_root,
    )
    def training_pipeline():
        get_data_component_task = get_data_component(
            project_id=project_params.get("gcp_project_id"),
            location=project_params.get("gcp_project_location"),
            start_year=data_params.get("start_year"),
            end_year=data_params.get("end_year"),
            label_name=data_params.get("label_column_name"),
            limit=data_params.get("limit", None),
            minimum_label_count=data_params.get("minimum_label_sample_count")
            )

        get_data_component_task.set_cpu_limit('1') 
        get_data_component_task.set_memory_limit('1G')

        train_data_prep_task = prepare_data_component(
            data=get_data_component_task.outputs['train_data'],
            dataset_name='train',
            feature_name='complaints',
            label_name=data_params.get("label_column_name"),
            label_map=label_2_idx_map,
            hugging_face_model_name=model_params.get("model_name"),
            max_sequence_length=model_params.get('max_sequence_length')
            )
        train_data_prep_task.set_display_name(f'Training Data Prep: {model_params.get("model_name")}')
        train_data_prep_task.set_cpu_limit('1')
        train_data_prep_task.set_memory_limit('1G')

        test_data_prep_task = prepare_data_component(
            data=get_data_component_task.outputs['holdout_data'],
            dataset_name='holdout',
            feature_name='complaints',
            label_name=data_params.get("label_column_name"),
            label_map=label_2_idx_map,
            hugging_face_model_name=model_params.get("model_name"),
            max_sequence_length=model_params.get('max_sequence_length')
        )
        test_data_prep_task.set_display_name(f'Holdout Data Prep: {model_params.get("model_name")}')
        test_data_prep_task.set_cpu_limit('1')
        test_data_prep_task.set_memory_limit('1G')

        train_task = train_huggingface_model_component(
            train_data=train_data_prep_task.outputs['tf_dataset'],
            label_map=label_2_idx_map,
            train_data_name='train',
            huggingface_model_name=model_params.get("model_name"),
            max_epochs=training_params.get("epochs"),
            batch_size=training_params.get("batch_size")
            )
        train_task.set_display_name(f'Training: {model_params.get("model_name")}')
        
        test_task = test_huggingface_model_component(
            project_id=project_params.get("gcp_project_id"),
            location=project_params.get("gcp_project_location"),
            test_data=test_data_prep_task.outputs['tf_dataset'],
            model=train_task.outputs['model_output'],
            test_data_name='holdout',
            label_map=label_2_idx_map,
            label_name=data_params.get("label_column_name"),
            huggingface_model_name=model_params.get("model_name"),
            batch_size=training_params.get("batch_size")
        )
        test_task.set_display_name(f'Testing: {model_params.get("model_name")}')

        bias_detection_task = detect_bias_component(
            accuracy_threshold=bias_detection_params.get("accuracy_threshold"),
            test_data=test_data_prep_task.outputs['tf_dataset'],
            model=test_task.outputs['reusable_model'],
            test_data_name='holdout',
            label_map=label_2_idx_map,
            huggingface_model_name=model_params.get("model_name"),
            batch_size=training_params.get("batch_size")
        )
        bias_detection_task.set_display_name(f'Bias Detection Task')
        bias_detection_task.set_cpu_limit('1')
        bias_detection_task.set_memory_limit('1G')
        
        if deploy_params.get("deploy"):
            registry_task = register_model_component(
                model_artifact=test_task.outputs["reusable_model"],
                project_id=project_params.get("gcp_project_id"),
                location=project_params.get("gcp_project_location"),
                model_display_name=f'model-{model_params.get("model_name")}'
                )
            deployment_task = deploy_model_component(
                model=registry_task.outputs['registered_model_artifact'],
                project_id=project_params.get("gcp_project_id"),
                location=project_params.get("gcp_project_location"),
                endpoint_display_name = f'endpoint-{model_params.get("model_name")}',
                deployed_model_display_name = f'deploy-model-{model_params.get("model_name")}',
            )
    
    return training_pipeline



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Submit kubeflow training pipeline job with configuration.")
    parser.add_argument('--config', type=str, required=True, help='Path to the training configuration YAML file.')
    # Parse the arguments
    args = parser.parse_args()

    # Load the YAML configuration file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    project_params = config.get('project', {})
    data_params = config.get('data_params', {})
    model_params = config.get('model_parms', {})
    training_params = config.get('training_params', {})
    bias_detection_params = config.get('bias_detection_params', {})
    deploy_params = config.get('deploy_params', {})

    label_2_idx_map = {label: idx for idx, label in enumerate(data_params.get("unique_label_values"))}
    idx_2_label_map = {idx: label for label, idx in label_2_idx_map.items()}

    pipeline_name = f'{data_params.get("label_column_name")}-{project_params.get("pipeline_name")}'

    aiplatform.init(
        project=project_params.get("gcp_project_id"),
        location=project_params.get("gcp_project_location"),
        staging_bucket=f'gs://{project_params.get("gcp_artifact_bucket")}',
        )
    pipeline_artifacts_dir = f'gs://{project_params.get("gcp_pipeline_artifact_directory")}/{project_params.get("pipeline_name")}'

    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

    training_pipeline = get_training_pipeline(
        pipeline_name=pipeline_name, description=project_params.get('description'), pipeline_root=pipeline_artifacts_dir, 
        project_params=project_params, data_params=data_params, model_params=model_params, training_params=training_params, 
        bias_detection_params=bias_detection_params, 
        label_2_idx_map=label_2_idx_map,
        deploy_params=deploy_params
        )
    compiler.Compiler().compile(
        pipeline_func=training_pipeline, package_path=f"training-pipeline-via-yml-{TIMESTAMP}.json"
        )
    
    job = aiplatform.PipelineJob(
        display_name=f"training-pipeline-via-yml-{TIMESTAMP}",
        template_path=f"training-pipeline-via-yml-{TIMESTAMP}.json",
        job_id=f"training-pipeline-via-yml-{TIMESTAMP}",
        enable_caching=True
    )
    # Submit the Training KubeFlow Pipeline Job to Vertex AI
    job.submit()