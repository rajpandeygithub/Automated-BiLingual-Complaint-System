import yaml
from datetime import datetime
from kfp import dsl, compiler
from google.cloud import aiplatform
from components.get_data import get_data_component
from components.prepare_data import prepare_data_component
from components.hf_model_train import train_huggingface_model_component
from components.hf_model_test import test_huggingface_model_component
from components.bias_detection import detect_bias_component

with open('train-department.yml', 'r') as file:
    config = yaml.safe_load(file)

project_parms = config.get('project', {})
data_params = config.get('data_params', {})
model_params = config.get('model_parms', {})
training_params = config.get('training_params', {})
bias_detection_params = config.get('bias_detection_params', {})

label_2_idx_map = {label: idx for idx, label in enumerate(data_params.get("unique_label_values"))}
idx_2_label_map = {idx: label for label, idx in label_2_idx_map.items()}

pipeline_name = f'{data_params.get("label_column_name")}-{project_parms.get("pipeline_name")}'

aiplatform.init(
    project=project_parms.get("gcp_project_id"),
    location=project_parms.get("gcp_project_location"),
    staging_bucket=f'gs://{project_parms.get("gcp_artifact_bucket")}',
    )
pipeline_artifacts_dir = f'gs://{project_parms.get("gcp_pipeline_artifact_directory")}/{project_parms.get("pipeline_name")}'

@dsl.pipeline(
    name=pipeline_name,
    description=project_parms.get("description"),
    pipeline_root=pipeline_artifacts_dir
)
def training_pipeline():
    get_data_component_task = get_data_component(
        project_id=project_parms.get("gcp_project_id"),
        location=project_parms.get("gcp_project_location"),
        start_year=data_params.get("start_year"),
        end_year=data_params.get("end_year"),
        label_name=data_params.get("label_column_name"),
        limit=data_params.get("limit", None)
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
        project_id=project_parms.get("gcp_project_id"),
        location=project_parms.get("gcp_project_location"),
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

if __name__ == '__main__':
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

    compiler.Compiler().compile(
        pipeline_func=training_pipeline, package_path=f"training-pipeline-via-yml-{TIMESTAMP}.json"
        )
    
    job = aiplatform.PipelineJob(
        display_name=f"training-pipeline-via-yml-{TIMESTAMP}",
        template_path=f"training-pipeline-via-yml-{TIMESTAMP}.json",
        job_id=f"training-pipeline-via-yml-{TIMESTAMP}",
        enable_caching=True
    )
    job.submit()