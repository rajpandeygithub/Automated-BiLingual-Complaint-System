from typing import Dict, NamedTuple
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, Artifact

@component(
    base_image="tensorflow/tensorflow:2.14.0",
    packages_to_install = [
        'pandas==1.5.3',
        'numpy==1.26.4',
        'transformers==4.44.2',
        'scikit-learn==1.5.2',
        'keras==2.14.0',
        'google-cloud-aiplatform==1.18.3',
        'protobuf==3.20.*'
    ]
)
def test_huggingface_model_component(
    test_data: Input[Dataset],
    model: Input[Model],
    project_id: str,
    location: str,
    metric: Output[Metrics],
    reusable_model: Output[Model],
    metrics_artifact: Output[Artifact],
    test_data_name: str,
    label_name: str,
    label_map: Dict[str, int],
    model_save_name: str = 'saved_tf_hf_model',
    batch_size: int = 8,
    max_sequence_length: int = 128,
    huggingface_model_name: str = 'bert-base-multilingual-cased'
) -> NamedTuple('Outputs', [
    ('precision', float),
    ('recall', float),
    ('f1_score', float)
]):
    import os
    import json
    import time
    import tensorflow as tf
    from collections import namedtuple
    import google.cloud.aiplatform as aiplatform
    from transformers import TFAutoModelForSequenceClassification
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    aiplatform.init(project=project_id, location=location, experiment=f'exp-{label_name}-{huggingface_model_name}')
    experiment_run_id = "run-{}".format(int(time.time()))

    aiplatform.start_run(experiment_run_id)

    idx_2_label_map = {v:k for k,v in label_map.items()}

    def parse_tfrecord_fn(example_proto, max_sequence_length=max_sequence_length):
      feature_description = {
          'feature': tf.io.FixedLenFeature([max_sequence_length], tf.int64),
          'label': tf.io.FixedLenFeature([], tf.int64),
      }
      parsed_example = tf.io.parse_single_example(example_proto, feature_description)
      return parsed_example['feature'], parsed_example['label']

    try:
      # Load test dataset
      record_path = os.path.join(test_data.path, f'{test_data_name}.tfrecord')
      test_dataset = tf.data.TFRecordDataset(record_path)
      test_dataset = test_dataset.map(parse_tfrecord_fn).batch(batch_size)

      # Load model from the trained model path
      model_path = os.path.join(model.path, f'{model_save_name}.keras')
      loaded_model = TFAutoModelForSequenceClassification.from_pretrained(
          huggingface_model_name,
          num_labels=len(label_map)
      )
      loaded_model.load_weights(model_path)

      # Get predictions
      predictions = loaded_model.predict(test_dataset)
      logits = predictions.logits

      # Calculate metrics
      y_true = [label for _, label in list(test_dataset.unbatch().as_numpy_iterator())]
      y_pred = list(tf.argmax(logits, axis=1).numpy())

      y_true_decoded = [idx_2_label_map.get(i) for i in y_true]
      y_pred_decoded = [idx_2_label_map.get(i) for i in y_pred]

      classification_metrics = {
        "matrix": confusion_matrix(y_true_decoded, y_pred_decoded, labels=list(idx_2_label_map.values())).tolist(),
        "labels": list(idx_2_label_map.values())
        }

      precision = precision_score(y_true, y_pred, average='weighted')
      recall = recall_score(y_true, y_pred, average='weighted')
      f1 = f1_score(y_true, y_pred, average='weighted')

      precision_per_label = precision_score(y_true, y_pred, average=None)
      recall_per_label = recall_score(y_true, y_pred, average=None)
      f1_per_label = f1_score(y_true, y_pred, average=None)

      reusable_model.uri = model.uri

      metric.log_metric('precision', precision)
      metric.log_metric('recall', recall)
      metric.log_metric('f1', f1)

      metrics_information = {
         'huggingface_model_name': huggingface_model_name,
         'precision': precision, 
         'recall': recall, 'f1': f1,
         'precision_per_label': json.dumps(precision_per_label.tolist()), 
         'recall_per_label': json.dumps(recall_per_label.tolist()), 
         'f1_per_label': json.dumps(f1_per_label.tolist())
         }
      
      aiplatform.log_classification_metrics(
         labels=classification_metrics["labels"],
         matrix=classification_metrics["matrix"],
         display_name=f"{huggingface_model_name}-confusion-matrix"
         )
      
      aiplatform.log_metrics(metrics_information)

      os.makedirs(os.path.dirname(metrics_artifact.path), exist_ok=True)

      with open(metrics_artifact.path, 'w') as f:
          json.dump(metrics_information, f)

      aiplatform.end_run()

      output = namedtuple('Outputs', ['precision', 'recall', 'f1_score'])
      return output(precision, recall, f1)

    except Exception as e:
      error_message = str(e)
      print(f"Error during model testing: {error_message}")
      raise e