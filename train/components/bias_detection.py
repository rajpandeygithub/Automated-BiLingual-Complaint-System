from typing import Dict
from kfp.dsl import component, Input, Output, Dataset, Model

@component(
    base_image="tensorflow/tensorflow:2.14.0",
    packages_to_install=[
        'pandas==1.5.3',
        'numpy==1.26.4',
        'fairlearn==0.8.0',
        'scikit-learn==1.5.2',
        'transformers==4.44.2',
    ]
)
def detect_bias_component(
    test_data: Input[Dataset],
    model: Input[Model],
    bias_report: Output[Dataset],
    test_data_name: str,
    label_map: Dict[str, int],
    batch_size: int = 8,
    max_sequence_length: int = 128,
    accuracy_threshold: float = 0.2,
    model_save_name: str = 'saved_tf_hf_model',
    huggingface_model_name: str = 'bert-base-multilingual-cased'
):
    import os
    import requests
    import pandas as pd
    import tensorflow as tf
    from datetime import datetime
    from transformers import TFAutoModelForSequenceClassification
    from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate, selection_rate
    
    # Hardcoded label map
    idx_2_label_map = {v:k for k,v in label_map.items()}

    # Function to send alert
    def send_alert(summary_message):
        SLACK_WEBHOOK_URL = 'https://hooks.slack.com/services/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'
        payload = {"text": summary_message}
        try:
            response = requests.post(SLACK_WEBHOOK_URL, json=payload)
            response.raise_for_status()
            print(f"Alert sent successfully: {summary_message}")
        except Exception as e:
            print(f"Failed to send alert: {str(e)}")

    # Load and parse test dataset
    def parse_tfrecord_fn(example_proto, max_sequence_length=max_sequence_length):
      feature_description = {
          'feature': tf.io.FixedLenFeature([max_sequence_length], tf.int64),
          'label': tf.io.FixedLenFeature([], tf.int64),
      }
      parsed_example = tf.io.parse_single_example(example_proto, feature_description)
      return parsed_example['feature'], parsed_example['label']


    try:
        record_path = os.path.join(test_data.path, f'{test_data_name}.tfrecord')
        test_dataset = tf.data.TFRecordDataset(record_path)
        test_dataset = test_dataset.map(parse_tfrecord_fn).batch(batch_size)

        model_path = os.path.join(model.path, f'{model_save_name}.keras')
        loaded_model = TFAutoModelForSequenceClassification.from_pretrained(
            huggingface_model_name,
            num_labels=len(label_map)
        )
        loaded_model.load_weights(model_path)

        predictions = loaded_model.predict(test_dataset)
        logits = predictions.logits

        y_true = [label for _, label in list(test_dataset.unbatch().as_numpy_iterator())]
        y_pred = list(tf.argmax(logits, axis=1).numpy())

        df = pd.DataFrame({
            'true_label': y_true,
            'predicted_label': y_pred
        })

        def multiclass_metric(metric_fn, y_true, y_pred, label):
            """Compute a metric for a specific label in a one-vs-rest approach."""
            y_true_binary = (y_true == label).astype(int)
            y_pred_binary = (y_pred == label).astype(int)
            return metric_fn(y_true_binary, y_pred_binary)

        metrics = {
            'accuracy': lambda y_true, y_pred: (y_true == y_pred).mean(),
            'true_positive_rate': lambda y_true, y_pred: {
                label: multiclass_metric(true_positive_rate, y_true, y_pred, label)
                for label in set(y_true)
            },
            'false_positive_rate': lambda y_true, y_pred: {
                label: multiclass_metric(false_positive_rate, y_true, y_pred, label)
                for label in set(y_true)
            },
            'selection_rate': lambda y_true, y_pred: {
                label: multiclass_metric(selection_rate, y_true, y_pred, label)
                for label in set(y_true)
            },
        }

        # Calculate metrics using Fairlearn
        metric_frame = MetricFrame(
            metrics=metrics,
            y_true=df['true_label'],
            y_pred=df['predicted_label'],
            sensitive_features=df['true_label']
        )
        bias_report_df = metric_frame.by_group.reset_index()
        bias_report_df.to_pickle(bias_report.path)

        # Identify slices with low accuracy
        grouping_column = 'true_label' if 'true_label' in bias_report_df.columns else bias_report_df.columns[0]
        low_accuracy_slices = bias_report_df[bias_report_df['accuracy'] < accuracy_threshold]

        # Generate alert if necessary
        if not low_accuracy_slices.empty:
            low_accuracy_slices_str = "\n".join([
                f"- {idx_2_label_map.get(row[grouping_column], 'Unknown')}: Accuracy {row['accuracy']:.2f}"
                for _, row in low_accuracy_slices.iterrows()
            ])
            summary_message = (
                f"Bias Alert: The following slices have accuracy below the threshold of {accuracy_threshold:.2f}:\n"
                f"{low_accuracy_slices_str}\n\n"
                f"**Action Required:** Please mitigate this issue as soon as possible to ensure fairness in the model."
            )
            send_alert(summary_message)
        else:
            print("No bias detected. All slices meet the accuracy threshold.")

    except Exception as e:
        print(f"Error in bias detection: {str(e)}")
        raise e
