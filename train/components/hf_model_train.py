from typing import Dict
from kfp.dsl import component, Input, Output, Dataset, Model

@component(
    base_image="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-14.py310:latest",
    packages_to_install = [
        'pandas==1.5.3',
        'numpy==1.26.4',
        'transformers==4.44.2',
        'keras==2.14.0',
        'sentencepiece==0.2.0'
        ]
    )
def train_huggingface_model_component(
    train_data: Input[Dataset],
    model_output: Output[Model],
    train_data_name: str,
    label_map: Dict[str, int],
    model_save_name: str = 'saved_tf_hf_model',
    huggingface_model_name: str = 'bert-base-multilingual-cased',
    max_epochs: int = 2,
    train_prec: float = 0.8,
    batch_size: int = 8,
    max_sequence_length: int = 128,
    slack_url: str = None,
    ):
  import os
  import requests
  import tensorflow as tf
  from transformers import TFAutoModelForSequenceClassification
  from datetime import datetime

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

  # Track the start time of the component execution
  start_time = datetime.now()

  def parse_tfrecord_fn(example_proto, max_sequence_length=max_sequence_length):
    # Define the feature description dictionary
    feature_description = {
        'feature': tf.io.FixedLenFeature([max_sequence_length], tf.int64),  # Adjust the shape [128] as per your data
        'label': tf.io.FixedLenFeature([], tf.int64),
        }
    # Parse the input tf.train.Example proto using the feature description
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    return parsed_example['feature'], parsed_example['label']

  try:

    if slack_url:
       send_slack_message(
          webhook_url=slack_url, message_str=f'KubeFlow Component: Train HuggingFace Model | Model: {huggingface_model_name} started',
          execution_date=start_time.date(), execution_time=start_time.time(), 
          duration=0, is_success=True
          )

    record_path = os.path.join(train_data.path, f'{train_data_name}.tfrecord')
    train_dataset = tf.data.TFRecordDataset(record_path)
    train_dataset = train_dataset.map(parse_tfrecord_fn)
    train_dataset = train_dataset.cache()

    # Determine the total number of elements in the train_dataset
    total_size = len(list(train_dataset.as_numpy_iterator()))
    train_size = int(total_size * train_prec)

    # Split the dataset
    train_subset = train_dataset.take(train_size)
    validation_subset = train_dataset.skip(train_size)

    train_subset = train_subset.batch(batch_size).shuffle(buffer_size=train_size)
    validation_subset = validation_subset.batch(batch_size)

    train_steps = train_size * max_epochs // batch_size

    tf.keras.backend.clear_session()
    model = TFAutoModelForSequenceClassification.from_pretrained(huggingface_model_name, num_labels=len(label_map))

    lr_schedular = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=5e-5,
        decay_steps=train_steps,
        end_learning_rate=0.0,
    )
    optimizer = tf.optimizers.Adam(
      learning_rate=lr_schedular
      )
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
        )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        min_lr=1e-6
    )

    history = model.fit(
          train_subset,
          validation_data=validation_subset,
          epochs=max_epochs,
          callbacks=[early_stopping, lr_scheduler]
          )
    
    if slack_url:
       send_slack_message(
          webhook_url=slack_url, message_str=f'KubeFlow Component: Train HuggingFace Model | Model: {huggingface_model_name} Successfully Trained', 
          execution_date=start_time.date(), execution_time=start_time.time(), 
          duration=(datetime.now() - start_time).total_seconds() / 60, is_success=True
       )

    os.makedirs(model_output.path, exist_ok=True)

    model.save(os.path.join(model_output.path, f'{model_save_name}.keras'))
    model.save(os.path.join(model_output.path))

    # Track the end time and calculate duration
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60  # Duration in minutes

  except Exception as e:
    if slack_url:
       send_slack_message(
          webhook_url=slack_url, message_str=f'KubeFlow Component: Train HuggingFace Model | Model: {huggingface_model_name} Failed to Train', 
          execution_date=start_time.date(), execution_time=start_time.time(), 
          duration=(datetime.now() - start_time).total_seconds() / 60, is_success=True
       )