from typing import Dict
from kfp.dsl import component, Input, Output, Dataset

@component(
    base_image="tensorflow/tensorflow:2.14.0",
    packages_to_install = [
        'pandas==1.5.3',
        'numpy==1.26.4',
        'transformers==4.44.2',
        'google-cloud-logging==3.11.3',
        'sentencepiece==0.2.0'
        ]
    )
def prepare_data_component(
    data: Input[Dataset],
    tf_dataset: Output[Dataset],
    dataset_name: str,
    feature_name: str, 
    label_name: str,
    label_map: Dict[str, int],
    max_sequence_length: int = 128,
    hugging_face_model_name: str = 'bert-base-multilingual-cased'
    ):
  import os
  import pickle
  import pandas as pd
  import tensorflow as tf
  from transformers import AutoTokenizer
  from google.cloud import logging as gcloud_logging

  logging_client = gcloud_logging.Client()
  logger = logging_client.logger("prepare_data_component")

  # Function to serialize each example
  def serialize_example(feature, label):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=feature))
    label = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    feature_dict = {
        'feature': feature,
        'label': label
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example_proto.SerializeToString()

  tokenizer = AutoTokenizer.from_pretrained(hugging_face_model_name)

  # Load the pickled dataset
  with open(data.path, 'rb') as f:
    loaded_dataframe: pd.DataFrame  = pickle.load(f)

  # Create a dictionary to map each unique label in label to an integer
  labels = loaded_dataframe[label_name].map(label_map).values

  # Convert labels to tensor format for compatibility with TensorFlow
  labels_tf = tf.convert_to_tensor(labels, dtype=tf.int32)

  try:
    features_tf = tokenizer(
      loaded_dataframe[feature_name].tolist(),
      padding=True, truncation=True,
      return_tensors="tf",
      max_length=max_sequence_length)["input_ids"]
  except Exception as e:
    logger.log_struct(
      {
        "severity": "ERROR",
        "message": f"Tokenization Failed. HF Model Name: {hugging_face_model_name}",
        "type": f"TOKENIZATION-ERROR",
        "count": 1
      })
    
  # Create TensorFlow datasets
  try:
    tf_prepared_dataset = tf.data.Dataset.from_tensor_slices((features_tf, labels_tf))
  except Exception as e:
    logger.log_struct(
      {
        "severity": "ERROR",
        "message": f"Tensorflow Dataset Creation Failed. HF Model Name: {hugging_face_model_name}\nFeature size: {len(features_tf)}",
        "type": f"TOKENIZATION-ERROR",
        "count": 1
      })

  # make the folder if does not exist
  os.makedirs(tf_dataset.path, exist_ok=True)

  tfrecord_file_path = os.path.join(tf_dataset.path, f'{dataset_name}.tfrecord')

  # Write the dataset to TFRecord
  with tf.io.TFRecordWriter(tfrecord_file_path) as writer:
    for feature, label in tf_prepared_dataset:
      serialized_example = serialize_example(feature, label)
      writer.write(serialized_example)
