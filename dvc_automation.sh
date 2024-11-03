#!/bin/bash

# Run preprocessing pipeline with the correct path
python dags/scripts/preprocessed.py  # Ensure this script outputs to data/preprocessed_data.parquet

# Track the preprocessed data with DVC
dvc add data/preprocessed_dataset.parquet

# Push the data and metadata to the DVC remote
dvc push
