#!/bin/bash

# Run preprocessing pipeline with the correct path
python dags/scripts/preprocessed.py  # Ensure this script outputs to data/preprocessed_data.parquet

# Track the preprocessed data with DVC
dvc add data/preprocessed_dataset.parquet

# Push the data and metadata to the DVC remote
dvc push

# Commit DVC metadata changes to Git
git add data/preprocessed_dataset.parquet.dvc dvc.lock
git commit -m "Automated DVC push for processed data"
git push
