# preprocessing.py
import pandas as pd
import os

def deduplicate_records(df):
    deduped_df = df.drop_duplicates(subset=['Product', 'Sub-product', 'Consumer complaint narrative'], keep='first')
    return deduped_df

def remove_null_records(df, columns):
    """Remove rows where any of the specified columns have null values."""
    cleaned_df = df.dropna(subset=columns)
    return cleaned_df


def data_cleaning(**kwargs):
    # Use the correct path where the file is mounted in Docker
    file_path = os.path.join(os.path.dirname(__file__), "../../data/JPMORGAN_CHASE_COMPLAINTS.csv")

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Deduplicate the records
    deduped_df = deduplicate_records(df)

    # Remove rows where specific columns have null values
    columns_to_check = ['Product', 'Sub-product', 'Department', 'Consumer complaint narrative']
    cleaned_df = remove_null_records(deduped_df, columns_to_check)

    # Save the processed data to a new file
    output_path = os.path.join(os.path.dirname(__file__), "../../data/Processed_JPMORGAN_CHASE_COMPLAINTS.csv")
    cleaned_df.to_csv(output_path, index=False)
