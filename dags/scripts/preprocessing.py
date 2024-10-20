# preprocessing.py
import pandas as pd
import os

def deduplicate_records(df):
    """Remove duplicate records based on specific columns."""
    deduped_df = df.drop_duplicates(subset=['Product', 'Sub-product', 'Consumer complaint narrative'], keep='first')
    return deduped_df

def remove_null_records(df, columns):
    """Remove rows where any of the specified columns have null values."""
    cleaned_df = df.dropna(subset=columns)
    return cleaned_df

def data_cleaning(**kwargs):
    """Main data cleaning function that deduplicates and removes null records."""
    file_path = os.path.join(os.path.dirname(__file__), "../../data/JPMORGAN_CHASE_COMPLAINTS.csv")

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Deduplicate the records
    deduped_df = deduplicate_records(df)

    # Remove rows where specific columns have null values
    columns_to_check = ['Product', 'Sub-product', 'Department', 'Consumer complaint narrative']
    cleaned_df = remove_null_records(deduped_df, columns_to_check)

    # Push the cleaned data to XCom for the next task to use
    return cleaned_df.to_dict()

def remove_abusive_data(**kwargs):
    """Remove abusive words from the 'Consumer complaint narrative' and save the cleaned data."""
    # Get the data from the previous task via XCom
    task_instance = kwargs['ti']
    df_dict = task_instance.xcom_pull(task_ids='datacleaning_process')

    if df_dict is None:
        raise ValueError("No data found from the previous task.")

    # Convert the dictionary back to a DataFrame
    df = pd.DataFrame(df_dict)

    # Load abusive words
    abusive_words_path = os.path.join(os.path.dirname(__file__), "../../data/profanity_bank_dataset.csv")
    abusive_words_df = pd.read_csv(abusive_words_path)
    abusive_words = abusive_words_df['Profanity'].dropna().tolist()
    abusive_words_set = set([word.lower() for word in abusive_words if word.strip()])

    def clean_text(input_text):
        """Clean a single text input by replacing abusive words with 'yyy'."""
        words = input_text.split()
        cleaned_words = ['yyy' if word.lower() in abusive_words_set else word for word in words]
        return ' '.join(cleaned_words)

    # Clean the 'Consumer complaint narrative' column in the DataFrame
    df['Consumer complaint narrative'] = df['Consumer complaint narrative'].apply(clean_text)

    # Save the cleaned data to a new file
    output_path = os.path.join(os.path.dirname(__file__), "../../data/PREPROCESSED_JPMORGAN_CHASE_COMPLAINTS.csv")
    df.to_csv(output_path, index=False)

    print(f"Data after removing abusive content has been saved to {output_path}")