import pytest
import polars as pl
import io
import os
import sys
import logging
import warnings
from datetime import datetime
from unittest.mock import patch
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress DeprecationWarnings related to google.protobuf
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*google\.protobuf\..*",
)

# Add the project root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preprocessing_pipeline.dags.scripts.preprocessing import (
    get_custom_logger,
    load_data,
    filter_records_by_word_count_and_date,
    filter_records_by_language,
    aggregate_filtered_task,
    clean_xxx_patterns,
    data_cleaning,
    remove_abusive_data,
    standardise_product_class,
)

def test_get_custom_logger():
    """
    Test the get_custom_logger function to ensure it initializes a logger with the correct settings.
    """
    logger = get_custom_logger()
    assert logger.name == "preprocessing_logger"
    assert logger.level == logging.INFO
    assert not logger.propagate
    assert len(logger.handlers) > 0

def test_load_data():
    """
    Test the load_data function to ensure it correctly loads and serializes the dataset.
    """
    # Mock dataset to be returned by pl.read_parquet
    mock_dataset = pl.DataFrame({
        'complaint_id': [1, 2, 3],
        'complaint': ['Complaint 1', 'Complaint 2', 'Complaint 3'],
        'complaint_hindi': ['शिकायत 1', 'शिकायत 2', 'शिकायत 3'],
    })

    # Mock pl.read_parquet to return the mock dataset
    def mock_read_parquet(path):
        return mock_dataset

    with patch('data_preprocessing_pipeline.dags.scripts.preprocessing.pl.read_parquet', new=mock_read_parquet):
        serialized_data = load_data()
        # Deserialize to check correctness
        loaded_data = pl.DataFrame.deserialize(io.StringIO(serialized_data), format='json')
        # Assertions
        assert len(loaded_data) == 3
        assert loaded_data['complaint_id'].to_list() == [1, 2, 3]
        assert loaded_data['complaint'][0] == 'Complaint 1'

def test_filter_records_by_word_count_and_date():
    """
    Test the filter_records_by_word_count_and_date function to ensure it correctly filters out records
    that do not meet the minimum word count and date range criteria.
    """
    # Create a sample dataset
    data = pl.DataFrame({
        "complaint": ["This is a valid complaint", "Too short", "", "Another valid complaint"],
        "complaint_hindi": ["यह एक वैध शिकायत है", "छोटी", "", "एक और वैध शिकायत"],
        "date_received": [datetime(2016, 1, 1), datetime(2014, 1, 1), datetime(2018, 5, 20), datetime(2025, 1, 1)],
        "complaint_id": [1, 2, 3, 4],
    })
    min_word_length = 3
    # Serialize the dataset
    serialized_data = data.serialize(format="json")

    # Apply the filtering function
    filtered_data_serialized = filter_records_by_word_count_and_date(serialized_data, min_word_length)
    # Deserialize the filtered data
    filtered_data = pl.DataFrame.deserialize(io.StringIO(filtered_data_serialized), format="json")

    # Assert that only the valid record within date range remains
    expected_ids = [1]  # Only the first record meets all criteria

    assert filtered_data["complaint_id"].to_list() == expected_ids

def test_filter_records_by_language():
    """
    Test the filter_records_by_language function to ensure it correctly filters out records
    that do not have the specified languages ('HI' or 'EN'), accounting for threading misalignment.
    """
    # Place the warnings filter at the very top, before any imports
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    import polars as pl
    import io
    from unittest.mock import patch

    # Sample dataset
    data = pl.DataFrame({
        'complaint': [
            'This is an English complaint',
            'यह हिंदी शिकायत है',
            'Ceci est une plainte en français'
        ],
        'complaint_id': [1, 2, 3],
    })
    serialized_data = data.serialize(format='json')

    # Mock detect_language function
    def mock_detect_language(text):
        if 'English' in text:
            return 'EN'
        elif 'हिंदी' in text:
            return 'HI'
        else:
            return 'FR'  # French

    # Mock as_completed to return futures in the order they were submitted
    from concurrent.futures import as_completed

    def mock_as_completed(futures):
        # Since futures are a dictionary in the original code, we need to return the keys (future objects) in order
        return list(futures.keys())

    with patch('data_preprocessing_pipeline.dags.scripts.preprocessing.detect_language', new=mock_detect_language), \
         patch('data_preprocessing_pipeline.dags.scripts.preprocessing.as_completed', new=mock_as_completed):
        filtered_data_serialized = filter_records_by_language(serialized_data)
        filtered_data = pl.DataFrame.deserialize(io.StringIO(filtered_data_serialized), format='json')

    # Assert that only records with 'HI' or 'EN' are retained
    assert len(filtered_data) == 2

    # Get the complaint_ids from the filtered data
    complaint_ids = sorted(filtered_data['complaint_id'].to_list())
    # Now the complaint_ids should be [1, 2]
    assert complaint_ids == [1, 2]
    
def test_aggregate_filtered_task():
    """
    Test the aggregate_filtered_task function to ensure it correctly joins two datasets on 'complaint_id' and
    selects the specified columns.
    """
    # Sample datasets with necessary columns
    data_a = pl.DataFrame({
        'complaint_id': [1, 2, 3],
        'date_received': [datetime(2020, 1, 1)] * 3,
        'complaint': ['Complaint A1', 'Complaint A2', 'Complaint A3'],
        'product': ['Product A', 'Product B', 'Product C'],
        'department': ['Dept A', 'Dept B', 'Dept C'],
        'time_resolved_in_days': [5, 10, 15],
        'sub_product': ['Sub Product A', 'Sub Product B', 'Sub Product C'],
        'issue': ['Issue A', 'Issue B', 'Issue C'],
        'sub_issue': ['Sub Issue A', 'Sub Issue B', 'Sub Issue C'],
        'company': ['Company A', 'Company B', 'Company C'],
        'state': ['CA', 'NY', 'TX'],
        'zipcode': ['90001', '10001', '73301'],
        'company_response_consumer': ['Response A', 'Response B', 'Response C'],
        'consumer_consent_provided': ['Yes', 'No', 'Yes'],
        'submitted_via': ['Web', 'Email', 'Fax'],
        'date_sent_to_company': [datetime(2020, 1, 2)] * 3,
        'timely_response': ['Yes', 'No', 'Yes'],
        'consumer_disputed': ['No', 'Yes', 'No'],
    })
    data_b = pl.DataFrame({
        'complaint_id': [2, 3, 4],
        'date_resolved': [datetime(2020, 2, 1)] * 3,
        'complaint_hindi': ['शिकायत B2', 'शिकायत B3', 'शिकायत B4'],
        # Include any additional columns if necessary
    })
    serialized_data_a = data_a.serialize(format='json')
    serialized_data_b = data_b.serialize(format='json')

    # Mock write_parquet to avoid actual file IO
    with patch('data_preprocessing_pipeline.dags.scripts.preprocessing.pl.DataFrame.write_parquet') as mock_write_parquet:
        aggregate_filtered_task(serialized_data_a, serialized_data_b)
        # Ensure write_parquet was called
        assert mock_write_parquet.called

def test_clean_xxx_patterns():
    """
    Test the clean_xxx_patterns function to ensure it removes patterns like 'xxx', 'xxxx2022', 'abcxxx' correctly.
    """
    # Input text with various patterns
    input_text = "This is xxxx2022 an example abcxxx text with xxx patterns xxx."
    expected_output = "This is an example abc text with patterns"

    # Apply the cleaning function
    cleaned_text = clean_xxx_patterns(input_text)

    # Assert the cleaned text matches the expected output
    assert cleaned_text == expected_output

def test_data_cleaning():
    """
    Test the data_cleaning function to ensure it correctly cleans the dataset.
    """
    # Sample dataset
    data = pl.DataFrame({
        'complaint_id': [1, 2, 3],
        'product': ['Product A', 'Product B', None],
        'department': ['Dept A', None, 'Dept C'],
        'complaint': ['Complaint with XXX1234', 'Another complaint', 'Duplicate complaint'],
        'complaint_hindi': ['शिकायत xxx', 'एक और शिकायत', 'Duplicate complaint'],
    })

    # Mock pl.read_parquet to return the sample dataset
    def mock_read_parquet(path):
        return data

    # Mock write_parquet to avoid actual file IO
    with patch('data_preprocessing_pipeline.dags.scripts.preprocessing.pl.read_parquet', new=mock_read_parquet):
        cleaned_data_serialized = data_cleaning()
        # Deserialize to check correctness
        cleaned_data = pl.DataFrame.deserialize(io.StringIO(cleaned_data_serialized), format='json')

    # Assertions
    # Check that records with nulls in 'product', 'department', 'complaint' are dropped
    assert len(cleaned_data) == 1  # Only the first record should remain
    # Check that 'complaint' is lowercased and cleaned
    assert cleaned_data['complaint'][0] == 'complaint with'
    # Check that 'complaint_hindi' is cleaned
    assert cleaned_data['complaint_hindi'][0] == 'शिकायत'

def test_remove_abusive_data():
    """
    Test the remove_abusive_data function to ensure it replaces abusive words with placeholders in both
    English and Hindi complaints.
    """
    # Sample dataset with abusive words
    data = pl.DataFrame({
        "complaint": ["This is a badword in the complaint", "Clean complaint"],
        "complaint_hindi": ["यह एक गाली है", "साफ शिकायत"],
        "complaint_id": [1, 2],
    })
    # Serialize the dataset
    serialized_data = data.serialize(format="json")

    # Mock the abusive words datasets
    def mock_read_parquet(path):
        if "profanity_bank_dataset.parquet" in path:
            return pl.DataFrame({"profanity": ["badword"]})
        elif "hindi_abuse_words.parquet" in path:
            return pl.DataFrame({"words": ["गाली"]})
        else:
            return pl.DataFrame()

    # Prepare to capture the dataset
    captured_dataset = []

    # Define a side effect function to capture the dataset
    def mock_write_parquet(self, path):
        captured_dataset.append(self)

    # Use monkeypatch to replace read_parquet and write_parquet functions
    with patch('data_preprocessing_pipeline.dags.scripts.preprocessing.pl.read_parquet', new=mock_read_parquet), \
         patch('data_preprocessing_pipeline.dags.scripts.preprocessing.pl.DataFrame.write_parquet', new=mock_write_parquet):

        output_path = remove_abusive_data(serialized_data)
        # Access the cleaned dataset
        cleaned_data = captured_dataset[0]

    # Expected placeholders
    english_placeholder = "<abusive_data>"
    hindi_placeholder = "<गाल>"

    # Assert that abusive words are replaced
    assert cleaned_data["abuse_free_complaint"][0] == f"This is a {english_placeholder} in the complaint"
    assert cleaned_data["abuse_free_complaint_hindi"][0] == f"यह एक {hindi_placeholder} है"
    # Assert that clean complaints remain unchanged
    assert cleaned_data["abuse_free_complaint"][1] == "Clean complaint"
    assert cleaned_data["abuse_free_complaint_hindi"][1] == "साफ शिकायत"

def test_standardise_product_class():
    """
    Test the standardise_product_class function to ensure it correctly standardizes product names.
    """
    # Sample dataset
    data = pl.DataFrame({
        'product': ['Credit Reporting', 'Debt Collection', 'Other Financial Service', 'Unknown Product'],
        'department': ['Dept A', 'Dept B', 'Dept C', 'Dept D'],
    })
    dataset_path = 'dummy/path/to/dataset.parquet'

    # Mock pl.read_parquet to return the sample dataset
    def mock_read_parquet(path):
        return data

    # Prepare to capture the dataset
    captured_dataset = []

    # Define a side effect function to capture the dataset
    def mock_write_parquet(self, path):
        captured_dataset.append(self)

    # Mock write_parquet to avoid actual file IO
    with patch('data_preprocessing_pipeline.dags.scripts.preprocessing.pl.read_parquet', new=mock_read_parquet), \
         patch('data_preprocessing_pipeline.dags.scripts.preprocessing.pl.DataFrame.write_parquet', new=mock_write_parquet):

        output_path = standardise_product_class(dataset_path)
        # Ensure write_parquet was called
        assert len(captured_dataset) > 0
        # Retrieve the DataFrame passed to write_parquet
        standardized_data = captured_dataset[0]

    # Expected products after standardization
    expected_products = [
        'credit / debt management & repair services',
        'credit / debt management & repair services',
        # 'Other Financial Service' should be dropped, so 'Unknown Product' remains
        'unknown product',
    ]

    # Assertions
    # Check that 'Other Financial Service' is dropped
    assert len(standardized_data) == 3  # One record should be dropped
    # Check that 'product' is lowercased and standardized
    assert standardized_data['product'].to_list() == expected_products