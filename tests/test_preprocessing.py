import pytest
from unittest.mock import patch, MagicMock
import polars as pl
import io
import os
import sys

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Add the project root to sys.path
sys.path.insert(0, project_root)

# Import the preprocessing module from the correct path
from data_preprocessing_pipeline.dags.scripts.preprocessing import (
    load_data,
    filter_records_by_word_count_and_date,
    filter_records_by_language,
    aggregate_filtered_task,
    data_cleaning,
    remove_abusive_data,
    insert_data_to_bigquery
)

# Function 1: load_data()
def test_load_data_success():
    # Test that data is loaded successfully
    data = load_data()
    assert data is not None
    dataset = pl.DataFrame.deserialize(io.StringIO(data), format="json")
    assert len(dataset) > 0

@patch('preprocessing.pl.read_parquet')
def test_load_data_file_not_found(mock_read_parquet):
    # Simulate FileNotFoundError
    mock_read_parquet.side_effect = FileNotFoundError("File not found")
    with pytest.raises(Exception) as exc_info:
        load_data()
    assert "Error With Dataset Loading" in str(exc_info.value)

@patch('preprocessing.pl.read_parquet')
def test_load_data_empty_dataset(mock_read_parquet):
    # Simulate empty dataset
    empty_df = pl.DataFrame()
    mock_read_parquet.return_value = empty_df
    data = load_data()
    dataset = pl.DataFrame.deserialize(io.StringIO(data), format="json")
    assert len(dataset) == 0

#  Function 2: filter_records_by_word_count_and_date

def test_filter_by_word_count():
    # Create sample data
    data = pl.DataFrame({
        'complaint': ['short', 'this is a longer complaint'],
        'date_received': [pl.date(2020, 1, 1), pl.date(2020, 1, 2)]
    })
    serialized_data = data.serialize(format="json")

    # Apply filter with min_word_length=2
    result = filter_records_by_word_count_and_date(serialized_data, min_word_length=2)
    filtered_data = pl.DataFrame.deserialize(io.StringIO(result), format="json")

    assert len(filtered_data) == 1
    assert filtered_data['complaint'][0] == 'this is a longer complaint'

def test_filter_by_date_range():
    # Create sample data
    data = pl.DataFrame({
        'complaint': ['valid complaint', 'another complaint'],
        'date_received': [pl.date(2014, 1, 1), pl.date(2025, 1, 1)]
    })
    serialized_data = data.serialize(format="json")

    # Apply filter
    result = filter_records_by_word_count_and_date(serialized_data, min_word_length=1)
    filtered_data = pl.DataFrame.deserialize(io.StringIO(result), format="json")

    assert len(filtered_data) == 0  # Both dates are outside the range

def test_filter_by_word_count_and_date():
    # Create sample data
    data = pl.DataFrame({
        'complaint': ['valid complaint', 'short', 'another valid complaint'],
        'date_received': [pl.date(2020, 6, 1), pl.date(2020, 6, 2), pl.date(2020, 6, 3)]
    })
    serialized_data = data.serialize(format="json")

    # Apply filter with min_word_length=2
    result = filter_records_by_word_count_and_date(serialized_data, min_word_length=2)
    filtered_data = pl.DataFrame.deserialize(io.StringIO(result), format="json")

    assert len(filtered_data) == 2
    assert all(filtered_data['complaint'] != 'short')

def test_no_records_meet_criteria():
    # Create sample data
    data = pl.DataFrame({
        'complaint': ['short', 'tiny'],
        'date_received': [pl.date(2014, 1, 1), pl.date(2025, 1, 1)]
    })
    serialized_data = data.serialize(format="json")

    # Apply filter
    result = filter_records_by_word_count_and_date(serialized_data, min_word_length=5)
    filtered_data = pl.DataFrame.deserialize(io.StringIO(result), format="json")

    assert len(filtered_data) == 0

# Function 3: filter_records_by_language

@patch('preprocessing.detect_language')
def test_filter_records_by_language_mixed(mock_detect_language):
    # Mock the language detection
    mock_detect_language.side_effect = ['EN', 'FR', 'HI']
    
    data = pl.DataFrame({
        'complaint': ['This is an English complaint', 'Ceci est une plainte en français', 'यह एक हिंदी शिकायत है']
    })
    serialized_data = data.serialize(format="json")

    result = filter_records_by_language(serialized_data)
    filtered_data = pl.DataFrame.deserialize(io.StringIO(result), format="json")

    assert len(filtered_data) == 2
    assert 'Ceci est une plainte en français' not in filtered_data['complaint'].to_list()

@patch('preprocessing.detect_language')
def test_filter_records_by_language_all_supported(mock_detect_language):
    mock_detect_language.side_effect = ['EN', 'HI']
    
    data = pl.DataFrame({
        'complaint': ['English complaint', 'हिंदी शिकायत']
    })
    serialized_data = data.serialize(format="json")

    result = filter_records_by_language(serialized_data)
    filtered_data = pl.DataFrame.deserialize(io.StringIO(result), format="json")

    assert len(filtered_data) == 2

@patch('preprocessing.detect_language')
def test_filter_records_by_language_none_supported(mock_detect_language):
    mock_detect_language.side_effect = ['FR', 'ES']
    
    data = pl.DataFrame({
        'complaint': ['Plainte en français', 'Queja en español']
    })
    serialized_data = data.serialize(format="json")

    result = filter_records_by_language(serialized_data)
    filtered_data = pl.DataFrame.deserialize(io.StringIO(result), format="json")

    assert len(filtered_data) == 0

@patch('preprocessing.detect_language')
def test_language_detection_accuracy(mock_detect_language):
    # Mock language detection to return unexpected languages
    mock_detect_language.side_effect = ['EN', 'DE', 'HI', 'IT']
    
    data = pl.DataFrame({
        'complaint': ['English text', 'Deutscher Text', 'हिंदी पाठ', 'Testo italiano']
    })
    serialized_data = data.serialize(format="json")

    result = filter_records_by_language(serialized_data)
    filtered_data = pl.DataFrame.deserialize(io.StringIO(result), format="json")

    assert len(filtered_data) == 2
    assert 'Deutscher Text' not in filtered_data['complaint'].to_list()
    assert 'Testo italiano' not in filtered_data['complaint'].to_list()

# Function 4: aggregate_filtered_task

def test_aggregate_successful_join(tmpdir):
    data_a = pl.DataFrame({
        'complaint_id': [1, 2],
        'data_a_col': ['a1', 'a2']
    })
    data_b = pl.DataFrame({
        'complaint_id': [1, 2],
        'data_b_col': ['b1', 'b2']
    })
    serialized_a = data_a.serialize(format="json")
    serialized_b = data_b.serialize(format="json")

    # Override the output path
    output_path = os.path.join(tmpdir, 'preprocessed_dataset.parquet')
    with patch('preprocessing.os.path.join', return_value=output_path):
        aggregate_filtered_task(serialized_a, serialized_b)
        assert os.path.exists(output_path)
        result = pl.read_parquet(output_path)
        assert len(result) == 2
        assert 'data_a_col' in result.columns
        assert 'data_b_col' in result.columns

def test_aggregate_no_common_ids(tmpdir):
    data_a = pl.DataFrame({
        'complaint_id': [1, 2],
        'data_a_col': ['a1', 'a2']
    })
    data_b = pl.DataFrame({
        'complaint_id': [3, 4],
        'data_b_col': ['b3', 'b4']
    })
    serialized_a = data_a.serialize(format="json")
    serialized_b = data_b.serialize(format="json")

    output_path = os.path.join(tmpdir, 'preprocessed_dataset.parquet')
    with patch('preprocessing.os.path.join', return_value=output_path):
        aggregate_filtered_task(serialized_a, serialized_b)
        result = pl.read_parquet(output_path)
        assert len(result) == 0

def test_aggregate_missing_columns(tmpdir):
    data_a = pl.DataFrame({
        'complaint_id': [1, 2],
        'data_a_col': ['a1', 'a2']
    })
    data_b = pl.DataFrame({
        'id': [1, 2],  # Missing 'complaint_id' column
        'data_b_col': ['b1', 'b2']
    })
    serialized_a = data_a.serialize(format="json")
    serialized_b = data_b.serialize(format="json")

    output_path = os.path.join(tmpdir, 'preprocessed_dataset.parquet')
    with patch('preprocessing.os.path.join', return_value=output_path):
        with pytest.raises(Exception):
            aggregate_filtered_task(serialized_a, serialized_b)

def test_aggregate_duplicate_ids(tmpdir):
    data_a = pl.DataFrame({
        'complaint_id': [1, 1],
        'data_a_col': ['a1', 'a1_duplicate']
    })
    data_b = pl.DataFrame({
        'complaint_id': [1],
        'data_b_col': ['b1']
    })
    serialized_a = data_a.serialize(format="json")
    serialized_b = data_b.serialize(format="json")

    output_path = os.path.join(tmpdir, 'preprocessed_dataset.parquet')
    with patch('preprocessing.os.path.join', return_value=output_path):
        aggregate_filtered_task(serialized_a, serialized_b)
        result = pl.read_parquet(output_path)
        assert len(result) == 2  # Should have two records due to duplicate IDs

# Function 5: data_cleaning

def test_data_cleaning_text_processing(tmpdir):
    # Create sample data
    data = pl.DataFrame({
        'product': ['Product1'],
        'department': ['Department1'],
        'complaint': ['This is A COMPLAINT! With Special #Characters$'],
        'complaint_id': [1]
    })
    output_path = os.path.join(tmpdir, 'preprocessed_dataset.parquet')
    data.write_parquet(output_path)

    with patch('preprocessing.os.path.join', return_value=output_path):
        cleaned_data_serialized = data_cleaning()
        cleaned_data = pl.DataFrame.deserialize(io.StringIO(cleaned_data_serialized), format="json")
        assert cleaned_data['complaint'][0] == 'this is a complaint with special characters'

def test_data_cleaning_duplicate_removal(tmpdir):
    data = pl.DataFrame({
        'product': ['Product1', 'Product1'],
        'department': ['Department1', 'Department1'],
        'complaint': ['Complaint text', 'Complaint text'],
        'complaint_id': [1, 2]
    })
    output_path = os.path.join(tmpdir, 'preprocessed_dataset.parquet')
    data.write_parquet(output_path)

    with patch('preprocessing.os.path.join', return_value=output_path):
        cleaned_data_serialized = data_cleaning()
        cleaned_data = pl.DataFrame.deserialize(io.StringIO(cleaned_data_serialized), format="json")
        assert len(cleaned_data) == 1

def test_data_cleaning_null_values(tmpdir):
    data = pl.DataFrame({
        'product': ['Product1', None],
        'department': ['Department1', 'Department2'],
        'complaint': ['Complaint text', 'Another complaint'],
        'complaint_id': [1, 2]
    })
    output_path = os.path.join(tmpdir, 'preprocessed_dataset.parquet')
    data.write_parquet(output_path)

    with patch('preprocessing.os.path.join', return_value=output_path):
        cleaned_data_serialized = data_cleaning()
        cleaned_data = pl.DataFrame.deserialize(io.StringIO(cleaned_data_serialized), format="json")
        assert len(cleaned_data) == 1
        assert cleaned_data['complaint'][0] == 'complaint text'

def test_data_cleaning_no_changes(tmpdir):
    data = pl.DataFrame({
        'product': ['product1'],
        'department': ['department1'],
        'complaint': ['clean complaint'],
        'complaint_id': [1]
    })
    output_path = os.path.join(tmpdir, 'preprocessed_dataset.parquet')
    data.write_parquet(output_path)

    with patch('preprocessing.os.path.join', return_value=output_path):
        cleaned_data_serialized = data_cleaning()
        cleaned_data = pl.DataFrame.deserialize(io.StringIO(cleaned_data_serialized), format="json")
        assert len(cleaned_data) == 1
        assert cleaned_data['complaint'][0] == 'clean complaint'

# Function 6: remove_abusive_data

@patch('preprocessing.pl.read_parquet')
def test_remove_abusive_data_english(mock_read_parquet, tmpdir):
    # Mock abusive words dataset
    abusive_words_eng = pl.DataFrame({'profanity': ['badword']})
    abusive_words_hindi = pl.DataFrame({'words': []})  # Empty for this test
    def side_effect(*args, **kwargs):
        if 'profanity_bank_dataset.parquet' in args[0]:
            return abusive_words_eng
        elif 'hindi_abuse_words.parquet' in args[0]:
            return abusive_words_hindi
    mock_read_parquet.side_effect = side_effect

    data = pl.DataFrame({
        'complaint': ['This is a badword in the complaint'],
        'complaint_hindi': ['']
    })
    serialized_data = data.serialize(format="json")
    output_path = os.path.join(tmpdir, 'preprocessed_dataset.parquet')

    with patch('preprocessing.os.path.join', return_value=output_path):
        result_path = remove_abusive_data(serialized_data)
        result_data = pl.read_parquet(result_path)
        assert 'abuse_free_complaint' in result_data.columns
        assert result_data['abuse_free_complaint'][0] == 'This is a <abusive_data> in the complaint'

@patch('preprocessing.pl.read_parquet')
def test_remove_abusive_data_no_abusive_words(mock_read_parquet, tmpdir):
    # Mock empty abusive words datasets
    abusive_words_eng = pl.DataFrame({'profanity': []})
    abusive_words_hindi = pl.DataFrame({'words': []})
    mock_read_parquet.return_value = abusive_words_eng

    data = pl.DataFrame({
        'complaint': ['Clean complaint text'],
        'complaint_hindi': ['']
    })
    serialized_data = data.serialize(format="json")
    output_path = os.path.join(tmpdir, 'preprocessed_dataset.parquet')

    with patch('preprocessing.os.path.join', return_value=output_path):
        result_path = remove_abusive_data(serialized_data)
        result_data = pl.read_parquet(result_path)
        assert result_data['abuse_free_complaint'][0] == 'Clean complaint text'

@patch('preprocessing.pl.read_parquet')
def test_remove_abusive_data_hindi(mock_read_parquet, tmpdir):
    abusive_words_eng = pl.DataFrame({'profanity': []})
    abusive_words_hindi = pl.DataFrame({'words': ['खराबशब्द']})
    def side_effect(*args, **kwargs):
        if 'profanity_bank_dataset.parquet' in args[0]:
            return abusive_words_eng
        elif 'hindi_abuse_words.parquet' in args[0]:
            return abusive_words_hindi
    mock_read_parquet.side_effect = side_effect

    data = pl.DataFrame({
        'complaint': [''],
        'complaint_hindi': ['यह एक खराबशब्द है']
    })
    serialized_data = data.serialize(format="json")
    output_path = os.path.join(tmpdir, 'preprocessed_dataset.parquet')

    with patch('preprocessing.os.path.join', return_value=output_path):
        result_path = remove_abusive_data(serialized_data)
        result_data = pl.read_parquet(result_path)
        assert result_data['abuse_free_complaint_hindi'][0] == 'यह एक <abusive_data> है'

@patch('preprocessing.pl.read_parquet')
def test_remove_abusive_data_empty_complaints(mock_read_parquet, tmpdir):
    abusive_words_eng = pl.DataFrame({'profanity': ['badword']})
    abusive_words_hindi = pl.DataFrame({'words': ['खराबशब्द']})
    mock_read_parquet.return_value = abusive_words_eng

    data = pl.DataFrame({
        'complaint': [''],
        'complaint_hindi': ['']
    })
    serialized_data = data.serialize(format="json")
    output_path = os.path.join(tmpdir, 'preprocessed_dataset.parquet')

    with patch('preprocessing.os.path.join', return_value=output_path):
        result_path = remove_abusive_data(serialized_data)
        result_data = pl.read_parquet(result_path)
        assert result_data['abuse_free_complaint'][0] == ''
        assert result_data['abuse_free_complaint_hindi'][0] == ''

# Function 7: insert_data_to_bigquery

def test_insert_data_to_bigquery_dry_run():
    with patch('preprocessing.DRY_RUN', True):
        # Should return early without any action
        result = insert_data_to_bigquery('dummy_path')
        assert result is None  # Function returns None

@patch('preprocessing.bigquery.Client')
def test_insert_data_to_bigquery_success(mock_bq_client):
    mock_client_instance = MagicMock()
    mock_bq_client.return_value = mock_client_instance

    with patch('preprocessing.DRY_RUN', False):
        insert_data_to_bigquery('dummy_path')
        assert mock_client_instance.load_table_from_dataframe.called

@patch('preprocessing.bigquery.Client')
def test_insert_data_to_bigquery_failure(mock_bq_client):
    mock_client_instance = MagicMock()
    mock_client_instance.load_table_from_dataframe.side_effect = Exception('Insertion failed')
    mock_bq_client.return_value = mock_client_instance

    with patch('preprocessing.DRY_RUN', False):
        with pytest.raises(Exception) as exc_info:
            insert_data_to_bigquery('dummy_path')
        assert 'Insertion failed' in str(exc_info.value)