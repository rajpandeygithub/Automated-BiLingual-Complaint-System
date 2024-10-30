import unittest
import polars as pl
import io
from unittest.mock import patch
import sys
import os

# Adjusting the path to include the scripts directory
sys.path.append(os.path.join(os.path.dirname(__file__), '../dags/scripts'))

from preprocessing import (
    load_data,
    filter_records_by_word_count,
    filter_records_by_language,
    aggregate_filtered_task,
    data_cleaning,
    remove_abusive_data
)

class TestPreprocessing(unittest.TestCase):

    @patch("preprocessing.pl.read_parquet")
    def test_load_data(self, mock_read_parquet):
        mock_df = pl.DataFrame({"complaint": ["test complaint"]})
        mock_read_parquet.return_value = mock_df
        result = load_data()
        self.assertIsInstance(result, str)

    @patch("preprocessing.pl.DataFrame.deserialize")
    def test_filter_records_by_word_count(self, mock_deserialize):
        mock_df = pl.DataFrame({"complaint": ["This is a test"]})
        mock_deserialize.return_value = mock_df
        dataset = mock_df.serialize(format="json")
        result = filter_records_by_word_count(dataset, min_word_length=2)
        result_df = pl.DataFrame.deserialize(io.StringIO(result), format="json")
        self.assertGreater(result_df.shape[0], 0)
        self.assertFalse("num_words" in result_df.columns)

    @patch("preprocessing.pl.DataFrame.deserialize")
    def test_filter_records_by_language(self, mock_deserialize):
        mock_df = pl.DataFrame({"complaint": ["hello", "नमस्ते"]})
        mock_deserialize.return_value = mock_df
        dataset = mock_df.serialize(format="json")
        result = filter_records_by_language(dataset)
        result_df = pl.DataFrame.deserialize(io.StringIO(result), format="json")
        self.assertIn("language", result_df.columns)
        for lang in result_df["language"].to_list():
            self.assertIn(lang, ["HI", "EN"])

    @patch("preprocessing.pl.DataFrame.deserialize")
    @patch("preprocessing.pl.DataFrame.write_parquet")
    def test_aggregate_filtered_task(self, mock_write_parquet, mock_deserialize):
        mock_df_a = pl.DataFrame({
            "complaint_id": [1],
            "complaint": ["test A"],
            "date_received": ["2023-01-01"],
            "date_resolved": ["2023-01-02"]
        })
        mock_df_b = pl.DataFrame({
            "complaint_id": [1],
            "complaint_hindi": ["टेस्ट बी"],
            "date_received": ["2023-01-01"],
            "date_resolved": ["2023-01-02"]
        })
        mock_deserialize.side_effect = [mock_df_a, mock_df_b]
        dataset_a = mock_df_a.serialize(format="json")
        dataset_b = mock_df_b.serialize(format="json")
        aggregate_filtered_task(dataset_a, dataset_b)
        mock_write_parquet.assert_called_once()

    @patch("preprocessing.pl.read_parquet")
    def test_data_cleaning(self, mock_read_parquet):
        mock_df = pl.DataFrame({
            "complaint": ["Test", "Hello"],
            "product": ["Product A", "Product B"],
            "sub_product": ["Sub A", "Sub B"],
            "department": ["Dept A", "Dept B"]
        })
        mock_read_parquet.return_value = mock_df
        result = data_cleaning()
        result_df = pl.DataFrame.deserialize(io.StringIO(result), format="json")
        self.assertIsInstance(result, str)

    @patch("preprocessing.pl.DataFrame.deserialize")
    @patch("preprocessing.pl.DataFrame.write_parquet")
    @patch("preprocessing.pl.read_parquet")
    def test_remove_abusive_data(self, mock_read_parquet, mock_write_parquet, mock_deserialize):
        mock_abusive_words = pl.DataFrame({"profanity": ["badword"]})
        mock_read_parquet.return_value = mock_abusive_words
        mock_df = pl.DataFrame({"complaint": ["This is a badword test"]})
        mock_deserialize.return_value = mock_df
        dataset = mock_df.serialize(format="json")
        result = remove_abusive_data(dataset)
        result_df = pl.DataFrame.deserialize(io.StringIO(result), format="json")
        self.assertIn("abuse_free_complaints", result_df.columns)
        self.assertTrue("yyy" in result_df["abuse_free_complaints"][0])

if __name__ == "__main__":
    unittest.main()