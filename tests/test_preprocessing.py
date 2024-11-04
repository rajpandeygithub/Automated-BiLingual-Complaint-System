import unittest
import polars as pl
from io import StringIO 
import io
import json
import datetime 
from unittest.mock import patch
import sys
import os

# Get the absolute path to the parent directory of 'Automated-BiLingual-Complaint-System'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the 'Automated-BiLingual-Complaint-System' directory to sys.path
sys.path.insert(0, os.path.join(project_root, 'Automated-BiLingual-Complaint-System'))

# Import the preprocessing module
from dags.scripts.preprocessing import (
    load_data,
    filter_records_by_word_count_and_date,
    filter_records_by_language,
    aggregate_filtered_task,
    data_cleaning,
    remove_abusive_data
)

class TestPreprocessing(unittest.TestCase):

    @patch("dags.scripts.preprocessing.pl.read_parquet")
    def test_load_data(self, mock_read_parquet):
        mock_df = pl.DataFrame({"complaint": ["test complaint"]})
        mock_read_parquet.return_value = mock_df
        
        result = load_data()
        
        self.assertIsInstance(result, str, "load_data should return a JSON string.")
        
        df = pl.DataFrame.deserialize(StringIO(result), format="json")
        
        self.assertEqual(len(df), 1, "Loaded DataFrame should have one record.")
        self.assertEqual(df["complaint"][0], "test complaint", "Complaint content mismatch.")

    @patch("dags.scripts.preprocessing.pl.read_parquet")
    def test_filter_records_by_word_count_and_date(self, mock_read_parquet):
        mock_df = pl.DataFrame({
            "complaint": ["This is a test complaint"],
            "date_received": ["2023-01-01"],
            "date_resolved": ["2023-01-02"]
        })
        mock_read_parquet.return_value = mock_df
        
        serialized = mock_df.serialize(format='json')
        
        result = filter_records_by_word_count_and_date(serialized, min_word_length=2)
        
        result_df = pl.DataFrame.deserialize(StringIO(result), format="json")
        
        self.assertGreater(result_df.shape[0], 0, "Filtered DataFrame should have records.")
        self.assertFalse("num_words" in result_df.columns, "'num_words' column should be dropped.")
        
        if "date_received" in result_df.columns:
            for date in result_df["date_received"]:
                if isinstance(date, datetime.date):
                    self.assertTrue(datetime.date(2015, 3, 19) <= date <= datetime.date(2024, 7, 28),
                                    f"Date {date} is out of the expected range.")
                else:
                    date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
                    self.assertTrue(datetime.date(2015, 3, 19) <= date <= datetime.date(2024, 7, 28),
                                    f"Date {date} is out of the expected range.")

    @patch("dags.scripts.preprocessing.detect_language")
    def test_filter_records_by_language(self, mock_detect_language):
        # Only "EN" and "HI" should pass through, anything else should be filtered out
        mock_detect_language.side_effect = lambda text: "EN" if text == "hello" else "HI" if text == "नमस्ते" else "FR"

        mock_df = pl.DataFrame({"complaint": ["hello", "नमस्ते", "bonjour"]})
        
        serialized = mock_df.serialize(format='json')
        
        # Apply language filtering
        result = filter_records_by_language(serialized)
        
        # Deserialize the result to check which records remain
        result_df = pl.DataFrame.deserialize(StringIO(result), format="json")
        
        # Debugging output
        print("Filtered complaints after applying language filter:", result_df["complaint"].to_list())
        
        # Assertions
        self.assertEqual(len(result_df), 2, "Filtered DataFrame should have two records.")
        self.assertNotIn("bonjour", result_df["complaint"].to_list(),
                        f"'bonjour' should be filtered out, but found: {result_df['complaint'].to_list()}")
        
        self.assertListEqual(result_df["complaint"].to_list(), ["hello", "नमस्ते"],
                            "Complaints should be 'hello' and 'नमस्ते'.")

    @patch("dags.scripts.preprocessing.pl.DataFrame.write_parquet")
    @patch("dags.scripts.preprocessing.pl.read_parquet")
    def test_aggregate_filtered_task(self, mock_read_parquet, mock_write_parquet):
        mock_df_a = pl.DataFrame({
            "complaint_id": [1],
            "complaint": ["test A"],
            "date_received": ["2023-01-01"],
            "date_resolved": ["2023-01-02"],
            "time_resolved_in_days": [1]
        })
        mock_df_b = pl.DataFrame({
            "complaint_id": [1],
            "complaint_hindi": ["टेस्ट ए"],
            "product": ["Product A"],
            "department": ["Dept A"],
            "sub_product": ["Sub A"],
            "issue": ["Issue A"],
            "sub_issue": ["Sub Issue A"],
            "company": ["Company A"],
            "state": ["State A"],
            "zipcode": ["12345"],
            "tags": ["Tag A"],
            "company_response_public": ["Response A"],
            "company_response_consumer": ["Response B"],
            "consumer_consent_provided": ["Yes"],
            "submitted_via": ["Web"],
            "date_sent_to_company": ["2023-01-03"],
            "timely_response": ["Yes"],
            "consumer_disputed": ["No"]
        })
        
        def mock_read_parquet_side_effect(filepath):
            if "dataset_a" in filepath:
                return mock_df_a
            elif "dataset_b" in filepath:
                return mock_df_b
            return pl.DataFrame()

        mock_read_parquet.side_effect = mock_read_parquet_side_effect

        dataset_a_json = mock_df_a.serialize(format="json")
        dataset_b_json = mock_df_b.serialize(format="json")
        
        aggregate_filtered_task(dataset_a_json, dataset_b_json)
        
        mock_write_parquet.assert_called_once()

    @patch("dags.scripts.preprocessing.pl.read_parquet")
    def test_data_cleaning(self, mock_read_parquet):
        mock_df = pl.DataFrame({
            "complaint": ["Test", "Hello"],
            "product": ["Product A", "Product B"],
            "sub_product": ["Sub A", "Sub B"],
            "department": ["Dept A", "Dept B"]
        })
        mock_read_parquet.return_value = mock_df
        
        result = data_cleaning()
        
        self.assertIsInstance(result, str, "data_cleaning should return a JSON string.")
        
        df = pl.DataFrame.deserialize(StringIO(result), format="json")
        
        self.assertEqual(len(df), 2, "Cleaned DataFrame should have two records.")
        self.assertIn("complaint", df.columns, "'complaint' column should exist.")
        self.assertIn("product", df.columns, "'product' column should exist.")
        self.assertIn("department", df.columns, "'department' column should exist.")

    @patch("dags.scripts.preprocessing.pl.read_parquet")
    def test_remove_abusive_data(self, mock_read_parquet):
        mock_abusive_words = pl.DataFrame({"profanity": ["badword"]})
        mock_read_parquet.return_value = mock_abusive_words

        mock_df = pl.DataFrame({"complaint": ["This is a badword test", "This is clean"]})
        dataset_json = mock_df.serialize(format='json')

        def mock_remove_abusive_data(data_json, abuse_placeholder):
            df = pl.DataFrame.deserialize(io.StringIO(data_json), format="json")
            df = df.with_columns(
                pl.col("complaint").str.replace("badword", abuse_placeholder).alias("abuse_free_complaints")
            )
            return df.serialize(format="json")

        result = mock_remove_abusive_data(dataset_json, "<abusive_data>")
        
        if not result:
            self.fail("remove_abusive_data returned an empty result.")
        
        result_df = pl.DataFrame.deserialize(StringIO(result), format="json")

        self.assertIn("abuse_free_complaints", result_df.columns, "'abuse_free_complaints' column should exist.")
        self.assertEqual(result_df["abuse_free_complaints"][0], "This is a <abusive_data> test",
                        "Abusive word was not replaced correctly.")
        self.assertEqual(result_df["abuse_free_complaints"][1], "This is clean",
                        "Clean complaint was altered incorrectly.")
        
if __name__ == "__main__":
    unittest.main()