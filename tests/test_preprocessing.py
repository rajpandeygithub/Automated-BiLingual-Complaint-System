import unittest
import polars as pl
import io
import sys
import os

# Adjusting the path to include the scripts directory
sys.path.append(os.path.join(os.path.dirname(__file__), '../dags/scripts'))
from preprocessing import (
    data_loading,
    minimum_word_check,
    detect_language,
    remove_special_characters,
    remove_abusive_data
)

class TestPreprocessing(unittest.TestCase):
    
    def test_data_loading(self):
        dataset = data_loading()
        self.assertIsInstance(dataset, str, "Data should be serialized as JSON string")

    def test_minimum_word_check(self):
        dataset = data_loading()
        filtered_data = minimum_word_check(dataset, min_word_length=3)
        deserialized_data = pl.DataFrame.deserialize(io.StringIO(filtered_data), format='json')
        
        for record in deserialized_data['num_words']:
            self.assertGreater(record, 3, "All records should meet minimum word count requirement")

    def test_detect_language(self):
        dataset = data_loading()
        language_filtered_data = detect_language(dataset)
        deserialized_data = pl.DataFrame.deserialize(io.StringIO(language_filtered_data), format='json')
        
        for lang in deserialized_data['language']:
            self.assertIn(lang, ['HI', 'EN'], "Detected language should be either 'HI' or 'EN'")

    def test_remove_special_characters(self):
        dataset = data_loading()
        cleaned_data = remove_special_characters(dataset)
        deserialized_data = pl.DataFrame.deserialize(io.StringIO(cleaned_data), format='json')
        
        for complaint in deserialized_data['Consumer complaint narrative']:
            self.assertNotRegex(complaint, r'[^A-Za-z0-9\s]', "Special characters should be removed")

    def test_remove_abusive_data(self):
        dataset = data_loading()
        cleaned_data = remove_abusive_data(dataset, abuse_placeholder='yyy')
        deserialized_data = pl.DataFrame.deserialize(io.StringIO(cleaned_data), format='json')
        
        for complaint in deserialized_data['abuse_free_complaints']:
            self.assertNotIn('badword', complaint, "Abusive words should be replaced with 'yyy'")

if __name__ == '__main__':
    unittest.main()