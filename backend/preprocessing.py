import re
import nltk
import sys
import polars as pl
import logging
from typing import Dict, List
from nltk.corpus import stopwords
from rapidfuzz import process, fuzz
from fast_langdetect import detect_language

nltk.download('stopwords')
abusive_words_path_eng = "https://storage.googleapis.com/mlops-group6-raw-data/profanity_bank_dataset.parquet"
abuse_words = pl.read_parquet(abusive_words_path_eng)["profanity"].to_list()

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format=log_format,   # Set the log format
    handlers=[
        logging.StreamHandler(sys.stdout),  # Output logs to stdout
        logging.FileHandler("app.log")      # Output logs to a file
    ]
)

logger = logging.getLogger("preprocessor_log")

class DataValidationPipeline:
    def __init__(self, checks: Dict[str, str]):
        self.min_words: int = checks.get('min_words')
        self.max_words: int = checks.get('max_words')
        self.allowed_languages: List[str] = checks.get("allowed_languages")

    def _accepted_word_count_check(self, num_words: int) -> bool:
        return self.min_words < num_words < self.max_words
    
    def _language_check(self, text: str) -> bool:
        text_language = detect_language(text)
        self.text_language = text_language
        return text_language in self.allowed_languages
    
    def get_recognised_language(self):
        return self.text_language

    def is_valid(self, text: str) -> bool:
        text = text.lower()
        words = text.split(' ')
        num_words_check = self._accepted_word_count_check(len(words))
        language_check = self._language_check(text)
        return num_words_check and language_check

class DataTransformationPipeline:
    def __init__(self, abuse_words: List[str] = abuse_words):
        self._eng_abusive_words = abuse_words
        self._abuse_placeholder = "<abusive_data>"
        self._eng_stopwords = stopwords.words('english')
        self._abusive_word_threshold = 90

    def _remove_abusive_words(self, text: str) -> str:
        words = [w for w in text.split(' ') if w not in self._eng_stopwords]
        redacted_words = []
        for w in words:
            similar_word = process.extractOne(
                w, self._eng_abusive_words, 
                score_cutoff=self._abusive_word_threshold, 
                scorer=fuzz.token_sort_ratio
                )
            if similar_word:
                redacted_words.append(w)
        
        pattern = r'\b(' + '|'.join(map(re.escape, redacted_words)) + r')\b'
        redacted_text = re.sub(pattern, self._abuse_placeholder, text, flags=re.IGNORECASE)
        return redacted_text

    def _process_hindi(self, text: str) -> str:
        return text
    def _process_english(self, text: str) -> str:
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Remove abusive words
        text = self._remove_abusive_words(text)
        # PII redaction
        return text.strip()

    def process_text(self, text: str, language: str) -> str:
        processed_text = ""
        if language == 'EN':
            logger.info(f'Preprocessing English Text')
            processed_text = self._process_english(text.lower())
        elif language == 'HI':
            logger.info(f'Preprocessing Hindi Text')
            processed_text = self._process_hindi(text)
        return processed_text


if __name__ == '__main__':
    pipeline = DataTransformationPipeline()
    text = 'hello, world! This is a test complaint. Please clean me'
    text = pipeline.process_text(text, language='EN')
    print(text)