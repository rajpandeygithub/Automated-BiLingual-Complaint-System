import re
import nltk
import sys
import polars as pl
import logging
from typing import Dict, List
from nltk.corpus import stopwords
from rapidfuzz import process, fuzz
from bloom_filter2 import BloomFilter
from fast_langdetect import detect_language

nltk.download("stopwords")
abusive_words_path_eng = "https://storage.googleapis.com/mlops-group6-raw-data/profanity_bank_dataset.parquet"
abusive_words_path_hindi = "https://storage.googleapis.com/mlops-group6-raw-data/hindi_abuse_words.parquet"

abuse_words_english = pl.read_parquet(abusive_words_path_eng)["profanity"].to_list()
abusive_words_hindi = pl.read_parquet(abusive_words_path_hindi)["words"].to_list()

profane_set_hindi = set()
profanity_bloom_hindi = BloomFilter(max_elements=500, error_rate=0.1)

for word in abusive_words_hindi:
    profanity_bloom_hindi.add(word)
    profane_set_hindi.add(word)

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format=log_format,  # Set the log format
    handlers=[
        logging.StreamHandler(sys.stdout),  # Output logs to stdout
        logging.FileHandler("app.log"),  # Output logs to a file
    ],
)

logger = logging.getLogger("preprocessor_log")


# Define regex patterns for different PII entities
PATTERNS = {
    "bank_account_numbers": r"\b\d{10,12}\b",
    "bank_routing_numbers": r"\b\d{9}\b",
    "credit_card_numbers": [
        r"\b(?:4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b",  # Visa
        r"\b(?:5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b",  # Mastercard
        r"\b(?:3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5})\b",  # American Express
        r"\b(?:6(?:011|5\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b",  # Discover
        r"\b(?:3(?:0[0-5]|[68]\d)\d{11,14})\b",  # Diners Club (updated)
        r"\b(?:(?:2131|1800|35\d{3})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b",  # JCB
        r"\b(?:(?:5[0678]\d\d|6304|6390|67\d\d)\d{8,15})\b",  # Maestro
        r"\b(?:\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b",  # Generic 16-digit pattern
    ],
    "transaction_date_or_date_of_birth": [
        r"\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])[-/](?:\d{4}|\d{2})\b",  # MM/DD/YYYY or MM/DD/YY
        r"\b(?:0[1-9]|[12][0-9]|3[01])[-/](?:0[1-9]|1[0-2])[-/](?:\d{4}|\d{2})\b",  # DD/MM/YYYY or DD/MM/YY
        r"\b\d{4}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])\b",  # YYYY/MM/DD or YYYY-MM-DD
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}\b",  # 1st January 2024
        r"\b\w+\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b",  # January 31, 2024
        r"\b\w{3}\s+\d{1,2},\s+\d{4}\b",  # Jan 31, 2024
        r"\b\d{1,2}\s+\w{3}\s+\d{4}\b",  # 31 Jan 2024
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}\s+\d{1,2}:\d{2}\b",  # 1st January 2024 14:30
        r"\b\w+\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\b",  # January 31, 2024 2:30 PM
        r"\b\d{4}-\d{1,2}-\d{1,2}T\d{1,2}:\d{2}:\d{2}Z\b",  # ISO 8601 Format
    ],
    "transaction_amounts": [
        r"\$\s*\d+(?:,\d{3})*(?:\.\d{2})?",
        r"\b\d+(?:,\d{3})*(?:\.\d{2})?\b",
    ],
    "ssn_tin": r"\b\d{3}-\d{2}-\d{4}\b",
    "ein": r"\b\d{2}-\d{7}\b",
    "passport_numbers": r"\b[A-Z]{1}\d{7}\b",
    "email_address": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone_numbers": [
        r"\+?\b(?:1[-.\\s]?)?(?:\(?[2-9]\d{2}\)?[-.\\s]?)?[2-9]\d{2}[-.\\s]?\d{4}\b|\b\d{10}\b|\b\(\d{3}\)\s?\d{3}[-.\\s]?\d{4}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        r"(\+\d{1,2}\s?)?(\(\d{3}\)\s?)?[\d\-\.\s]{10,}",
    ],
    "dates_of_birth": r"\b(?:\+?1[-.\s]?)?(?:\(?[2-9]\d{2}\)?[-.\s]?)?[2-9]\d{2}[-.\s]?\d{4}\b|\b\d{10}\b|\b\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "home_address": r"\b\d{1,9},\s\w+(?:\s\w+)*,\s\w+(?:\s\w+)*,\s[A-Z]{2}\s\d{5}(?:-\d{4})?\b",
    "race": r"\b(?:White|Black|Asian|Native American|Pacific Islander|Multiracial|Biracial)\b",
    "ethnicity": r"\b(?:Hispanic|Latino|Latinx|African American|Caucasian|Arab|Jewish|Slavic|Celtic|Germanic|Scandinavian|Mediterranean|Ashkenazi|Sephardic)\b",
    "region": r"\b(?:European|African|Middle Eastern|East Asian|South Asian|Southeast Asian|Central Asian|North American|South American|Central American|Caribbean|Oceanian|Polynesian|Micronesian|Melanesian)\b",
    "gender": r"\b(?:male|female|man|woman|boy|girl|non-binary|genderqueer|transgender|trans|cisgender|cis|agender|bigender|genderfluid)\b",
}


class DataValidationPipeline:
    def __init__(self, checks: Dict[str, str]):
        self.min_words: int = checks.get("min_words")
        self.max_words: int = checks.get("max_words")
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
        words = text.split(" ")
        num_words_check = self._accepted_word_count_check(len(words))
        language_check = self._language_check(text)
        return num_words_check and language_check


class DataTransformationPipeline:
    def __init__(
        self,
        abuse_words_english: List[str] = abuse_words_english,
        abuse_bloomfilter_hindi: BloomFilter = profanity_bloom_hindi,
        pii_patterns: Dict[str, str] = PATTERNS,
    ):
        self._eng_abusive_words = abuse_words_english
        self._hin_abusive_bf = abuse_bloomfilter_hindi
        self._abuse_placeholder = "<abusive_data>"
        self._eng_stopwords = stopwords.words("english")
        self._abusive_word_threshold = 90
        self.pii_patterns = pii_patterns

    def _remove_english_pii(self, text: str):

        pronoun_map = {
            r"\bhe\b": "they",
            r"\bshe\b": "they",
            r"\bhim\b": "them",
            r"\bher\b": "them",
            r"\bhis\b": "their",
            r"\bhers\b": "theirs",
            r"\bhimself\b": "themself",
            r"\bherself\b": "themself",
        }

        for key, pattern in self.pii_patterns.items():
            if isinstance(
                pattern, list
            ):  # For patterns that are lists (like credit cards)
                for p in pattern:
                    text = re.sub(p, f"<{key}>", text, flags=re.IGNORECASE)
            else:
                text = re.sub(pattern, f"<{key}>", text, flags=re.IGNORECASE)

        # Gender neutral PII
        for pattern, _ in pronoun_map.items():
            text = re.sub(pattern, "<gender>", text, flags=re.IGNORECASE)

        return text

    def _remove_english_abusive_words(self, text: str) -> str:
        words = [w for w in text.split(" ") if w not in self._eng_stopwords]
        redacted_words = []
        for w in words:
            similar_word = process.extractOne(
                w,
                self._eng_abusive_words,
                score_cutoff=self._abusive_word_threshold,
                scorer=fuzz.token_sort_ratio,
            )
            if similar_word:
                redacted_words.append(w)

        pattern = r"\b(" + "|".join(map(re.escape, redacted_words)) + r")\b"
        redacted_text = re.sub(
            pattern, self._abuse_placeholder, text, flags=re.IGNORECASE
        )
        return redacted_text
    
    def _remove_hindi_abusive_words(self, text: str) -> str:
        redacted_words = [w for w in text.split(" ") if w in self._hin_abusive_bf]
        logger.info(f'Hindi Bad words: {redacted_words}')
        if len(redacted_words) > 0:
            pattern = r"\b(" + "|".join(map(re.escape, redacted_words)) + r")\b"
            text = re.sub(
                pattern, self._abuse_placeholder, text
            )
        return text

    def _process_hindi(self, text: str) -> str:
        # Remove Abusive words
        text = self._remove_hindi_abusive_words(text)
        # Remove any tailing spaces
        return text.strip()

    def _process_english(self, text: str) -> str:
        # Remove special characters
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        # Remove abusive words
        text = self._remove_english_abusive_words(text)
        # PII redaction
        text = self._remove_english_pii(text)
        # Remove any tailing spaces
        return text.strip()

    def process_text(self, text: str, language: str) -> str:
        processed_text = ""
        if language == "EN":
            logger.info(f"Preprocessing English Text")
            processed_text = self._process_english(text.lower())
        elif language == "HI":
            logger.info(f"Preprocessing Hindi Text")
            processed_text = self._process_hindi(text)
        return processed_text


if __name__ == "__main__":
    pipeline = DataTransformationPipeline()
    text = "hello, world! This is a test complaint. Please clean me"
    text = pipeline.process_text(text, language="EN")
    print(text)
