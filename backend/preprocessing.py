import os
from typing import Dict, List
from fast_langdetect import detect_language


class DataValidationPipeline:
    def __init__(self, checks: Dict[str, str]):
        self.min_words: int = checks.get('min_words')
        self.max_words: int = checks.get('max_words')
        self.allowed_languages: List[str] = checks.get("allowed_languages")

    def _accepted_word_count_check(self, num_words: int) -> bool:
        return self.min_words < num_words < self.max_words
    
    def _language_check(self, text: str) -> bool:
        text_language = detect_language(text)
        return text_language in self.allowed_languages

    def is_valid(self, text: str) -> bool:
        text = text.lower()
        words = text.split(' ')
        num_words_check = self._accepted_word_count_check(len(words))
        language_check = self._language_check(text)
        return num_words_check and language_check

class DataTransformationPipeline:
    def __init__(self, min_word_length: int = 5):
        pass
    def process(self, text: str):
        return text.lower()


if __name__ == '__main__':
    pipeline = DataTransformationPipeline()
    text = 'hello, world! This is a test complaint. Please clean me'
    text = pipeline.process(text)
    print(text)