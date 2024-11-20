import os

class PreprocessingPieline:
    def __init__(self, min_word_length: int = 5):
        pass
    def process(self, text: str):
        print(f'Hello')
        return text.lower()


if __name__ == '__main__':
    pipeline = PreprocessingPieline()
    text = 'hello, world! This is a test complaint. Please clean me'
    text = pipeline.process(text)
    print(text)