import os
import re
import io
import logging
import polars as pl
from lingua import Language, LanguageDetectorBuilder
from rbloom import Bloom
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    filename='preprocessinglog.txt',
    level=logging.INFO,
    filemode='a'
    )

def data_loading() -> str:
    data_path = os.path.join(os.path.dirname(__file__), "../../data/JPMORGAN_CHASE_COMPLAINTS.csv")
    try:
        # Serialize and return the loaded dataset
        dataset = pl.read_csv(data_path).serialize(format='json')
        logging.info('dataset loading complete!')
        return dataset
    except Exception as e:
        print(f'Exception:\n{e}')
        raise("Error With Dataset Loading")

def minimum_word_check(
        dataset: str,
        min_word_length: int
        ) -> str:
    # Deserialize the dataset and filter out the records that don't match minimum word length
    dataset = pl.DataFrame.deserialize(io.StringIO(dataset), format='json')
    # Filtering based on Number of words
    dataset = dataset.with_columns(
        num_words = pl.col('Consumer complaint narrative').str.split(' ').list.len()
        ).filter(
            pl.col('num_words') > min_word_length
            )
    logging.info('Min word check complete!')
    # Serialize and return dataset
    return dataset.serialize(format='json')

def detect_language(
    dataset: str
    ) -> str:
    
    # Deserialize the dataset and filter out the records that don't meet the language criteria
    dataset = pl.DataFrame.deserialize(io.StringIO(dataset), format='json')
    
    # Setup language detector
    languages = list(Language.all_spoken_ones())
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    # Multi-threading the language detection task
    language_detected = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks to the thread pool
        future_to_sentence = {executor.submit(detector.detect_language_of, sentence): sentence for sentence in dataset['Consumer complaint narrative'].to_list()}

        # Collect the results as they complete
        for future in as_completed(future_to_sentence):
            language_detected.append(future.result())

    language_detected = [item.iso_code_639_1.name for item in language_detected]
    # Adding and filtering based on language column
    dataset.with_columns(
        pl.Series(name='language', values=language_detected)
    ).filter(
        pl.col('language').is_in(['HI', 'EN'])
    ).drop(['language'])

    logging.info('Language detection complete!')
    # Serialize and return dataset
    return dataset.serialize(format='json')

def aggregate_filtered_task(
        dataset_a: str,
        dataset_b: str,
        ) -> None:
    
    output_path = os.path.join(os.path.dirname(__file__), "../../data/PREPROCESSED_JPMORGAN_CHASE_COMPLAINTS.parquet")

    # Deserialize Set A and Set B and Join them based on their company id
    dataset_a = pl.DataFrame.deserialize(io.StringIO(dataset_a), format='json')
    dataset_b = pl.DataFrame.deserialize(io.StringIO(dataset_b), format='json')
    # Dont select additional columns
    dataset_joined = dataset_a.join(dataset_b, on='Complaint ID', how='inner').select(['Date received','Product','Sub-product','Issue','Sub-issue','Consumer complaint narrative','Company public response','Company', 'State','ZIP code','Tags','Consumer consent provided?','Submitted via','Date sent to company','Company response to consumer','Timely response?','Consumer disputed?','Complaint ID','Department',])
    # Write the output to the output file
    dataset_joined.write_parquet(output_path, compression='gzip')

def data_cleaning() -> str:
    # Read the Dataset processed from other DAG
    datapath = os.path.join(os.path.dirname(__file__), "../../data/PREPROCESSED_JPMORGAN_CHASE_COMPLAINTS.parquet")
    dataset = pl.read_parquet(datapath)

    # Lowercase complaints
    dataset = dataset.with_columns(
        pl.col('Consumer complaint narrative').str.to_lowercase().alias('Consumer complaint narrative')
    )

    # Deduplicate the records
    dataset = dataset.unique(subset=['Product', 'Sub-product', 'Consumer complaint narrative'], maintain_order=True)
    
    # Drop Nulls
    dataset = dataset.drop_nulls(subset=['Product', 'Sub-product', 'Department', 'Consumer complaint narrative'])

    # Serialize and return dataset
    return dataset.serialize(format='json')


def remove_special_characters(dataset: str) -> str:
    # Deserialize dataset
    dataset = pl.DataFrame.deserialize(io.StringIO(dataset), format='json')
    # Removing Special Characters
    dataset = dataset.with_columns(
        pl.col('Consumer complaint narrative').map_elements(lambda x: re.sub(r'[^A-Za-z0-9\s]', '', x), return_dtype=pl.Utf8).alias('Consumer complaint narrative')
    )
    # Serialize and return dataset
    return dataset.serialize(format='json')


def remove_abusive_data(
        dataset: str,
        abuse_placeholder: str = 'yyy'
        ) -> str:
    # Deserialize dataset
    dataset = pl.DataFrame.deserialize(io.StringIO(dataset), format='json')
    # Set Output Path
    output_path = os.path.join(os.path.dirname(__file__), "../../data/PREPROCESSED_JPMORGAN_CHASE_COMPLAINTS.parquet")

    # Setup Bloom Filter
    profane_set = set()
    profanity_bloom = Bloom(200_000, 0.01)

    # Load abusive words
    abusive_words_path = os.path.join(os.path.dirname(__file__), "../../data/profanity_bank_dataset.csv")
    abusive_words = pl.read_csv(abusive_words_path).with_columns(
        pl.col('Profanity').str.to_lowercase()
        ).filter(
            pl.col('Profanity').is_not_null()
            )['Profanity'].to_list()

    for item in abusive_words:
        profanity_bloom.add(item)
        profane_set.add(item)
    
    # Tokenize Complaints
    tokenized_complaints = dataset.with_columns(
        pl.col('Consumer complaint narrative').str.split(' ')
    )['Consumer complaint narrative'].to_list()

    # Remove abusive words
    cleaned_records = []
    for idx, record in enumerate(tokenized_complaints):
        clean_record = []
        for w in record:
            if w not in profanity_bloom:
                clean_record.append(w)
            # The 10% error case or the token is actually a profane word
            elif w in profane_set:
                clean_record.append(abuse_placeholder)
            else:
                clean_record.append(w)
        
        cleaned_records.append(" ".join(clean_record))

    dataset.with_columns(
        pl.Series(name='abuse_free_complaints', values=cleaned_records)
    )
    # Save the processed results to output path
    dataset.write_parquet(output_path, compression='gzip')