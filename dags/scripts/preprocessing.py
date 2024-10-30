import os
import re
import io
import logging
import polars as pl
from lingua import Language, LanguageDetectorBuilder
from rbloom import Bloom
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(filename="preprocessinglog.txt", level=logging.INFO, filemode="a")


def load_data() -> str:
    """
    Load the JPMorgan Chase complaints dataset in Parquet format.
    Returns:
        str: Serialized dataset in JSON format.
    Raises:
        Exception: If there is an error loading the dataset.
    """
    data_path = os.path.join(os.path.dirname(__file__), "../../data/dataset.parquet")
    try:
        dataset = pl.read_parquet(data_path).serialize(format="json")
        return dataset
    except Exception as error:
        print(f"Exception:\n{error}")
        raise Exception("Error With Dataset Loading")


def filter_records_by_word_count_and_date(dataset: str, min_word_length: int) -> str:
    """
    Remove records from the dataset that do not meet the minimum word count
    in the 'complaint' column.
    Args:
        dataset (str): Serialized dataset in JSON format.
        min_word_length (int): Minimum word count required for each record.
    Returns:
        str: Serialized dataset in JSON format with records removed if they
             have fewer words than the specified minimum.
    """
    # Deserialize the dataset
    dataset = pl.DataFrame.deserialize(io.StringIO(dataset), format="json")

    # Filter records based on the minimum word count and remove the count column
    dataset = (
    dataset
    .with_columns(num_words=pl.col("complaint").str.split(" ").arr.lengths())
    .filter(pl.col("num_words") > min_word_length)
    .drop("num_words")
    .with_columns(pl.col("date_received").str.strptime(pl.Date, "%Y-%m-%d").alias("date_received"))
    .filter(
        (pl.col("date_received") >= pl.date("2020-01-01")) &
        (pl.col("date_received") <= pl.date("2023-12-31"))
        )
    )

    # Serialize and return the filtered dataset
    return dataset.serialize(format="json")


def filter_records_by_language(dataset: str) -> str:
    """
    Detect the language of each complaint in the dataset and filter out records
    that do not meet the specified language criteria ('HI' or 'EN').
    Args:
        dataset (str): Serialized dataset in JSON format.
    Returns:
        str: Serialized dataset in JSON format with records filtered to only
             include specified languages.
    """
    # Deserialize the dataset
    dataset = pl.DataFrame.deserialize(io.StringIO(dataset), format="json")

    # Initialize language detector
    languages = list(Language.all_spoken_ones())
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    # Perform language detection with multi-threading
    language_detected = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_sentence = {
            executor.submit(detector.detect_language_of, sentence): sentence
            for sentence in dataset["complaint"].to_list()
        }

        # Collect results as they complete
        for future in as_completed(future_to_sentence):
            language_detected.append(future.result())

    # Convert detected languages to ISO codes
    language_codes = [item.iso_code_639_1.name for item in language_detected]

    # Add language column to dataset and filter based on language criteria
    dataset = (
        dataset.with_columns(pl.Series(name="language", values=language_codes))
        .filter(pl.col("language").is_in(["HI", "EN"]))
        .drop(["language"])
    )

    # Serialize and return the filtered dataset
    return dataset.serialize(format="json")


def aggregate_filtered_task(dataset_a: str, dataset_b: str) -> None:
    """
    Aggregate two datasets by joining them on 'complaint_id' and selecting
    specific columns. The result is saved to a specified parquet file.
    Args:
        dataset_a (str): Serialized first dataset in JSON format.
        dataset_b (str): Serialized second dataset in JSON format.
    """
    output_path = os.path.join(
        os.path.dirname(__file__),
        "../../data/preprocessed_dataset.parquet",
    )

    # Deserialize datasets and perform an inner join on 'Complaint ID'
    dataset_a = pl.DataFrame.deserialize(io.StringIO(dataset_a), format="json")
    dataset_b = pl.DataFrame.deserialize(io.StringIO(dataset_b), format="json")

    # Join datasets and select specified columns
    selected_columns = [
        "complaint_id",
        "date_received",
        "date_resolved",
        "time_resolved_in_days",
        "complaint",
        "complaint_hindi",
        "product",
        "department",
        "sub_product",
        "issue",
        "sub_issue",
        "company",
        "state",
        "zipcode",
        "tags",
        "company_response_public",
        "company_response_consumer",
        "consumer_consent_provided",
        "submitted_via",
        "date_sent_to_company",
        "timely_response",
        "consumer_disputed",
    ]
    dataset_joined = dataset_a.join(dataset_b, on="complaint_id", how="inner").select(
        selected_columns
    )

    # Write the output to the specified parquet file
    dataset_joined.write_parquet(output_path)


def data_cleaning() -> str:
    """
    Clean the dataset by lowercasing complaint narratives, removing special characters, removing duplicates, and dropping records with null values in key columns.
    Returns:
        str: Serialized cleaned dataset in JSON serialized format.
    """
    # Define the data path and read the dataset
    data_path = os.path.join(
        os.path.dirname(__file__),
        "../../data/preprocessed_dataset.parquet",
    )
    dataset = pl.read_parquet(data_path)

    # Lowercase complaint narratives
    dataset = dataset.with_columns(pl.col("complaint").str.to_lowercase())

    # Remove special characters from 'complaint' column
    dataset = dataset.with_columns(
        pl.col("complaint").map_elements(
            lambda x: re.sub(r"[^A-Za-z0-9\s]", "", x), return_dtype=pl.Utf8
        )
    )

    # Remove duplicate records based on specific columns
    dataset = dataset.unique(
        subset=["product", "sub_product", "complaint"],
        maintain_order=True,
    )

    # Drop records with nulls in specified columns
    dataset = dataset.drop_nulls(
        subset=["product", "sub_product", "department", "complaint"]
    )

    # Serialize and return the cleaned dataset
    return dataset.serialize(format="json")


def remove_abusive_data(dataset: str, abuse_placeholder: str = "yyy") -> str:
    """
    Remove abusive words from 'complaint' column in the dataset,
    replacing them with a specified placeholder. The cleaned dataset is saved to a
    predefined output path.
    Args:
        dataset (str): Serialized dataset in JSON format.
        abuse_placeholder (str): Placeholder to replace abusive words.
    Returns:
        str: Serialized dataset with abusive words removed.
    """
    # Start the abusive data removal process
    logging.info("Starting abusive data filtering.")
    # Define paths for input and output
    output_path = os.path.join(
        os.path.dirname(__file__),
        "../../data/preprocessed_dataset.parquet",
    )
    abusive_words_path = os.path.join(
        os.path.dirname(__file__), "../../data/profanity_bank_dataset.parquet"
    )

    # Deserialize the dataset
    dataset = pl.DataFrame.deserialize(io.StringIO(dataset), format="json")

    # Set up Bloom Filter for abusive words
    profane_set = set()
    profanity_bloom = Bloom(200_000, 0.01)

    # Load abusive words
    abusive_words = pl.read_parquet(abusive_words_path)["profanity"].to_list()

    for word in abusive_words:
        profanity_bloom.add(word)
        profane_set.add(word)

    # Tokenize and clean complaints
    tokenized_complaints = dataset.with_columns(pl.col("complaint").str.split(" "))[
        "complaint"
    ].to_list()

    cleaned_records = []
    for record in tokenized_complaints:
        clean_record = [
            w if w not in profanity_bloom or w not in profane_set else abuse_placeholder
            for w in record
        ]
        cleaned_records.append(" ".join(clean_record))

    # Add the cleaned complaints to the dataset
    dataset = dataset.with_columns(
        pl.Series(name="abuse_free_complaints", values=cleaned_records)
    )

    logging.info("Abusive data filtering complete. Saving results to file.")
    # Save the processed dataset to output path
    dataset.write_parquet(output_path)

    # Return the serialized dataset
    return dataset.serialize(format="json")