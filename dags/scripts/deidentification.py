import polars as pl
import os
import re
import io
import logging

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


# Defining a custom logger
def get_custom_logger():
    # Customer logs are stored in the below path
    log_path = os.path.join(
        os.path.dirname(__file__), "../../logs/application_logs/preprocessing_log.txt"
    )
    custom_logger = logging.getLogger("preprocessing_logger")
    custom_logger.setLevel(logging.INFO)

    # Avoid default logs by setting propagate to False
    custom_logger.propagate = False

    # Set up a file handler for the custom logger
    if not custom_logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="a")
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        custom_logger.addHandler(file_handler)

    return custom_logger


def replace_pii_with_placeholders(text):

    for key, pattern in PATTERNS.items():
        if isinstance(pattern, list):  # For patterns that are lists (like credit cards)
            for p in pattern:
                text = re.sub(p, f"<{key}>", text, flags=re.IGNORECASE)
        else:
            text = re.sub(pattern, f"<{key}>", text, flags=re.IGNORECASE)
    return text


def anonymize_sensitive_data(dataset: str) -> str:

    logger = get_custom_logger()
    logger.info("Starting anonymization of sensitive data in complaints")

    # Define paths for input and output
    output_path = os.path.join(
        os.path.dirname(__file__),
        "../../data/preprocessed_dataset.parquet",
    )
    logger.info(
        f"Total number of records before anonymizing sensitive data: {len(dataset)}"
    )

    # Deserialize the dataset
    preprocessed_dataset = pl.DataFrame.deserialize(io.StringIO(dataset), format="json")

    anonymized_dataset = preprocessed_dataset.with_columns(
        pl.col("complaint").map_elements(replace_pii_with_placeholders)
    )
    logger.info(
        f"Total number of records after anonymizing sensitive data: {len(anonymized_dataset)}"
    )

    logger.info("Anonymization of sensitive data completed successfully.")
    logger.info("Saving anonymized data to file.")

    # Save the processed dataset to output path
    anonymized_dataset.write_parquet(output_path)
    return anonymized_dataset.serialize(format="json")
