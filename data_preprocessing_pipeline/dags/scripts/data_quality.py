import polars as pl
from datetime import datetime
import logging
import os
import io


# Defining a custom logger
def get_custom_logger():
    # Customer logs are stored in the below path
    log_path = os.path.join(
        os.path.dirname(__file__), "../../logs/application_logs/preprocessing_log.txt"
    )

    log_directory = os.path.dirname(log_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
        print(f"Directory '{log_directory}' has been created.")

    # Create the file if it doesn't exist
    if not os.path.exists(log_path):
        with open(log_path, 'w') as file:
            pass  # Create an log empty file
        print(f"File '{log_path}' has been created.")

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


logger = get_custom_logger()


def text_quality_checks(df):
    issues = []

    # Check for empty complaints
    empty_complaints = df.filter(pl.col("complaint").str.strip_chars() == "").height
    if empty_complaints > 0:
        issues.append(f"Found {empty_complaints} empty complaints")

    # Check for very short complaints
    short_complaints = df.filter(pl.col("complaint").str.len_chars() < 10).height
    if short_complaints > 0:
        issues.append(f"Found {short_complaints} complaints shorter than 10 characters")

    # Check for very long complaints
    long_complaints = df.filter(pl.col("complaint").str.len_chars() > 5000).height
    if long_complaints > 0:
        issues.append(f"Found {long_complaints} complaints longer than 5000 characters")

    # Check for non-ASCII characters in English complaints
    non_ascii = df.filter(pl.col("complaint").str.contains(r"[^\x00-\x7F]")).height
    if non_ascii > 0:
        issues.append(f"Found {non_ascii} complaints with non-ASCII characters")

    return issues


def validate_and_transform_dates(df):
    date_fields = ["date_sent_to_company"]

    # Convert columns to Date
    df = df.with_columns(
        [
            pl.col(field).str.strptime(pl.Date, format="%Y-%m-%d").cast(pl.Date)
            for field in date_fields
        ]
    )

    current_date = datetime.now().date()

    # Filter out future dates for 'date_received'
    future_received = df.filter(pl.col("date_received") > current_date)
    if future_received.height > 0:
        logger.info(
            f"Warning: {future_received.height} records have 'date_received' in the future."
        )
        df = df.filter(pl.col("date_received") <= current_date)

    # Ensure 'date_resolved' is not earlier than 'date_received'
    invalid_resolution = df.filter(pl.col("date_resolved") < pl.col("date_received"))
    if invalid_resolution.height > 0:
        logger.info(
            f"Warning: {invalid_resolution.height} records have 'date_resolved' earlier than 'date_received'."
        )
        df = df.filter(pl.col("date_resolved") >= pl.col("date_received"))

    return df


def validate_numeric_fields(df):
    logger.info("Entered")
    # Constraint for 'complaint_id': must be positive and unique
    df = df.with_columns(pl.col("complaint_id").cast(pl.Int64).alias("complaint_id"))
    logger.info("First step done")

    invalid_complaint_ids = df.filter(pl.col("complaint_id") <= 0)
    if invalid_complaint_ids.height > 0:
        logger.info(
            f"Warning: {invalid_complaint_ids.height} records have non-positive 'complaint_id'."
        )
        df = df.filter(pl.col("complaint_id") > 0)

    # Drop duplicates
    df = df.unique(subset=["complaint_id"])

    # Constraint for 'time_resolved_in_days': must be non-negative
    df = df.with_columns(
        pl.col("time_resolved_in_days").cast(pl.Int64).alias("time_resolved_in_days")
    )

    negative_resolution_times = df.filter(pl.col("time_resolved_in_days") < 0)
    if negative_resolution_times.height > 0:
        logger.info(
            f"Warning: {negative_resolution_times.height} records have negative 'time_resolved_in_days'."
        )
        df = df.with_columns(
            pl.when(pl.col("time_resolved_in_days") < 0)
            .then(0)
            .otherwise(pl.col("time_resolved_in_days"))
            .alias("time_resolved_in_days")
        )

    return df


def validate_text_fields(df):
    # Change 'complaint' and 'complaint_hindi' from BYTES to STRING type
    df = df.with_columns(
        pl.col("complaint").cast(pl.Utf8).alias("complaint"),
        pl.col("complaint_hindi").cast(pl.Utf8).alias("complaint_hindi"),
    )

    # Add length constraint for 'complaint' (maximum 5000 characters)
    long_complaints = df.filter(pl.col("complaint").str.len_chars() > 5000)
    if long_complaints.height > 0:
        logger.info(
            f"Warning: {long_complaints.height} complaints exceed 5000 characters."
        )
        df = df.with_columns(pl.col("complaint").str.slice(0, 5000).alias("complaint"))

    return df


def validate_issues_and_sub_issues(df, required_issues):
    df = df.with_columns(
        pl.col("issue").cast(pl.Utf8).alias("issue"),
        pl.col("sub_issue").cast(pl.Utf8).alias("sub_issue"),
    )

    missing_sub_issues = df.filter(
        (pl.col("issue").is_in(required_issues))
        & (
            pl.col("sub_issue").is_null()
            | (pl.col("sub_issue").str.strip_chars("") == "")
        )
    )

    if missing_sub_issues.height > 0:
        logger.info(
            f"Warning: {missing_sub_issues.height} records have missing 'sub_issue' for required 'issue':"
        )
        logger.info(missing_sub_issues[["issue", "sub_issue"]])

    return df


def validate_interdependent_constraints(df):
    df = df.with_columns(
        pl.col("issue").cast(pl.Utf8).alias("issue"),
        pl.col("sub_issue").cast(pl.Utf8).alias("sub_issue"),
    )

    invalid_responses = df.filter(
        (pl.col("consumer_disputed") == "Yes")
        & (pl.col("company_response_consumer") == "In progress")
    )

    if invalid_responses.height > 0:
        logger.info(
            f"Warning: {invalid_responses.height} records violate the constraint:"
        )
        logger.info(
            invalid_responses[["consumer_disputed", "company_response_consumer"]]
        )

    return df


def validate_date_constraints(df):

    invalid_dates = df.filter(pl.col("date_sent_to_company") < pl.col("date_received"))

    if invalid_dates.height > 0:
        logger.info(
            f"Warning: {invalid_dates.height} records have 'date_sent_to_company' earlier than 'date_received':"
        )

    return df


def validate_zipcode(df):
    df = df.with_columns(pl.col("zipcode").cast(pl.Utf8).alias("zipcode"))

    zip_pattern = r"^\d{5}(?:-\d{4})?$"
    invalid_zipcodes = df.filter(~pl.col("zipcode").str.contains(zip_pattern))

    if invalid_zipcodes.height > 0:
        logger.info(
            f"Warning: {invalid_zipcodes.height} records have invalid ZIP codes:"
        )

    return df


def validate_enum_values(df, feature_name, valid_values):
    if feature_name in df.columns:
        invalid_mask = ~df[feature_name].is_in(valid_values)
        df = df.with_columns(
            pl.when(invalid_mask)
            .then(None)
            .otherwise(pl.col(feature_name))
            .alias(feature_name)
        )
        invalid_entries = df.filter(invalid_mask)

        if invalid_entries.height > 0:
            logger.info(f"Invalid entries found in '{feature_name}':")
            logger.info(invalid_entries)
        else:
            logger.info(f"All values in '{feature_name}' are valid.")
    else:
        logger.info(f"Feature '{feature_name}' does not exist in the DataFrame.")

    return df


def validate_data_quality(dataset: str) -> str:

    logger.info("Data Quality Valdiation process started")

    # Define paths for input and output
    output_path = os.path.join(
        os.path.dirname(__file__),
        "../../data/preprocessed_dataset.parquet",
    )

    raw_dataset = pl.DataFrame.deserialize(io.StringIO(dataset), format="json")

    logger.info("Columns present : {}".format(raw_dataset.columns))

    # Set 'state' to NULL (None) where it is None
    raw_dataset = raw_dataset.with_columns(
        pl.when(pl.col("state").is_null())
        .then(pl.lit(None))
        .otherwise(pl.col("state"))
        .alias("state"),
        pl.when(pl.col("tags").is_null())
        .then(pl.lit("None"))
        .otherwise(pl.col("tags"))
        .alias("tags"),
    )
    raw_dataset = validate_numeric_fields(raw_dataset)
    raw_dataset = validate_and_transform_dates(raw_dataset)

    text_issues = text_quality_checks(raw_dataset)
    for issue in text_issues:
        logger.info(issue)
    raw_dataset = validate_text_fields(raw_dataset)
    raw_dataset = validate_date_constraints(raw_dataset)
    raw_dataset = validate_interdependent_constraints(raw_dataset)
    raw_dataset = validate_zipcode(raw_dataset)

    # Define valid enum values
    valid_timely_response = {"Yes", "No"}
    valid_consumer_disputed = {"Yes", "No", "N/A"}
    # Validate the 'timely_response' field
    raw_dataset = validate_enum_values(
        raw_dataset, "timely_response", valid_timely_response
    )
    # Validate the 'consumer_disputed' field
    raw_dataset = validate_enum_values(
        raw_dataset, "consumer_disputed", valid_consumer_disputed
    )
    # Define the required issues that need a sub_issue
    required_issues = ["Fraud or scam", "Closing an account"]
    # Validate and transform the issue and sub_issue fields
    validated_dataset = validate_issues_and_sub_issues(raw_dataset, required_issues)

    logger.info("Data Quality Validation has been successfully completed.")
    logger.info("Saving validated data to the file.")

    # Save the processed dataset to output path
    validated_dataset.write_parquet(output_path)
    return validated_dataset.serialize(format="json")
