import pandas as pd
import io
from pandas import Int64Dtype, StringDtype
import logging
import os
import io
import polars as pl


# Defining a custom logger
def get_custom_logger():
    log_path = os.path.join(
        os.path.dirname(__file__), "../../logs/application_logs/preprocessing_log.txt"
    )
    custom_logger = logging.getLogger("preprocessing_logger")
    custom_logger.setLevel(logging.INFO)
    custom_logger.propagate = False

    if not custom_logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="a")
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        custom_logger.addHandler(file_handler)

    return custom_logger


logger = get_custom_logger()


def identify_complaint_outliers(dataframe, max_length=6000):
    # Calculate the length of each complaint
    dataframe = dataframe.with_columns(
        pl.col("complaint").str.len_chars().alias("complaint_length")
    )

    # Identify outliers
    outliers = dataframe.filter(pl.col("complaint_length") > max_length)

    # Log summary of outliers
    logging.info(f"Number of outliers: {outliers.shape[0]}")
    logging.info(
        f"Percentage of outliers: {outliers.shape[0] / dataframe.shape[0] * 100:.2f}%"
    )

    # Log some information about the outliers
    if outliers.shape[0] > 0:
        logging.info("\nSample of outliers:")
        logging.info(
            outliers.select(["complaint", "complaint_length"])
            .head()
            .to_pandas()
            .to_string()
        )

    # Optionally, return the outliers DataFrame
    return outliers


def consistency_checks(df):
    issues = []
    # Check if date_resolved is after date_received
    inconsistent_dates = df.filter(pl.col("date_resolved") < pl.col("date_received"))
    if inconsistent_dates.shape[0] > 0:
        issues.append(
            f"Found {inconsistent_dates.shape[0]} complaints where resolution date is before received date"
        )

    return issues


def generate_summary_report(df, schema, anomalies, consistency_issues):
    report = []
    report.append("# Data Quality Summary Report\n")

    report.append("## General Statistics")
    report.append(f"- Total number of complaints: {df.shape[0]:,}")
    report.append(f"- Number of features: {len(df.columns)}")
    report.append(
        f"- Date range: {df['date_received'].min().strftime('%Y-%m-%d')} to {df['date_received'].max().strftime('%Y-%m-%d')}"
    )

    report.append("\n## Data Quality Issues")
    report.append(f"- Anomalies type: {type(anomalies)}")

    if hasattr(anomalies, "anomaly_info"):
        anomaly_list = anomalies.anomaly_info
        report.append(f"- Number of anomalies detected: {len(anomaly_list)}")
        report.append("\n### Anomalies:")
        for anomaly in anomaly_list:
            if hasattr(anomaly, "feature_name") and hasattr(anomaly, "description"):
                report.append(f"  - {anomaly.feature_name}: {anomaly.description}")
            else:
                report.append(f"  - {anomaly}")
    else:
        report.append("- No anomaly info available")

    report.append(f"\n- Number of consistency issues: {len(consistency_issues)}")
    if consistency_issues:
        report.append("\n### Consistency Issues:")
        for issue in consistency_issues:
            report.append(f"  - {issue}")

    report.append("\n## Data Distribution")
    report.append("\n### Top 5 Products:")
    for product, percentage in (
        df["product"].value_counts(normalize=True).head().iter_rows()
    ):
        report.append(f"- {product}: {percentage:.2%}")

    report.append("\n### Top 5 Issues:")
    for issue, percentage in (
        df["issue"].value_counts(normalize=True).head().iter_rows()
    ):
        report.append(f"- {issue}: {percentage:.2%}")

    return "\n".join(report)


def schema_and_statistics_generation(dataset: str) -> str:
    logger.info("The schema and statistics generation process has started")

    # Define paths for input and output
    output_path = os.path.join(
        os.path.dirname(__file__),
        "../../data/preprocessed_dataset.parquet",
    )
    # Deserialize the dataset
    raw_dataset = pl.DataFrame.deserialize(io.StringIO(dataset), format="json")

    df = raw_dataset.with_columns(
        [
            pl.col("complaint_id").cast(pl.Int64),
            pl.col("date_received").cast(pl.Datetime, strict=False),
            pl.col("date_resolved").cast(pl.Datetime, strict=False),
            pl.col("time_resolved_in_days").cast(pl.Int16),
            pl.col("complaint").cast(pl.Utf8),
            pl.col("complaint_hindi").cast(pl.Utf8),
            pl.col("product").cast(pl.Utf8),
            pl.col("department").cast(pl.Utf8),
            pl.col("sub_product").cast(pl.Utf8),
            pl.col("issue").cast(pl.Utf8),
            pl.col("sub_issue").cast(pl.Utf8),
            pl.col("company").cast(pl.Utf8),
            pl.col("state").cast(pl.Utf8),
            pl.col("zipcode").cast(pl.Utf8),
            pl.col("company_response_consumer").cast(pl.Utf8),
            pl.col("consumer_consent_provided").cast(pl.Utf8),
            pl.col("submitted_via").cast(pl.Utf8),
            pl.col("date_sent_to_company").cast(pl.Datetime, strict=False),
            pl.col("timely_response").cast(pl.Utf8),
            pl.col("consumer_disputed").cast(pl.Utf8),
        ]
    )

    df = df.drop(["tags", "company_response_public"])
    df = df.with_columns(
        pl.when(pl.col("sub_issue").is_null())
        .then(pl.lit("NA"))
        .otherwise(pl.col("sub_issue"))
        .alias("sub_issue")
    )

    # Identify outliers
    identify_complaint_outliers(df)

    # If you want to remove the outliers from your DataFrame
    df = df.filter(pl.col("complaint").str.len_chars() <= 6000)

    for cat_feature in ["product", "department", "issue"]:
        distribution = df[cat_feature].value_counts(normalize=True)
        logger.info(f"\n{cat_feature.capitalize()} Distribution:")
        logger.info(distribution)
        # Check for class imbalance
        imbalance_threshold = 0.01  # 1%
        minority_classes = distribution.filter(
            pl.col("proportion") < imbalance_threshold
        )
        if minority_classes.shape[0] > 0:
            logging.warning(
                f"Found {minority_classes.shape[0]} minority {cat_feature}s with less than {imbalance_threshold:.1%} representation:"
            )
            logging.warning(f"{minority_classes}")

    yearly_stats = df.group_by(pl.col("date_received").dt.year().alias("year")).agg(
        [
            pl.count("complaint_id").alias("complaint_count"),
            pl.n_unique("product").alias("product_count"),
            pl.n_unique("issue").alias("issue_count"),
        ]
    )

    logging.info("Yearly Statistics:")
    logging.info(f"\n{yearly_stats}")

    # Check for significant changes in complaint volume or category count
    yearly_changes = yearly_stats.with_columns(
        [
            (pl.col("complaint_count") / pl.col("complaint_count").shift(1) - 1).alias(
                "complaint_change"
            ),
            (pl.col("product_count") / pl.col("product_count").shift(1) - 1).alias(
                "product_change"
            ),
            (pl.col("issue_count") / pl.col("issue_count").shift(1) - 1).alias(
                "issue_change"
            ),
        ]
    )
    significant_changes = yearly_changes.filter(
        (pl.col("complaint_change").abs() > 0.5)
        | (pl.col("product_change").abs() > 0.3)
        | (pl.col("issue_change").abs() > 0.3)
    )

    if significant_changes.shape[0] > 0:
        logging.warning("Significant changes detected in the following years:")
        logging.warning(f"{significant_changes}")

    duplicate_counts = df.select(
        [pl.col("complaint").value_counts().alias("complaint_counts")]
    )

    duplicate_counts = df.select(
        [pl.col("complaint").value_counts().alias("complaint_counts")]
    )

    # Count occurrences of each 'complaint' in the DataFrame
    duplicate_counts = df.select(
        pl.col("complaint").value_counts().alias("complaint_counts")
    )

    # Count occurrences of each 'complaint' in the DataFrame
    duplicate_counts = df.select(
        pl.col("complaint").value_counts().alias("complaint_counts")
    )

    # Extract the individual fields from the struct without exploding
    duplicate_counts = duplicate_counts.with_columns(
        [
            pl.col("complaint_counts").struct.field("complaint").alias("complaint"),
            pl.col("complaint_counts").struct.field("count").alias("complaint_count"),
        ]
    )

    # Filter for complaints with more than one occurrence
    duplicates = duplicate_counts.filter(pl.col("complaint_count") > 1)

    # Check if there are duplicates
    if duplicates.shape[0] > 0:
        logger.warning(f"Found {duplicates.shape[0]} duplicate complaints")

        # Join back with the original DataFrame to retrieve full duplicate rows
        duplicate_rows = df.join(
            duplicates.select("complaint"), on="complaint", how="inner"
        )
        logger.warning(f"Duplicate samples:\n{duplicate_rows.head()}")

    consistency_issues = consistency_checks(df)
    for issue in consistency_issues:
        print(issue)

    df.write_parquet(output_path)
    return df.serialize(format="json")
