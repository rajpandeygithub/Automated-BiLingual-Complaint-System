# Automated-BiLingual-Complaint-System

## Project Overview
This project implements an automated system for handling customer complaints in the banking domain. Leveraging machine learning, the system analyzes complaints submitted in two languages (English & Hindi), and categorizes them using two distinct classifiers. The first classifier identifies the banking product related to the complaint (e.g., credit card, loan, account services), while the second routes the complaint to the appropriate department for efficient resolution. This solution streamlines complaint management, enhances response times, and significantly improves customer satisfaction by ensuring complaints are swiftly directed to the relevant teams.

Key features include:
- Simplified UI for complaint submission.
- Support for two languages (English, & Hindi).
- Automatic classification of products and departments using machine learning.
- Agent assignment based on language proficiency and availability.


## Data Information

We make use of the Consumer Complaint Database from Consumer Financial Protection Bureau (CFPB). This database compiles real-time U.S. consumer complaints regarding financial products and services. It includes data on consumer narratives (anonymized), banking products, and company responses, providing insight into financial sector issues.

For more information: [Link](https://www.consumerfinance.gov/data-research/consumer-complaints/) <br>
For API Access: [Link](https://cfpb.github.io/api/ccdb/api.html)

### Data Schema

| Attribute Name              | Description                                                                                   | Data Type |
|-----------------------------|-----------------------------------------------------------------------------------------------|-----------|
| `complaint_id`              | Unique identifier for each complaint                                                          | Integer     |
| `date_received`             | Date when CFPB received the complaint                                                          | Datetime      |
| `date_resolved`             | Date when the complaint was resolved by the bank                                                          | Integer      |
| `time_resolved_in_days`     | Duration in days taken to resolve the complaint by the bank                                               | Integer     |
| `complaint`                 | Consumer's answer to "what happened" from the complaint. Consumers must opt-in to share their narrative. The  narrative is not published unless the consumer consents, and consumers can opt-out at any time. The CFPB takes reasonable steps to scrub personal information from each complaint that could be used to identify the consumer.                                                    | String    |
| `complaint_hindi`           | Text content of the complaint (in Hindi)                                                      | String    |
| `product`                   | The type of product the consumer identified in the complaint                       | String    |
| `department`                | The department responsible for handling the complaint                                         | String    |
| `sub_product`               | The type of sub-product the consumer identified in the complaint                                       | String    |
| `issue`                     | The issue the consumer identified in the complain                                                            | String    |
| `sub_issue`                 | The sub-issue the consumer identified in the complaint                                                             | String    |
| `company`                   | Company associated with the complaint                                                         | String    |
| `state`                     | The state of the mailing address provided by the consumer                                                   | String    |
| `zipcode`                   | The mailing ZIP code provided by the consumer                                                    | String    |
| `tags`                      | Complaints are tagged based on submitter details: those involving consumers aged 62+ are tagged “Older American,” while complaints from servicemembers or their families are tagged “Servicemember.” This category includes active duty, National Guard, Reservists, Veterans, and retirees.               | String    |
| `company_response_public`   | The company's optional, public-facing response to a consumer's complaint. Companies can choose to select a response from a pre-set list of options that will be posted on the public database. For example, "Company believes complaint is the result of an isolated error."                                        | String    |
| `company_response_consumer` | This is how the company responded. For example, "Closed with explanation"                                                     | String    |
| `consumer_consent_provided` | Identifies whether the consumer opted in to publish their complaint narrative. The narrative is not published unless the consumer consents and consumers can opt-out at any time                                     | String    |
| `submitted_via`             | How the complaint was submitted to the CFPB                          | String    |
| `date_sent_to_company`      | The date when CFPB sent the complaint to the company                                 | String    |
| `timely_response`           | Whether the company gave a timely response                                 | String    |
| `consumer_disputed`         | Whether the consumer disputed the company’s response                                     | String    |


## Project Setup

### Clone the Repository:
```bash
git clone https://github.com/rajpandeygithub/Automated-BiLingual-Complaint-System.git
cd Automated-BiLingual-Complaint-System
docker compose up
```

**Note: Make sure you have docker installed, and provided enough resources to run**


## Data Preprocessing Pipeline

The preprocessing pipeline performs comprehensive data cleaning, filtering, and anonymization on the Consumer Complaint Database from Consumer Financial Protection Bureau (CFPB). The pipeline includes multiple stages to ensure data quality and privacy.

### Preprocessing Steps

### 1. Data Loading
- Loads the raw dataset from Google Cloud Storage
- Dataset format: Parquet

### 2. Basic Filtering
- Removes records with insufficient word count
- Filters records based on date range (March 19, 2015 to July 28, 2024)
- Converts date fields to proper date format

### 3. Language Detection
- Uses `fast_langdetect` (based on fast-text) for language identification
- Implements multi-threaded processing for improved performance
- Retains only English (EN) and Hindi (HI) complaints
- Removes records in other languages

### 4. Data Cleaning
- Converts complaint text to lowercase
- Removes special characters using regex
- Eliminates duplicate records based on:
  - Product
  - Sub-product
  - Complaint text
- Removes records with null values in critical fields:
  - Product
  - Sub-product
  - Department
  - Complaint text

### 5. Abusive Content Filtering
- Removes words from a set of abusive words
- Implements Bloom filter for abusive words set to efficient filter out abusive words
- Replaces abusive words with placeholder text
- Processes text while maintaining sentence structure

### 6. PII Data Anonymization
- Detects and masks personally identifiable information (PII)
- PII types handled:
  - Bank account numbers
  - Routing numbers
  - Credit card numbers (multiple formats)
  - Transaction dates
  - SSN/TIN numbers
  - EIN numbers
  - Passport numbers
  - Email addresses
  - Phone numbers
  - Home addresses
  - Demographic information (race, ethnicity, gender)
  - Transaction amounts

### 7. Data Aggregation
- Joins filtered datasets on complaint ID
- Selects and maintains relevant columns

### Output
- Final preprocessed dataset saved in Parquet format
- Location: `data/preprocessed_dataset.parquet`
- Includes comprehensive logging of all preprocessing steps
- Send a pipeline success / failure email

### Preprocessing Workflow
Summarizing entire Airflow Orchestration Graph Below:

```mermaid
graph TB
    subgraph "Data Preprocessing INIT DAG"
        A[Start] --> B[Trigger Data Validation]
    end

    subgraph "Data Validation Pipeline DAG"
        C[Load Data] --> D[Filter Records]
        D --> |Parallel Process 1| E[Word Count & Date Filter]
        D --> |Parallel Process 2| F[Language Detection]
        E --> G[Aggregate Results]
        F --> G
        G --> H[Trigger Data Cleaning]
    end

    subgraph "Data Cleaning Pipeline DAG"
        I[Data Cleaning] --> J[Anonymize Sensitive Data]
        J --> K[Remove Abusive Content]
        K --> L[Send Success Email]
    end

    B --> C
    H --> I
```
## License Information

The Consumer Financial Protection Bureau (CFPB) data is open to the public under the OPEN Government Data Act, enabling transparency and broad accessibility. Users should follow privacy guidelines, particularly around any personally identifiable information (PII) within consumer data.

For further details, refer to the [CFPB Data Policy](https://www.consumerfinance.gov/data/).