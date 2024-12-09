import json
import random
from flask import Request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import requests
import logging
from google.cloud import bigquery
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import random

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Public URL of the reference embeddings file
PUBLIC_URL = "https://storage.googleapis.com/embed_pickle/ref_embeddings_hindi.pkl"

# Path to save the downloaded file
TMP_FILE_PATH = "/tmp/ref_embeddings_hindi.pkl"

# Load the Sentence-BERT model globally
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# # Email configuration
# SENDER_EMAIL = "sucessemailtrigger@gmail.com"
# PASSWORD = "jomnpxbfunwjgitb"
# RECEIVER_EMAILS = [
#     "hegde.anir@northeastern.edu",
#     "nenavath.r@northeastern.edu",
#     "pandey.raj@northeastern.edu",
#     "khatri.say@northeastern.edu",
#     "singh.arc@northeastern.edu",
#     "goparaju.v@northeastern.edu",
# ]

# Download reference embeddings from the public URL
def download_ref_embeddings(public_url: str, local_path: str):
    try:
        response = requests.get(public_url)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Downloaded embeddings from {public_url} to {local_path}.")
        else:
            raise Exception(f"Failed to download embeddings. HTTP Status Code: {response.status_code}")
    except Exception as e:
        raise Exception(f"Error downloading embeddings: {str(e)}")

# Load embeddings from local file
def load_embeddings(local_path: str):
    with open(local_path, "rb") as f:
        return pickle.load(f)

# Ensure reference embeddings are loaded during initialization
try:
    download_ref_embeddings(PUBLIC_URL, TMP_FILE_PATH)
    ref_embeddings = load_embeddings(TMP_FILE_PATH)
    logger.info("Reference embeddings loaded successfully.")
except Exception as e:
    logger.error(f"Error loading reference embeddings: {e}")
    ref_embeddings = None


# # Function to send email notifications
# def send_drift_email():
#     logger.info("Starting to send drift detection email to the team.")

#     subject = "Drift Detection Alert"
#     body = "Hi team,\nDrift has been detected in the model. Please check the system for details."

#     try:
#         server = smtplib.SMTP("smtp.gmail.com", 587)
#         server.starttls()
#         server.login(SENDER_EMAIL, PASSWORD)
#         logger.info("Logged into SMTP server successfully.")

#         for receiver_email in RECEIVER_EMAILS:
#             message = MIMEMultipart()
#             message["From"] = SENDER_EMAIL
#             message["To"] = receiver_email
#             message["Subject"] = subject
#             message.attach(MIMEText(body, "plain"))

#             server.sendmail(SENDER_EMAIL, receiver_email, message.as_string())
#             logger.info(f"Drift detection email sent successfully to {receiver_email}.")

#     except Exception as e:
#         logger.error(f"Error sending drift detection email: {e}")
#     finally:
#         server.quit()
#         logger.info("SMTP server connection closed.")


# Function to insert data into BigQuery with auto-incremented `entity_id`
def insert_into_bigquery(current_text, max_cosine_similarity):
    client = bigquery.Client()
    table_id = "bilingualcomplaint-system.MLOps.drift_records"  # Update with your BigQuery details

    # Get the current maximum entity_id
    query = f"""
    SELECT MAX(entity_id) AS max_entity_id
    FROM `{table_id}`
    """
    query_job = client.query(query)
    result = query_job.result()
    max_entity_id = 9641755
    for row in result:
        max_entity_id = row.max_entity_id

    if max_entity_id is None:
        max_entity_id = 9641755  # First entity_id will be 1

    # Increment the entity_id
    new_entity_id = max_entity_id + 1

    # Define random products and departments
    products = ['checking or savings account',
    'credit / debt management & repair services', 
    'credit / prepaid card services',
    'money transfers','mortgage',
    'payday loan','student loan','vehicle loan']

    departments = ['fraud and security',
    'customer relations and compliance',
    'payments and transactions',
    'loans and credit',
    'account services']

    # Randomly select product and department
    product = random.choice(products)
    department = random.choice(departments)

    # Create a row to insert with auto-incremented entity_id and random product/department
    rows_to_insert = [
        {
            "entity_id": new_entity_id,
            "abuse_free_complaint_hindi": current_text,
            "max_cosine_similarity": max_cosine_similarity,
            "product": product,
            "department": department
        }
    ]

    # Insert rows into BigQuery
    try:
        errors = client.insert_rows_json(table_id, rows_to_insert)
        if errors:
            logger.error(f"Error inserting rows: {errors}")
        else:
            logger.info("Data inserted successfully into BigQuery.")
    except Exception as e:
        logger.error(f"Error connecting to BigQuery: {e}")


# Compute similarity and insert data if drift is detected
def compute_similarity(request: Request):
    try:
        # Parse the incoming request JSON
        request_json = request.get_json(silent=True)
        if not request_json or "current_text" not in request_json:
            return jsonify({"error": "Invalid input. 'current_text' key is required."}), 400
        
        current_text = request_json["current_text"]
        if not isinstance(current_text, list):
            return jsonify({"error": "'current_text' must be a list of strings."}), 400

        # Check if embeddings are loaded
        if ref_embeddings is None:
            return jsonify({"error": "Reference embeddings not loaded."}), 500

        # Generate embeddings for the current text
        current_embeddings = model.encode(current_text, show_progress_bar=False)

        # Compute cosine similarity
        cos_similarities = cosine_similarity(current_embeddings, ref_embeddings)
        max_cos_sim_index = int(np.argmax(cos_similarities))  # Convert to Python int
        max_cos_sim = float(cos_similarities[0][max_cos_sim_index])  # Convert to Python float

        # Drift detection threshold
        COSINE_THRESHOLD = 0.70
        drift_detected = bool(max_cos_sim < COSINE_THRESHOLD)  # Convert to Python bool

        # Insert into BigQuery if drift is detected
        if drift_detected:
            insert_into_bigquery(current_text[0], max_cos_sim)  

        # Return the result
        result = {
            "max_cosine_similarity": max_cos_sim,
            "drift_detected": drift_detected,
            "most_similar_reference_index": max_cos_sim_index,
        }
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

