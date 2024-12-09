from google.cloud import bigquery, logging as gcloud_logging


logger_client = gcloud_logging.Client()
logger = logger_client.logger("bigquery-logger")

def insert_to_prediction_table(
        bqClient: bigquery.Client,
        processed_complaint: str,
        language: str,
        product: str,
        department: str,
        table_id: str
        ) -> bool:
    max_entity_query = f"""
    SELECT COALESCE(MAX(entity_id), 9641755) + 1 AS next_entity_id
    FROM {table_id}
    """
    query_job = bqClient.query(max_entity_query)
    result = query_job.result()
    next_entity_id = list(result)[0]["next_entity_id"]

    if language == 'EN':
        new_record = {
            "entity_id": next_entity_id,
            "abuse_free_complaints": processed_complaint,
            "abuse_free_complaint_hindi": None,  
            "product": product,
            "department": department,
        }
    elif language == 'HI':
        new_record = {
            "entity_id": next_entity_id,
            "abuse_free_complaints": None,  
            "abuse_free_complaint_hindi": processed_complaint,
            "product": product,
            "department": department,
        }
    
    errors = bqClient.insert_rows_json(
        table_id, [new_record]
        )  

    if errors:
        logger.log_struct(
             {
                 "severity": "ERROR",
                 "message": f"{errors}",
                 "type": "BIGQUERY-PREDICTION-INSERT-ERROR",
                 "count": 1,
            })
        return False
    else:
        logger.log_struct(
             {
                 "severity": "INFO",
                 "message": f"Entry ID: {next_entity_id} | Prediction Successfully Logged",
                 "type": "BIGQUERY-PREDICTION-INSERT-ERROR",
                 "count": 1,
            })
        return True