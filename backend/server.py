import os
import uvicorn
import numpy as np
import requests
from google.cloud import aiplatform, logging as gcloud_logging
from transformers import BertTokenizer
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from custom_exceptions import ValidationException, DriftException
from object_models import Complaint, PredictionResponse, ErrorResponse
from preprocessing import DataValidationPipeline, DataTransformationPipeline
from inference import make_inference

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

client = gcloud_logging.Client()
logger = client.logger("complaint-portal")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.log_struct({
        "severity": "INFO",
        "message": "Application startup"}
        )
    yield
    # Shutdown logic
    logger.log_struct({
        "severity": "INFO",
        "message": "Application shutdown"}
        )


validation_checks = {
    "min_words": 5,
    "max_words": 300,
    "allowed_languages": ["HI", "EN"],
}
validation_pipeline = DataValidationPipeline(validation_checks)
preprocessing_pipeline = DataTransformationPipeline()

# Vertex AI Project Config
PROJECT_ID = 'bilingualcomplaint-system'
LOCATION = 'us-east1'

# Endpoint Config
product_endpoint_id = '5930875671386521600'
department_endpoint_id = '1069239873640071168'

aiplatform.init(project=PROJECT_ID, location=LOCATION)
product_endpoint = aiplatform.Endpoint(product_endpoint_id)
department_endpoint = aiplatform.Endpoint(department_endpoint_id)

# Model Config
max_length = 128
hugging_face_model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(hugging_face_model_name)
 
# Product Config
product_labels = [
    'Credit reporting, credit repair services, or other personal consumer reports',
    'Debt collection',
    'Checking or savings account',
    'Credit card or prepaid card',
    'Mortgage',
    'Money transfer, virtual currency, or money service',
    'Vehicle loan or lease',
    'Student loan'
    ]

department_labels = [
    "Customer Relations and Compliance",
    "Loans and Credit",
    "Fraud and Security",
    "Account Services",
    "Payments and Transactions"
]

product_2_idx_map = {label: idx for idx, label in enumerate(product_labels)}
idx_2_product_map = {idx: label for label, idx in product_2_idx_map.items()}

department_2_idx_map = {label: idx for idx, label in enumerate(department_labels)}
idx_2_department_map = {idx: label for label, idx in department_2_idx_map.items()}

cloud_function_url = "https://us-east1-bilingualcomplaint-system.cloudfunctions.net/data_drift"

app = FastAPI(
    title="MLOps - Bilingual Complaint Classification System",
    description="Backend API Server to handle complaints from banking domain",
    version="0.1",
    lifespan=lifespan,
    debug=False
)


@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error_code=exc.error_code,
            error_message=exc.error_message,
        ).dict(),
    )

@app.exception_handler(DriftException)
async def drift_exception_handler(request: Request, exc: DriftException):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error_code=exc.error_code,
            error_message=exc.error_message,
        ).dict(),
    )


@app.get("/ping")
def ping():
    return {"service-name": "bilingual-complaints-system-api", "status": "active"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation Error"},
        400: {"model": ErrorResponse, "description": "Drift Detection"}
    },
)
async def submit_complaint(complaint: Complaint):
    logger.log_struct(
                     {
                         "severity": "INFO",
                         "message": "New Request Recieved",
                         "type": "REQUEST-RECIEVED",
                         "count": 1,
                    }
                    )
    try:
        is_valid = validation_pipeline.is_valid(text=complaint.complaint_text)
        if not is_valid:
            logger.log_struct(
                     {
                         "severity": "WARNING",
                         "message": "Input did not pass validation check",
                         "type": "VALIDATION-ERROR",
                         "count": 1,
                    }
                    )
            raise ValidationException(
                error_code=1001,
                error_message="Complaint Recieved failed validation checks",
            )
        complaint_language = validation_pipeline.get_recognised_language()
        processed_text = preprocessing_pipeline.process_text(
            text=complaint.complaint_text, language=complaint_language
        )

        # Drift Detection
        response = requests.post(
            cloud_function_url,
            json={"current_text": [processed_text]},  
            headers={"Content-Type": "application/json"}
            )

        if response.status_code == 200:
            result = response.json()
            drift_status = result.get('drift_detected')
            if drift_status:
                logger.log_struct(
                    {
                        "severity": "WARNING",
                        "message": "Drift Detected in the current text",
                        "type": "DRIFT-DETECTED",
                        "count": 1,
                    }
                )
                raise DriftException(
                    error_code=1002,
                    error_message="Drift Detected!"
                    )
        else:
            logger.log_struct(
                {
                        "severity": "ERROR",
                        "message": f"Drift Detection Failed\nError: {response.text}",
                        "type": "DRIFT-DETECTION-FAILED",
                        "count": 1,
                }
            )

        try:
            product_prediction = make_inference(text=processed_text, tokenizer=tokenizer, max_length=max_length, endpoint=product_endpoint)
            predicted_product = idx_2_product_map.get(np.argmax(product_prediction.predictions[0]))
            logger.log_struct(
                     {
                         "severity": "INFO",
                         "message": "Product Prediction Complete",
                         "type": "PRODUCT-PREDICTION-SUCCESS",
                         "count": 1,
                    }
                    )
        except Exception as e:
            logger.log_struct(
                     {
                         "severity": "ERROR",
                         "message": "Failed Predicting Product Class",
                         "type": f"PRODUCT-PREDICTION-ERROR\nException: {e}",
                         "count": 1,
                    }
                    )
        
        try:
            department_prediction = make_inference(text=processed_text, tokenizer=tokenizer, max_length=max_length, endpoint=department_endpoint)
            predicted_department = idx_2_department_map.get(np.argmax(department_prediction.predictions[0]))
            logger.log_struct(
                     {
                         "severity": "INFO",
                         "message": "Department Prediction Complete",
                         "type": "DEPARTMENT-PREDICTION-SUCCESS",
                         "count": 1,
                    }
                    )
        except Exception as e:
            logger.log_struct(
                     {
                         "severity": "ERROR",
                         "message": f"Failed Predicting Department Class\nException: {e}",
                         "type": "DEPARTMENT-PREDICTION-ERROR",
                         "count": 1,
                    }
                    )

        return PredictionResponse(
            product=predicted_product,
            department=predicted_department,
            processed_text=processed_text,
        )
    except DriftException as de:
        raise de
    except ValidationException as ve:
        raise ve
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred in prediction route."
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
