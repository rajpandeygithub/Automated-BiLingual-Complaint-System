import os
import sys
import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from custom_exceptions import ValidationException
from object_models import Complaint, PredictionResponse, ErrorResponse
from preprocessing import DataValidationPipeline, DataTransformationPipeline

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format=log_format,  # Set the log format
    handlers=[
        logging.StreamHandler(sys.stdout),  # Output logs to stdout
        logging.FileHandler("app.log"),  # Output logs to a file
    ],
)

logger = logging.getLogger("my_fastapi_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Application startup")
    yield
    # Shutdown logic
    logger.info("Application shutdown")


validation_checks = {
    "min_words": 5,
    "max_words": 300,
    "allowed_languages": ["HI", "EN"],
}
validation_pipeline = DataValidationPipeline(validation_checks)
preprocessing_pipeline = DataTransformationPipeline()

app = FastAPI(
    title="MLOps - Bilingual Complaint Classification System",
    description="Backend API Server to handle complaints from banking domain",
    version="0.1",
    lifespan=lifespan,
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


@app.get("/ping")
def ping():
    return {"service-name": "bilingual-complaints-system-api", "status": "active"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation Error"},
    },
)
async def submit_complaint(complaint: Complaint):
    logger.info("Prediction Service Accessed")
    try:
        is_valid = validation_pipeline.is_valid(text=complaint.complaint_text)
        if not is_valid:
            logger.info("Complaint Recieved failed validation checks")
            raise ValidationException(
                error_code=1001,
                error_message="Complaint Recieved failed validation checks",
            )
        complaint_language = validation_pipeline.get_recognised_language()
        processed_text = preprocessing_pipeline.process_text(
            text=complaint.complaint_text, language=complaint_language
        )
        predicted_product = "student loan"
        predicted_departmet = "loan services"

        return PredictionResponse(
            product=predicted_product,
            department=predicted_departmet,
            processed_text=processed_text,
        )
    except ValidationException as ve:
        raise ve
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred in prediction route."
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
