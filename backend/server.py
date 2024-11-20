import os
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from object_models import Complaint, PredictionResponse, ErrorResponse
from preprocessing import DataValidationPipeline, DataTransformationPipeline

validation_checks = {
    'min_words': 5,
    'max_words': 300,
    'allowed_languages': ["HI", "EN"]
}
validation_pipeline = DataValidationPipeline(validation_checks)
preprocessing_pipeline = DataTransformationPipeline()

app = FastAPI(
    title="MLOps - Bilingual Complaint Classification System",
    description="Backend API Server to handle complaints from banking domain",
    version="0.1",
)

class CustomException(HTTPException):
    def __init__(self, 
                 status_code: int,
                 error_code: int, error_message: str, 
                 details: Optional[str] = None):
        self.error_code = error_code
        self.error_message = error_message
        self.details = details
        super().__init__(status_code=status_code, detail=error_message)

@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    error_response = ErrorResponse(
        error_code=exc.error_code,
        error_message=exc.error_message,
        details=exc.details
    )
    return JSONResponse(
        status_code=exc.status_code, 
        content=error_response.dict()
        )

@app.get("/ping")
def ping():
    return {
        "service-name": "bilingual-complaints-system-api",
        "status": "active"
    }

@app.post("/predict")
async def submit_complaint(complaint: Complaint, response_model=PredictionResponse):

    try:
        validation_pipeline.is_valid(text=complaint.complaint_text)
    except Exception as e:
        raise CustomException(
            status_code=400,
            error_code=1001,
            error_message="Prediction failed due to invalid input",
            details=str(e)
        )
    
    processed_text = preprocessing_pipeline.process(text=complaint.complaint_text)
    predicted_product = "student loan" 
    predicted_departmet = "loan services"

    return PredictionResponse(
        product=predicted_product, 
        department=predicted_departmet,
        processed_text=processed_text
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)