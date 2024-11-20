import os
import re
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from preprocessing import PreprocessingPieline

class Complaint(BaseModel):
    complaint_text: str

class PredictionResponse(BaseModel):
    product: str
    department: str
    processed_text: str

preprocessing_pipeline = PreprocessingPieline()

app = FastAPI(
    title="MLOps - Bilingual Complaint Classification System",
    description="Backend API Server to handle complaints from banking domain",
    version="0.1",
)

@app.get("/ping")
def ping():
    return {
        "service-name": "bilingual-complaints-system-api",
        "status": "active"
    }

@app.post("/predict")
async def submit_complaint(complaint: Complaint, response_model=PredictionResponse):
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