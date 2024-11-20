from typing import Optional
from pydantic import BaseModel


class Complaint(BaseModel):
    complaint_text: str

class PredictionResponse(BaseModel):
    product: str
    department: str
    processed_text: str

class ErrorResponse(BaseModel):
    error_code: int
    error_message: str
    details: Optional[str] = None