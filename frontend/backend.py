import json
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the backend API URL from the environment variables
BACKEND_API_URL = os.getenv("BACKEND_API_URL")

def preprocess_text(complaint_text):
    """
    Preprocesses the complaint text by removing newlines and any other required cleaning.
    """
    # Remove newline characters
    complaint_text = complaint_text.replace('\n', ' ')
    return complaint_text

def fetch_backend_response(complaint_text):
    """
    Sends a POST request to the backend API with the complaint text.
    """

    # Preprocess the complaint_text
    cleaned_text = preprocess_text(complaint_text)
    response = requests.post(
        BACKEND_API_URL,
        json={"complaint_text": cleaned_text}
    )
    if response.status_code == 200:
        return json.loads(response.content)
    else:
        # Extract the error message from the backend response if available
        try:
            error_data = json.loads(response.content)
            return {"error": error_data.get("error_message", f"Status Code: {response.status_code}")}
        except json.JSONDecodeError:
            return {"error": f"Unexpected Error: Status Code {response.status_code}"}
        except Exception as e:
            return {"error": f"An error occurred while communicating with the backend: {str(e)}"}
