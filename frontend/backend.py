import json
import requests

def fetch_backend_response(complaint_text):
    """
    Sends a POST request to the backend API with the complaint text.
    """
    # Uncomment the following lines to connect to the actual backend API
    """
    response = requests.post(
        'https://backend-api-server-661860051070.us-east1.run.app/predict',
        json={"complaint_text": complaint_text}
    )
    if response.status_code == 200:
        return json.loads(response.content)
    else:
        return {"error": f"Status Code: {response.status_code}"}
    """
    # Hardcoded response for now
    return {
        "agent": "Agent1",
        "product_department": "Department XYZ",
        "product": "Product PQR"
    }
