from locust import HttpUser, task, between
class APIUser(HttpUser):
    wait_time = between(1, 2)  
    host = "https://backend-api-server-661860051070.us-east1.run.app"  
    @task
    def test_predict_endpoint(self):
        # Request payload
        payload = {
            "complaint_text": "On XX/XX/XXXX, I requested removal of an entry from my consumer report placed there by JP Chase regarding a debt of a XXXX  Account Credit Card that appeared on my Personal Consumer Report via Certified mail. I made a second attempt to cure the inaccurate information from the Consumer Report on XX/XX/XXXX, via certified mail, no response. I made a third attempt on XX/XX/XXXX, via certified Mail to contact chase once they had closed, charged off without notice and sold the debt to a third party without my permission violating 15 USC 1681b Permissible purpose. On XX/XX/XXXX, we responded to Chase with a Demand for Validation and Proof of Claim, and we still have not received the disclosures. I sent the final notice for validation of debt on XX/XX/XXXX, via certified mail. However, as per the 30-day validation period mandated by 12 CFR 1006.34, JP Chase has failed to respond to any of my request for additional information and validation of the debt, which I sent on the dates listed in the above paragraph. The 30-day period ended 90 days ago, and I have continued to attempt to contact them via US mail . They have also taken payments from me on a closed charged off account which I have included proof off and they have not provided any communication or validation since. This lack of response and failure to validate the debt within the legal timeframe constitutes a violation of the FDCPA and Regulation F. As a result, I request that the CFPB investigate this matter and take appropriate action against JP Chase , including : Ordering JP Chase to cease all collection efforts on this alleged debt. "
        }
        # Send POST request to the /predict endpoint
        response = self.client.post("/predict", json=payload)
        
        if response.status_code != 200:
            print(f"Request failed: {response.status_code}, {response.text}")
        else:
            print(f"Response: {response.json()}")