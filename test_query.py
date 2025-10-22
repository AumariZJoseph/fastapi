import requests
import json

url = "http://localhost:8000/api/v1/query"
data = {
    "user_id": "7df4c9f8-e66e-4135-8499-0985de85e8b8",
    "question": "What is in my documents?"
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response:", response.json())