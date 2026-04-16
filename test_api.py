from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_api_is_running():
    response = client.get("/")
    assert response.status_code == 200

def test_ai_triage_critical():
    # We send a fake critical report to the API
    payload = {"text": "Severe chest pain, unresponsive, vitals dropping."}
    response = client.post("/api/dispatch", json=payload)
    
    # Assert the API works
    assert response.status_code == 200
    
    # Assert the AI is correctly classifying it as CRITICAL
    data = response.json()
    assert data["ai_triage_result"]["severity_class"] == "CRITICAL"