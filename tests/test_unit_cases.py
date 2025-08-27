# tests/test_unit_cases.py

import pytest
from fastapi.testclient import TestClient
from api.main import app   # or your FastAPI entrypoint

client = TestClient(app)

#Verfication of the UI page with "Document Portal" text from HTML page.
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Portal" in response.text