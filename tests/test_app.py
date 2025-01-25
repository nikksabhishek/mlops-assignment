from iris_model_api import app
from fastapi.testclient import TestClient

# Correct initialization
client = fastapi.testclient.TestClient(app)

# Test 1: Health Check Endpoint
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Iris Model API is running!"}

# Test 2: Valid Prediction
def test_valid_prediction():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()
    assert response.json()["confidence"] >= 0.0
    assert response.json()["confidence"] <= 1.0

# Test 3: Invalid Input Data
def test_invalid_input():
    payload = {
        "sepal_length": "invalid",  # Invalid value
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

# Test 4: Missing Input Data
def test_missing_input():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5
        # Missing "petal_length" and "petal_width"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

# Test 5: Large Input Values
def test_large_values():
    payload = {
        "sepal_length": 1000.0,
        "sepal_width": 1000.0,
        "petal_length": 1000.0,
        "petal_width": 1000.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()

# Test 6: Small/Negative Input Values
def test_negative_values():
    payload = {
        "sepal_length": -5.0,
        "sepal_width": -3.5,
        "petal_length": -1.4,
        "petal_width": -0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()

# Test 7: Boundary Values
def test_boundary_values():
    payload = {
        "sepal_length": 0.0,
        "sepal_width": 0.0,
        "petal_length": 0.0,
        "petal_width": 0.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()

# Test 8: Additional Features in Payload
def test_additional_features():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        "extra_feature": 42  # Extra feature not expected
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
