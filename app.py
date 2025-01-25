import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI
from pydantic import BaseModel

# Step 1: Train a Proper Model
# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Save the trained model to a file
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Step 2: Serve the Model as an API
# Load the model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize the FastAPI app
app = FastAPI()

# Define the input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# API endpoint for health check
@app.get("/")
def root():
    return {"message": "Iris Model API is running!"}

# API endpoint for prediction
@app.post("/predict")
def predict(iris_data: IrisInput):
    # Convert input data to a 2D array
    input_array = np.array([[iris_data.sepal_length, iris_data.sepal_width, iris_data.petal_length, iris_data.petal_width]])
    prediction = model.predict(input_array)
    prediction_proba = model.predict_proba(input_array)

    # Map prediction to species name
    species = iris.target_names[prediction[0]]
    confidence = prediction_proba[0][prediction[0]]

    return {
        "prediction": species,
        "confidence": round(confidence, 2)
    }
