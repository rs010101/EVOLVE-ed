import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Enable CORS (Allow frontend to communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (update this for production security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load model with correct path
model_path = os.path.join(os.path.dirname(__file__), "adaptive_model.pkl")
try:
    model = joblib.load(model_path)
    logging.info("✅ Model loaded successfully!")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="❌ Model file not found at " + model_path)

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/focus/")
async def receive_focus_data(data: dict):
    logging.debug(f"📩 Received Focus Data: {data}")
    return {"message": "Focus data received"}

@app.post("/predict/")
def predict(student_data: dict):
    try:
        logging.debug(f"📩 Received Data: {student_data}")

        # Get the feature order used during training
        expected_features = model.feature_names_in_.tolist()  # Ensures correct feature order
        logging.debug(f"✅ Expected Features: {expected_features}")

        # Convert input data to DataFrame
        X_new = pd.DataFrame([student_data])

        # Ensure all expected features exist in input
        for feature in expected_features:
            if feature not in X_new.columns:
                raise HTTPException(status_code=400, detail=f"🚨 Missing feature in input: {feature}")
        
        # Reorder the columns to match the training order
        X_new = X_new[expected_features]

        logging.debug(f"📝 Processed DataFrame: \n{X_new}")

        # Make prediction
        prediction = model.predict(X_new)
        difficulty_map = {0: "Easy", 1: "Medium", 2: "Hard", 3: "Advanced"}

        return {"difficulty_level": difficulty_map.get(prediction[0], "Unknown")}

    except HTTPException as e:
        raise e

    except Exception as e:
        logging.error(f"⚠️ Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))