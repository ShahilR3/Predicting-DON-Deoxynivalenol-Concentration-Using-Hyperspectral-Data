from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
from keras.metrics import MeanSquaredError  # type: ignore # Standard metric
import logging

# Set up basic logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

app = FastAPI()

# Load trained model
try:
    model = tf.keras.models.load_model("optimized_don_model.h5", compile=False)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")

# Load the pre-fitted scaler
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded successfully!")
except Exception as e:
    logger.error(f"Error loading scaler: {e}")

# Define input schema using Pydantic
class ReflectanceValues(BaseModel):
    reflectance_values: list

@app.post("/predict/")
def predict(data: ReflectanceValues):
    try:
        logger.debug(f"Received data: {data.reflectance_values}")

        # Convert input to NumPy array and reshape for prediction
        X_new = np.array(data.reflectance_values).reshape(1, -1)
        logger.debug(f"Reshaped input for prediction: {X_new}")

        # Apply scaling
        X_scaled = scaler.transform(X_new)
        logger.debug(f"Scaled input: {X_scaled}")

        # Recompile model if necessary
        try:
            model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])
        except Exception as e:
            logger.error(f"Error recompiling the model: {e}")
            raise HTTPException(status_code=500, detail="Model recompilation failed")

        # Make prediction
        try:
            prediction = model.predict(X_scaled)[0][0]
            logger.debug(f"Prediction result: {prediction}")

            # Convert the prediction to a native Python float before returning
            prediction = float(prediction)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed")

        return {"predicted_don_concentration": prediction}
    
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing input: {e}")