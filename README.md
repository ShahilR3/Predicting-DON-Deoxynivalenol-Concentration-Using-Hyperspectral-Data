## **Project Title: Predicting DON (Deoxynivalenol) Concentration Using Hyperspectral Data**

This project involves a pipeline for predicting **Deoxynivalenol (DON) concentrations** in corn samples using a hyperspectral dataset, a deep learning model, and two distinct ways of interaction:
1. A **REST API** powered by FastAPI for real-time prediction.
2. A user-friendly **Streamlit-based UI** for entering reflectance values and obtaining predictions.


### **Table of Contents**
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Pipeline Overview](#pipeline-overview)
4. [Steps to Run the Project](#steps-to-run-the-project)
    - [Run the Machine Learning Code](#run-the-machine-learning-code)
    - [Run the REST API](#run-the-rest-api)
    - [Run the Streamlit UI](#run-the-streamlit-ui)
5. [Environment and Dependencies](#environment-and-dependencies)
6. [Key Metrics](#key-metrics)
7. [Conclusion](#conclusion)


### **Introduction**
The project uses hyperspectral reflectance data from corn samples to predict **Deoxynivalenol (DON)** levels, ensuring food safety in agriculture and animal feed. DON, or vomitoxin, is a harmful mycotoxin produced by fungi that affects grains like corn.

### **Features:**
- **Machine Learning Pipeline:** Implements a neural network with hyperparameter optimization via Optuna.
- **REST API:** Provides predictions through FastAPI endpoints.
- **Streamlit UI:** A graphical interface for easy input of spectral reflectance values.


### **Dataset Description**
- **Features:** Reflectance values across 448 spectral wavelengths.
- **Target Variable:** `vomitoxin_ppb` (DON concentration in parts per billion).
- **Format:** CSV file with the following columns:
  - `hsi_id`: Identifier for each corn sample.
  - 448 columns of reflectance values for each spectral wavelength.
  - `vomitoxin_ppb`: The DON concentration (target).


### **Pipeline Overview**

1. **Data Preprocessing:**
   - Handle missing and duplicate values.
   - Label encode categorical variables.
   - Standardize reflectance features using `StandardScaler`.

2. **Neural Network Architecture:**
   - Feedforward neural network with L2 regularization and dropout to prevent overfitting.
   - The model is trained to predict DON concentrations.

3. **Hyperparameter Optimization:**
   - Use **Optuna** to fine-tune model parameters like the number of units, learning rate, and dropout rates.

4. **Deployment:**
   - **REST API:** Serves predictions via an HTTP POST request.
   - **Streamlit UI:** Provides an interactive interface for manual entry of reflectance values.


### **Steps to Run the Project**

#### **Run the Machine Learning Code**
1. **Prepare the Dataset:**
   - Place the dataset file (`MLE-Assignment.csv`) in the appropriate location.

2. **Execute the ML Pipeline:**
   - Preprocess the data, train the neural network model, and optimize hyperparameters using the provided Python script.
   - The model and scaler will be saved:
     - Model: `optimized_don_model.h5`
     - Scaler: `scaler.pkl`

3. **Evaluate the Model:**
   - Key metrics (MAE, RMSE, and R²) will be printed.



#### **Run the REST API**
1. **Start the FastAPI Server:**
   - Navigate to the directory containing the API code.
   - Run the following command:
     ```bash
     uvicorn app_name:app --reload
     ```
     Replace `app_name` with the name of the Python file containing the FastAPI code.

2. **Test the API Endpoint:**
   - Use an API testing tool (e.g., Postman) or command-line tools (e.g., `curl`) to send requests.

3. **Sample Request:**
   - Send a JSON object with 448 reflectance values:
     ```json
     {
       "reflectance_values": [0.1, 0.2, 0.3, ..., 0.15]
     }
     ```

4. **Sample Response:**
   - The API will return the predicted DON concentration:
     ```json
     {
       "predicted_don_concentration": 3589.76
     }
     ```


#### **Run the Streamlit UI**
1. **Start the Streamlit Application:**
   - Navigate to the directory containing the Streamlit code.
   - Run the following command:
     ```bash
     streamlit run app_name.py
     ```
     Replace `app_name` with the name of the Python file containing the Streamlit code.

2. **Enter Reflectance Values:**
   - Input spectral reflectance values directly in the UI. The number of inputs required matches the dataset (e.g., 448 wavelengths).

3. **Get the Prediction:**
   - Click the **Predict DON Concentration** button to send the input data to the REST API.
   - The predicted DON concentration will be displayed.

### **Environment and Dependencies**
Ensure the following dependencies are installed:

- **Python Version:** 3.8+
- **Libraries:**
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `tensorflow`, `scikit-learn`
  - `optuna`, `fastapi`, `uvicorn`, `streamlit`, `requests`

Install all dependencies using:
```bash
pip install -r requirements.txt
```

### **Key Metrics**
After training and optimization, the model achieved the following performance metrics:
- **Mean Absolute Error (MAE):** e.g., `3397.9960149002077`
- **Root Mean Squared Error (RMSE):**  `9558.347529323804`
- **R² Score:**  `0.6731621291273264`

### **Conclusion**
This project demonstrates a complete solution for predicting DON concentrations using hyperspectral data. The pipeline provides flexibility for researchers and practitioners through a REST API and Streamlit-based user interface. The model ensures food safety by enabling efficient, non-invasive testing of mycotoxin levels in corn samples.

Feel free to contribute or extend this project further. If you have any questions or suggestions, let us know!