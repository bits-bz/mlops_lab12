from fastapi import FastAPI, UploadFile, File
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel
import io

# Initialize the FastAPI app
app = FastAPI()

# Load the model from the local path
model_path = "gradient_boosting_model"
model = mlflow.sklearn.load_model(model_path)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    # Convert the file contents to a DataFrame
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    # Get predictions
    predictions = model.predict(df)
    # Return the results as JSON
    return {"predictions": predictions.tolist()}