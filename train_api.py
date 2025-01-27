import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Set our tracking server URI for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Set the experiment name; this creates it if it doesn't exist
experiment_name = "MLOps_Lab13"
mlflow.set_experiment(experiment_name)

# Load the Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split the dataset into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start a new MLflow run; this will track the entire experiment's steps
with mlflow.start_run() as run:
    # Instantiate the GradientBoostingRegressor with specified hyperparameters
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    # Train the model on the training dataset
    model.fit(X_train, y_train)
    # Log the trained model to MLflow
    mlflow.sklearn.log_model(model, "gradient_boosting_model")
    # Save the model locally
    model_path = "gradient_boosting_model"
    mlflow.sklearn.save_model(model, model_path)
    # Calculate the model mean squared error using the test dataset
    mse = mean_squared_error(y_test, model.predict(X_test))
    # Log the mean squared error as a metric in MLflow
    mlflow.log_metric("mse", mse)
    # Retrieve and print the run ID for future reference (e.g., loading or deployment)
    run_id = run.info.run_id
    print("Run ID:", run_id)  # Print the run ID for use in later tasks
    print("Model saved at:", model_path)  # Print the local model path