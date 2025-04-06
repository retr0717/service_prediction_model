from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

app = FastAPI(title="Vehicle Maintenance Prediction API")

# Load preprocessors and model
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Define the same model architecture as in training
class TabularModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, input_features=None, labels=None, **kwargs):
        logits = self.layers(input_features)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

def load_model(model_path: str) -> TabularModel:
    """
    Load the saved model for prediction
    """
    # Get the number of features from a sample of the training data
    sample_data = pd.read_csv('processed_data.csv')
    input_dim = sample_data.drop('Need_Maintenance', axis=1).shape[1]
    
    # Create model with the same architecture
    model = TabularModel(input_dim=input_dim)
    
    # Find the model file - search for it in model_path or its subdirectories
    model_file = None
    
    # First try the expected location
    expected_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(expected_path):
        model_file = expected_path
    else:
        # If not found, try to find in results directory
        results_dir = './results'
        print(f"Model not found at {expected_path}, searching in {results_dir}...")
        
        # Check if results directory exists
        if os.path.exists(results_dir):
            import glob
            # Look for pytorch_model.bin files in results or its subdirectories
            model_files = glob.glob(os.path.join(results_dir, "**", "pytorch_model.bin"), recursive=True)
            if model_files:
                # Use the most recently modified file
                model_file = max(model_files, key=os.path.getmtime)
                print(f"Found model file: {model_file}")
            else:
                # Look for checkpoint files if no pytorch_model.bin found
                checkpoint_dirs = glob.glob(os.path.join(results_dir, "checkpoint-*"))
                if checkpoint_dirs:
                    latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
                    checkpoint_model = os.path.join(latest_checkpoint, "pytorch_model.bin")
                    if os.path.exists(checkpoint_model):
                        model_file = checkpoint_model
                        print(f"Found checkpoint model: {model_file}")
    
    if model_file is None:
        print("No model file found. Running model finder tool to locate any model files...")
        # Run the model finder
        import subprocess
        subprocess.run(["python", "/home/r3tr0/projects/service-model/check_model_location.py"])
        raise FileNotFoundError(f"Could not find pytorch_model.bin in {model_path} or {results_dir}")
    
    # Load the saved weights
    print(f"Loading model from: {model_file}")
    # Use weights_only=True to safely load just the model weights
    model.load_state_dict(torch.load(model_file, weights_only=True))
    model.eval()  # Set model to evaluation mode
    
    return model

model = load_model('./vehicle_maintenance_model')

# Define the input model
class VehicleData(BaseModel):
    Vehicle_Model: str
    Mileage: float
    Maintenance_History: str
    Reported_Issues: int
    Vehicle_Age: float
    Fuel_Type: str
    Transmission_Type: str
    Engine_Size: float
    Odometer_Reading: float
    Last_Service_Date: str
    Warranty_Expiry_Date: str
    Owner_Type: str
    Insurance_Premium: float
    Service_History: int
    Accident_History: int
    Fuel_Efficiency: float
    Tire_Condition: str
    Brake_Condition: str
    Battery_Status: str

# Define the response model
class PredictionResponse(BaseModel):
    needs_maintenance: bool
    probability: float
    recommended_actions: List[str]
    next_service_date: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_maintenance(data: VehicleData):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Process dates
        current_date = datetime.now()
        input_df['Days_Since_Last_Service'] = (
            current_date - pd.to_datetime(input_df['Last_Service_Date'])
        ).dt.days
        input_df['Days_Until_Warranty_Expiry'] = (
            pd.to_datetime(input_df['Warranty_Expiry_Date']) - current_date
        ).dt.days
        
        # Drop original date columns
        input_df = input_df.drop(['Last_Service_Date', 'Warranty_Expiry_Date'], axis=1)
        
        # Define categorical and numerical features
        categorical_features = [
            'Vehicle_Model', 'Maintenance_History', 'Fuel_Type', 
            'Transmission_Type', 'Owner_Type', 'Tire_Condition', 
            'Brake_Condition', 'Battery_Status'
        ]
        
        numerical_features = [
            'Mileage', 'Reported_Issues', 'Vehicle_Age', 'Engine_Size',
            'Odometer_Reading', 'Insurance_Premium', 'Service_History',
            'Accident_History', 'Fuel_Efficiency', 'Days_Since_Last_Service',
            'Days_Until_Warranty_Expiry'
        ]
        
        # Encode categorical features
        encoded_cats = encoder.transform(input_df[categorical_features])
        encoded_df = pd.DataFrame(
            encoded_cats,
            columns=encoder.get_feature_names_out(categorical_features)
        )
        
        # Scale numerical features
        scaled_nums = scaler.transform(input_df[numerical_features])
        scaled_df = pd.DataFrame(
            scaled_nums,
            columns=numerical_features
        )
        
        # Combine processed features
        processed_input = pd.concat([scaled_df, encoded_df], axis=1)
        
        # Convert to tensor
        input_tensor = torch.tensor(processed_input.values, dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_features=input_tensor)
            logits = outputs["logits"]
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            maintenance_probability = probabilities[0, 1].item()
        
        # Generate recommended actions based on input features
        recommendations = []
        if predicted_class == 1:
            if data.Tire_Condition == "Worn Out":
                recommendations.append("Replace tires")
            if data.Brake_Condition == "Worn Out":
                recommendations.append("Service brake system")
            if data.Battery_Status == "Weak":
                recommendations.append("Replace battery")
            if data.Mileage > 10000 and data.Days_Since_Last_Service > 180:
                recommendations.append("Schedule oil change and general service")
        
        if not recommendations and predicted_class == 1:
            recommendations.append("Schedule general inspection based on prediction")
        
        # Calculate next service date (simplified)
        next_service_days = 30 if predicted_class == 1 else 180
        next_service = (current_date + pd.Timedelta(days=next_service_days)).strftime('%Y-%m-%d')
        
        return PredictionResponse(
            needs_maintenance=bool(predicted_class),
            probability=float(maintenance_probability),
            recommended_actions=recommendations,
            next_service_date=next_service
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)