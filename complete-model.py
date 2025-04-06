# file: main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification
from typing import List, Dict, Any, Optional
from datetime import datetime

app = FastAPI(title="Vehicle Maintenance Prediction API")

# Load preprocessors and model
scaler = joblib.load('model_output/scaler.pkl')
encoder = joblib.load('model_output/encoder.pkl')
model = AutoModelForSequenceClassification.from_pretrained('./model_output/model')
model.eval()  # Set model to evaluation mode

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
            outputs = model(input_tensor)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            maintenance_probability = probabilities[0][1].item()
        
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

# file: train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import Trainer, TrainingArguments
from datasets import Dataset as HFDataset
import os
import json
from datetime import datetime

def main():
    print("Starting vehicle maintenance prediction model training...")
    
    # Create output directory
    os.makedirs('model_output', exist_ok=True)
    
    # Load and preprocess data
    df = load_sample_data(5000)  # Increase samples for better training
    processed_data, scaler, encoder, categorical_features, numerical_features = preprocess_data(df)
    
    # Save preprocessors
    joblib.dump(scaler, 'model_output/scaler.pkl')
    joblib.dump(encoder, 'model_output/encoder.pkl')
    
    # Save feature lists
    with open('model_output/feature_info.json', 'w') as f:
        json.dump({
            'categorical_features': categorical_features,
            'numerical_features': numerical_features
        }, f, indent=2)
    
    # Split data
    X = processed_data.drop('Need_Maintenance', axis=1)
    y = processed_data['Need_Maintenance']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Class distribution in training: {y_train.value_counts().to_dict()}")
    
    # Train model
    model, trainer = train_model(X_train, X_test, y_train, y_test)
    
    # Manually evaluate on test set
    test_dataset = HFDataset.from_dict({
        'features': [row.tolist() for row in X_test.values],
        'labels': y_test.tolist()
    })
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    # Try a sample prediction
    print("\nTesting prediction with a sample:")
    predict_sample(model, scaler, encoder, categorical_features, numerical_features)
    
    print("\nTraining completed successfully!")

# 1. Data loading (synthetic data for demonstration)
def load_sample_data(n_samples=1000):
    """Create synthetic data for demonstration purposes"""
    np.random.seed(42)
    
    vehicle_models = ['Car', 'SUV', 'Van', 'Truck', 'Bus', 'Motorcycle']
    maintenance_history = ['Good', 'Average', 'Poor']
    fuel_types = ['Diesel', 'Petrol', 'Electric']
    transmission_types = ['Automatic', 'Manual']
    owner_types = ['First', 'Second', 'Third']
    conditions = ['New', 'Good', 'Worn Out']
    battery_status = ['New', 'Good', 'Weak']
    
    data = {
        'Vehicle_Model': np.random.choice(vehicle_models, n_samples),
        'Mileage': np.random.uniform(1000, 100000, n_samples),
        'Maintenance_History': np.random.choice(maintenance_history, n_samples),
        'Reported_Issues': np.random.randint(0, 10, n_samples),
        'Vehicle_Age': np.random.uniform(0, 15, n_samples),
        'Fuel_Type': np.random.choice(fuel_types, n_samples),
        'Transmission_Type': np.random.choice(transmission_types, n_samples),
        'Engine_Size': np.random.uniform(800, 5000, n_samples),
        'Odometer_Reading': np.random.uniform(1000, 150000, n_samples),
        'Last_Service_Date': [(datetime.now() - pd.Timedelta(days=np.random.randint(1, 730))).strftime('%Y-%m-%d') for _ in range(n_samples)],
        'Warranty_Expiry_Date': [(datetime.now() + pd.Timedelta(days=np.random.randint(-100, 1000))).strftime('%Y-%m-%d') for _ in range(n_samples)],
        'Owner_Type': np.random.choice(owner_types, n_samples),
        'Insurance_Premium': np.random.uniform(500, 5000, n_samples),
        'Service_History': np.random.randint(0, 20, n_samples),
        'Accident_History': np.random.randint(0, 5, n_samples),
        'Fuel_Efficiency': np.random.uniform(5, 30, n_samples),
        'Tire_Condition': np.random.choice(conditions, n_samples),
        'Brake_Condition': np.random.choice(conditions, n_samples),
        'Battery_Status': np.random.choice(battery_status, n_samples),
    }
    
    # Create target based on features (simple rule-based logic for demo)
    df = pd.DataFrame(data)
    df['Need_Maintenance'] = ((df['Vehicle_Age'] > 7) | 
                             (df['Mileage'] > 50000) | 
                             (df['Reported_Issues'] > 5) |
                             (df['Maintenance_History'] == 'Poor') |
                             (df['Tire_Condition'] == 'Worn Out') |
                             (df['Brake_Condition'] == 'Worn Out') |
                             (df['Battery_Status'] == 'Weak')).astype(int)
    
    print(f"Generated {n_samples} synthetic vehicle records")
    print(f"Maintenance needed in {df['Need_Maintenance'].sum()} cases ({df['Need_Maintenance'].mean()*100:.1f}%)")
    
    return df

# 2. Data preprocessing
def preprocess_data(df):
    print("Preprocessing data...")
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Convert date strings to days since values
    current_date = datetime.now()
    data['Days_Since_Last_Service'] = data['Last_Service_Date'].apply(
        lambda x: (current_date - datetime.strptime(x, '%Y-%m-%d')).days)
    data['Days_Until_Warranty_Expiry'] = data['Warranty_Expiry_Date'].apply(
        lambda x: (datetime.strptime(x, '%Y-%m-%d') - current_date).days)
    
    # Drop original date columns
    data = data.drop(['Last_Service_Date', 'Warranty_Expiry_Date'], axis=1)
    
    # Create categorical features list
    categorical_features = [
        'Vehicle_Model', 'Maintenance_History', 'Fuel_Type', 
        'Transmission_Type', 'Owner_Type', 'Tire_Condition', 
        'Brake_Condition', 'Battery_Status'
    ]
    
    # Create numerical features list
    numerical_features = [
        'Mileage', 'Reported_Issues', 'Vehicle_Age', 'Engine_Size',
        'Odometer_Reading', 'Insurance_Premium', 'Service_History',
        'Accident_History', 'Fuel_Efficiency', 'Days_Since_Last_Service',
        'Days_Until_Warranty_Expiry'
    ]
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(data[categorical_features])
    
    # Create DataFrame with encoded categories
    encoded_df = pd.DataFrame(
        encoded_cats,
        columns=encoder.get_feature_names_out(categorical_features)
    )
    
    # Standardize numerical features
    scaler = StandardScaler()
    scaled_nums = scaler.fit_transform(data[numerical_features])
    
    # Create DataFrame with scaled numerical features
    scaled_df = pd.DataFrame(
        scaled_nums,
        columns=numerical_features
    )
    
    # Combine processed features
    processed_data = pd.concat([scaled_df, encoded_df], axis=1)
    
    # Add target variable
    processed_data['Need_Maintenance'] = data['Need_Maintenance']
    
    print(f"Preprocessed data shape: {processed_data.shape}")
    return processed_data, scaler, encoder, categorical_features, numerical_features

# 3. Training function
def train_model(X_train, X_test, y_train, y_test):
    print("Training model...")
    # Convert to HuggingFace Dataset format
    train_dataset = HFDataset.from_dict({
        'features': [row.tolist() for row in X_train.values],
        'labels': y_train.tolist()
    })
    
    eval_dataset = HFDataset.from_dict({
        'features': [row.tolist() for row in X_test.values],
        'labels': y_test.tolist()
    })
    
    # Define model - create a custom config
    input_dim = X_train.shape[1]
    config = AutoConfig.from_pretrained(
        'bert-base-uncased',
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        num_labels=2,  # Binary classification (0 or 1)
    )
    
    # Initialize the model
    model = AutoModelForSequenceClassification.from_config(config)
    
    # Redefine input layer to match tabular data size
    model.bert.embeddings.word_embeddings = torch.nn.Linear(input_dim, 128)
    
    # Define metrics computation function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1_micro,
            'precision': precision_micro,
            'recall': recall_micro
        }
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./model_output/checkpoints',
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir='./model_output/logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",  # Disable reporting to wandb, etc.
    )
    
    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate the model
    eval_result = trainer.evaluate()
    print(f"Evaluation results: {eval_result}")
    
    # Save evaluation results
    with open('./model_output/eval_results.json', 'w') as f:
        json.dump(eval_result, f, indent=4)
    
    # Save the model
    trainer.save_model('./model_output/model')
    
    return model, trainer

# 4. Prediction function
def predict_sample(model, scaler, encoder, categorical_features, numerical_features):
    """Test prediction with a sample input"""
    # Create a sample input
    sample = {
        'Vehicle_Model': 'SUV',
        'Mileage': 75000,
        'Maintenance_History': 'Poor',
        'Reported_Issues': 4,
        'Vehicle_Age': 6.5,
        'Fuel_Type': 'Petrol',
        'Transmission_Type': 'Automatic',
        'Engine_Size': 2500,
        'Odometer_Reading': 75000,
        'Last_Service_Date': (datetime.now() - pd.Timedelta(days=200)).strftime('%Y-%m-%d'),
        'Warranty_Expiry_Date': (datetime.now() + pd.Timedelta(days=100)).strftime('%Y-%m-%d'),
        'Owner_Type': 'First',
        'Insurance_Premium': 2000,
        'Service_History': 5,
        'Accident_History': 1,
        'Fuel_Efficiency': 12.5,
        'Tire_Condition': 'Worn Out',
        'Brake_Condition': 'Good',
        'Battery_Status': 'Good'
    }
    
    # Convert to DataFrame
    sample_df = pd.DataFrame([sample])
    
    # Process dates
    current_date = datetime.now()
    sample_df['Days_Since_Last_Service'] = (
        current_date - pd.to_datetime(sample_df['Last_Service_Date'])
    ).dt.days
    sample_df['Days_Until_Warranty_Expiry'] = (
        pd.to_datetime(sample_df['Warranty_Expiry_Date']) - current_date
    ).dt.days
    
    # Drop original date columns
    sample_df = sample_df.drop(['Last_Service_Date', 'Warranty_Expiry_Date'], axis=1)
    
    # Encode categorical features
    encoded_cats = encoder.transform(sample_df[categorical_features])
    encoded_df = pd.DataFrame(
        encoded_cats,
        columns=encoder.get_feature_names_out(categorical_features)
    )
    
    # Scale numerical features
    scaled_nums = scaler.transform(sample_df[numerical_features])
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
        outputs = model(input_tensor)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        maintenance_probability = probabilities[0][1].item()
    
    print(f"Prediction: {'Needs maintenance' if predicted_class == 1 else 'No maintenance needed'}")
    print(f"Confidence: {maintenance_probability:.2f}")
    
    # Generate maintenance recommendations
    recommendations = []
    if predicted_class == 1:
        if sample['Tire_Condition'] == "Worn Out":
            recommendations.append("Replace tires")
        if sample['Brake_Condition'] == "Worn Out":
            recommendations.append("Service brake system")
        if sample['Battery_Status'] == "Weak":
            recommendations.append("Replace battery")
        if sample['Mileage'] > 10000 and sample_df['Days_Since_Last_Service'].values[0] > 180:
            recommendations.append("Schedule oil change and general service")
    
    if not recommendations and predicted_class == 1:
        recommendations.append("Schedule general inspection based on prediction")
    
    if recommendations:
        print("Recommended actions:")
        for rec in recommendations:
            print(f"- {rec}")

if __name__ == "__main__":
    main()

# file: convex_integration.ts
import { v } from "convex/values";
import { query, mutation } from "./_generated/server";
import axios from "axios";

// Environment variable for the prediction service URL
const PREDICTION_SERVICE_URL = process.env.PREDICTION_SERVICE_URL || "http://localhost:8000";

// Function to get vehicle data in the format expected by the prediction service
export const getPredictionData = query({
  args: { vehicleId: v.id("vehicles") },
  handler: async (ctx, args) => {
    // Get the vehicle
    const vehicle = await ctx.db.get(args.vehicleId);
    if (!vehicle) {
      throw new Error("Vehicle not found");
    }
    
    // Get vehicle history
    const vehicleHistory = await ctx.db
      .query("vehicleHistory")
      .withIndex("by_vehicle", (q) => q.eq("vehicleId", args.vehicleId))
      .first();
    
    if (!vehicleHistory) {
      throw new Error("Vehicle history not found");
    }
    
    // Get maintenance records for service history
    const maintenanceRecords = await ctx.db
      .query("maintenanceRecords")
      .withIndex("by_vehicle", (q) => q.eq("vehicleId", args.vehicleId))
      .collect();
    
    // Calculate days since last service
    const lastServiceRecord = maintenanceRecords
      .sort((a, b) => new Date(b.completedDate).getTime() - new Date(a.completedDate).getTime())
      .shift();
    
    const lastServiceDate = lastServiceRecord ? lastServiceRecord.completedDate : new Date().toISOString();
    
    // Create prediction data
    return {
      Vehicle_Model: vehicle.name,
      Mileage: vehicle.metrics.odometer || 0,
      Maintenance_History: maintenanceRecords.length > 3 ? "Good" : maintenanceRecords.length > 1 ? "Average" : "Poor",
      Reported_Issues: vehicle.alerts.sensorFailures.length + 
        (vehicle.alerts.checkEngine ? 1 : 0) + 
        (vehicle.alerts.highTemp ? 1 : 0) + 
        (vehicle.alerts.brakeSystem ? 1 : 0) + 
        (vehicle.alerts.tirePressure ? 1 : 0) + 
        (vehicle.alerts.lowFuel ? 1 : 0),
      Vehicle_Age: new Date().getFullYear() - parseInt(vehicle.system.firmwareVersion.substring(0, 4), 10),
      Fuel_Type: vehicle.vehicleType === "EV" ? "Electric" : "Petrol",
      Transmission_Type: "Automatic", // Assuming all vehicles are automatic
      Engine_Size: vehicle.vehicleType === "EV" ? 0 : 2000, // Default value
      Odometer_Reading: vehicle.metrics.odometer,
      Last_Service_Date: lastServiceDate.substring(0, 10),
      Warranty_Expiry_Date: new Date(new Date().setFullYear(new Date().getFullYear() + 2)).toISOString().substring(0, 10),
      Owner_Type: "First", // Default value
      Insurance_Premium: 1000, // Default value
      Service_History: maintenanceRecords.length,
      Accident_History: 0, // Default value
      Fuel_Efficiency: vehicle.vehicleType === "EV" ? 25 : vehicle.metrics.fuelLevel > 0 ? vehicle.metrics.odometer / vehicle.metrics.fuelLevel : 10,
      Tire_Condition: "Good", // Default value
      Brake_Condition: vehicle.alerts.brakeSystem ? "Worn Out" : "Good",
      Battery_Status: vehicle.alerts.batteryHealth === "GOOD" ? "Good" : 
                      vehicle.alerts.batteryHealth === "LOW" ? "Weak" : "Degrading",
    };
  },
});

// Function to predict maintenance for a vehicle
export const predictMaintenance = mutation({
  args: { vehicleId: v.id("vehicles") },
  handler: async (ctx, args) => {
    // Get prediction data from our helper function
    const predictionData = await ctx.runQuery(getPredictionData, { vehicleId: args.vehicleId });
    
    try {
      // Call the prediction service
      const response = await axios.post(`${PREDICTION_SERVICE_URL}/predict`, predictionData);
      
      // Store the prediction results
      const result = {
        vehicleId: args.vehicleId,
        needs_maintenance: response.data.needs_maintenance,
        probability: response.data.probability,
        recommended_actions: response.data.recommended_actions,
        next_service_date: response.data.next_service_date,
        predicted_at: new Date().toISOString(),
      };
      
      // You could store this in a new table if needed
      // await ctx.db.insert("maintenancePredictions", result);
      
      // If maintenance is needed, schedule it automatically
      if (response.data.needs_maintenance) {
        await ctx.db.insert("maintenanceSchedules", {
          vehicleId: args.vehicleId,
          taskName: "Predicted Maintenance",
          taskType: "GENERAL_INSPECTION",
          intervalDays: 30,
          nextDueDate: response.data.next_service_date,
          notes: `Automated prediction: ${response.data.recommended_actions.join(", ")}`,
          notificationDays: 7,
          active: true,
        });
      }
      
      return result;
    } catch (error) {
      console.error("Error calling prediction service:", error);
      throw new Error("Failed to get maintenance prediction");
    }
  },
});

# file: Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# file: requirements.txt
fastapi==0.104.1
uvicorn==0.23.2
pandas==2.1.0
numpy==1.26.0
scikit-learn==1.3.1
transformers==4.33.3
torch==2.0.1
pydantic==2.4.2
joblib==1.3.2
datasets==2.14.5

# file: docker-compose.yml
version: '3'

services:
  maintenance-prediction:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./model_output:/app/model_output
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped

# file: README.md
# Vehicle Maintenance Prediction Service

This project provides a machine learning service that predicts when vehicles need maintenance based on their metrics and history.

## Features

- Predicts maintenance needs based on vehicle data
- Provides confidence scores and specific maintenance recommendations
- Serves predictions via a REST API
- Integrates with Convex backend for your fleet management app

## Setup Instructions

### Training the Model

1. First, make sure you have the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the training script:
   ```
   python train_model.py
   ```

This will:
- Generate synthetic training data (replace with your real data)
- Preprocess the data and train the model
- Save the model and preprocessing tools in the `model_output` directory
- Print evaluation metrics

### Running the Prediction Service

Option 1: Run with Python directly
```
uvicorn main:app --host 0.0.0.0 --port 8000
```

Option 2: Run with Docker
```
docker-compose up
```

### API Usage

The service exposes an endpoint at `/predict` that accepts POST requests with vehicle data.

Example request:
```json
{
  "Vehicle_Model": "SUV",
  "Mileage": 75000,
  "Maintenance_History": "Poor",
  "Reported_Issues": 4,
  "Vehicle_Age": 6.5,
  "Fuel_Type": "Petrol",
  "Transmission_Type": "Automatic",
  "Engine_Size": 2500,
  "Odometer_Reading": 75000,
  "Last_Service_Date": "2023-10-10",
  "Warranty_Expiry_Date": "2025-10-10",
  "Owner_Type": "First",
  "Insurance_Premium": 2000,
  "Service_History": 5,
  "Accident_History": 1,
  "Fuel_Efficiency": 12.5,
  "Tire_Condition": "Worn Out",
  "Brake_Condition": "Good",
  "Battery_Status": "Good"
}
```

Example response:
```json
{
  "needs_maintenance": true,
  "probability": 0.89,
  "recommended_actions": ["Replace tires", "Schedule oil change and general service"],
  "next_service_date": "2025-05-05"
}
```

## Convex Integration

To use this prediction service with your Convex backend:

1. Add the provided `convex_integration.ts` file to your Convex project
2. Set the `PREDICTION_SERVICE_URL` environment variable to point to your deployed service
3. Call the `predictMaintenance` mutation with a vehicle ID to get predictions

## Production Considerations

For production deployment:
- Deploy the service on a reliable host with enough resources
- Consider using a managed Kubernetes service or similar for scaling
- Set up proper