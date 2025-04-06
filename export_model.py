"""
This script creates a clean export of the trained model by:
1. Loading the model from wherever it was saved
2. Saving it in a standard format to a specified location
"""

import torch
import os
import glob
import pandas as pd
import sys

def find_model_file():
    """Find the latest model file in the expected locations"""
    # Check standard locations
    search_paths = [
        './vehicle_maintenance_model',
        './results',
    ]
    
    for path in search_paths:
        if not os.path.exists(path):
            continue
            
        # Check for pytorch_model.bin directly in the directory
        direct_model = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(direct_model):
            return direct_model
            
        # Check in checkpoint directories
        if path == './results':
            checkpoint_dirs = glob.glob(os.path.join(path, "checkpoint-*"))
            if checkpoint_dirs:
                latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
                checkpoint_model = os.path.join(latest_checkpoint, "pytorch_model.bin")
                if os.path.exists(checkpoint_model):
                    return checkpoint_model
    
    return None

def main():
    # Define the custom model structure (must match training)
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
    
    # Find the trained model file
    model_file = find_model_file()
    if not model_file:
        print("Error: Could not find any model file.")
        print("Please run the training script first or specify the model path.")
        sys.exit(1)
    
    print(f"Found model file: {model_file}")
    
    # Get input dimension from the data
    try:
        sample_data = pd.read_csv('processed_data.csv')
        input_dim = sample_data.drop('Need_Maintenance', axis=1).shape[1]
    except Exception as e:
        print(f"Error reading processed_data.csv: {e}")
        print("Using default input dimension of 20")
        input_dim = 20
    
    # Create a new model with the same architecture
    model = TabularModel(input_dim=input_dim)
    
    # Load the trained weights
    try:
        model.load_state_dict(torch.load(model_file, weights_only=True))
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Create the export directory if it doesn't exist
    export_dir = "./exported_model"
    os.makedirs(export_dir, exist_ok=True)
    
    # Save the model in a clean format
    export_path = os.path.join(export_dir, "vehicle_maintenance_model.pt")
    torch.save(model.state_dict(), export_path)
    
    print(f"Model successfully exported to {export_path}")
    print("\nTo use this model:")
    print("1. Load it with: model.load_state_dict(torch.load('exported_model/vehicle_maintenance_model.pt'))")
    print("2. Update both make_prediction.py and service.py to use this path")

if __name__ == "__main__":
    main()
