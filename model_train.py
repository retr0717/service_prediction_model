import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import subprocess
import importlib.util
import os

# Check if accelerate is installed with required version
try:
    accelerate_spec = importlib.util.find_spec("accelerate")
    if accelerate_spec is None:
        print("Installing accelerate package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate>=0.26.0"])
    else:
        import accelerate
        accelerate_version = accelerate.__version__
        if not accelerate_version >= "0.26.0":
            print(f"Upgrading accelerate from {accelerate_version} to >=0.26.0")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate>=0.26.0", "--upgrade"])
except Exception as e:
    print(f"Error installing/checking accelerate: {e}")
    print("Please manually run: pip install 'accelerate>=0.26.0'")
    sys.exit(1)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load preprocessed data
processed_data = pd.read_csv('processed_data.csv')

# Define a custom dataset class for tabular data
class VehicleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_values': self.features[idx],
            'label': self.labels[idx]
        }

def train_huggingface_model(X_train, X_test, y_train, y_test):
    # Convert to HuggingFace Dataset format
    train_dataset = HFDataset.from_dict({
        'input_features': [row.tolist() for row in X_train.values],
        'labels': y_train.tolist()
    })
    
    eval_dataset = HFDataset.from_dict({
        'input_features': [row.tolist() for row in X_test.values],
        'labels': y_test.tolist()
    })
    
    # Define a simple custom model for tabular data
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
    
    # Define input dimension from training data
    input_dim = X_train.shape[1]
    
    # Initialize the model
    model = TabularModel(input_dim=input_dim)
    
    # Define metrics computation function
    def compute_metrics(pred):
        logits = pred.predictions
        labels = pred.label_ids
        preds = np.argmax(logits, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        do_eval=True,
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
    
    # Save the model
    trainer.save_model('./vehicle_maintenance_model')
    
    # Explicitly save the model weights in PyTorch format for easier loading
    output_dir = "./vehicle_maintenance_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save just the state dict for easier loading
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    print(f"Model weights explicitly saved to {os.path.join(output_dir, 'pytorch_model.bin')}")
    
    return model, trainer

# Example usage
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data.drop('Need_Maintenance', axis=1), 
        processed_data['Need_Maintenance'], 
        test_size=0.2, 
        random_state=42
    )
    model, trainer = train_huggingface_model(X_train, X_test, y_train, y_test)
    
    # Evaluate the model on the test set
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save the model parameters to a standard PyTorch file
    torch.save(model.state_dict(), "model_weights.pt")
    print("Additional model weights saved to model_weights.pt")