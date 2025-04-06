import os
import glob
import time

def find_model_files():
    """
    Utility function to help locate model files in the project directory
    """
    print("Searching for model files...")
    
    # Define common places to look
    search_paths = [
        './vehicle_maintenance_model',
        './results',
        '.',
        './logs'
    ]
    
    found_files = []
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            print(f"\nChecking in {search_path}:")
            
            # If it's a directory, search recursively
            if os.path.isdir(search_path):
                # Look for various model files
                pattern_list = [
                    "**/*.bin",
                    "**/*.pt",
                    "**/*.pth",
                    "**/pytorch_model.bin",
                    "**/model.safetensors"
                ]
                
                for pattern in pattern_list:
                    files = glob.glob(os.path.join(search_path, pattern), recursive=True)
                    for file in files:
                        size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
                        modified = time.ctime(os.path.getmtime(file))
                        found_files.append((file, size, modified))
                        print(f"  - {file} ({size:.2f} MB, modified: {modified})")
            
            # If no files found in this path
            if not any(path.startswith(search_path) for path, _, _ in found_files):
                print("  No model files found in this location")
    
    if not found_files:
        print("\nNo model files found in any of the expected locations.")
        print("Please check if the model was saved correctly during training.")
    else:
        print(f"\nFound {len(found_files)} potential model file(s).")
        print("\nTo use a specific model file, you can modify the load_model function to point directly to it:")
        print("model.load_state_dict(torch.load('path/to/your/model/file'))")
    
    # Also check the outputs from the training process
    training_outputs = './results'
    if os.path.exists(training_outputs):
        print("\nTraining output directories found:")
        for item in os.listdir(training_outputs):
            item_path = os.path.join(training_outputs, item)
            if os.path.isdir(item_path):
                checkpoint_files = glob.glob(os.path.join(item_path, "**", "*.bin"), recursive=True)
                if checkpoint_files:
                    print(f"  - {item_path} contains {len(checkpoint_files)} checkpoint file(s)")
                else:
                    print(f"  - {item_path} (no checkpoint files)")

if __name__ == "__main__":
    find_model_files()
    print("\nIf you can't find your model files, you may need to re-run the training script.")
    print("After training completes, the model should be saved to './vehicle_maintenance_model' or './results'.")
