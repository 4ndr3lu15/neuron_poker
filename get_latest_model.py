import os

def get_latest_model(models_dir):
    # List all files in the directory
    files = os.listdir(models_dir)
    # Filter for files ending in ".pth"
    model_files = [f for f in files if f.endswith('.pth')]
    # Sort files numerically by the number before '.pth'
    model_files.sort(key=lambda x: int(x.split('.')[0]))
    # Return the latest model file
    return os.path.join(models_dir, model_files[-1]) if model_files else None

# Example usage
models_dir = "models"
latest_model = get_latest_model(models_dir)
if latest_model:
    print(f"The latest model is: {latest_model}")
else:
    print("No models found in the directory.")