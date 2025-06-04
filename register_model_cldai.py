# register_model.py
from azureml.core import Workspace, Model
import os

def register_local_model():
    """Register the locally trained model in Azure ML"""
    
    # Connect to workspace
    ws = Workspace.from_config()
    print(f"Connected to workspace: {ws.name}")
    
    # Check if model files exist
    model_dir = './outputs'
    required_files = ['sem_model.pth', 'model_config.json', 'class_mapping.pkl']
    
    print("Checking for required model files...")
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            return None
    
    try:
        # Register the model
        model = Model.register(
            workspace=ws,
            model_name='sem_classification_model',
            model_path=model_dir,
            description='SEM Image Classification Model (Local Training)',
            tags={
                'type': 'image_classification', 
                'framework': 'pytorch', 
                'source': 'local',
                'accuracy': '89.33%'
            }
        )
        
        print(f"✅ Model registered successfully!")
        print(f"   Name: {model.name}")
        print(f"   Version: {model.version}")
        print(f"   ID: {model.id}")
        
        return model
        
    except Exception as e:
        print(f"❌ Registration failed: {e}")
        return None

if __name__ == "__main__":
    register_local_model()