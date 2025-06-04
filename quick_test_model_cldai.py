# quick_test_model.py
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import pickle
import os
import numpy as np

def load_model(model_dir):
    """Load the trained model"""
    # Load model configuration
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load class mapping
    class_mapping_path = os.path.join(model_dir, 'class_mapping.pkl')
    with open(class_mapping_path, 'rb') as f:
        class_to_idx = pickle.load(f)
    
    # Create model
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
    
    # Load weights
    model_path = os.path.join(model_dir, 'sem_model.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, class_to_idx, config

def test_model_on_samples():
    """Test the model on some sample images"""
    model_dir = './outputs'
    data_dir = './outputs/data'
    
    print("🔍 Loading trained model...")
    model, class_to_idx, config = load_model(model_dir)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Classes: {list(class_to_idx.keys())}")
    print(f"🎯 Model input size: {config['input_size']}")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((config['input_size'], config['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Test on sample images from each class
    correct_predictions = 0
    total_predictions = 0
    
    for class_name in class_to_idx.keys():
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            # Get first 3 images from this class
            images = [f for f in os.listdir(class_dir) if f.endswith('.png')][:3]
            
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                
                # Load and preprocess image
                image = Image.open(img_path).convert('L')
                image_tensor = transform(image).unsqueeze(0)
                
                # Make prediction
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    predicted_class = idx_to_class[predicted.item()]
                    confidence = probabilities[0][predicted.item()].item()
                
                # Check if correct
                is_correct = predicted_class == class_name
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                status = "✅" if is_correct else "❌"
                print(f"{status} {img_name}: True={class_name}, Predicted={predicted_class}, Confidence={confidence:.3f}")
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\n🎯 Quick Test Results:")
    print(f"   Correct: {correct_predictions}/{total_predictions}")
    print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    return accuracy

if __name__ == "__main__":
    try:
        test_model_on_samples()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n📁 Checking file structure...")
        for root, dirs, files in os.walk('./outputs'):
            level = root.replace('./outputs', '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")