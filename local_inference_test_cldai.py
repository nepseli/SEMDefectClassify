# local_inference_test.py
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import pickle
import os
import base64
from io import BytesIO
import argparse

class SEMInferenceService:
    def __init__(self, model_dir='./outputs'):
        self.model_dir = model_dir
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.transform = None
        self.device = torch.device('cpu')
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing components"""
        print("🔄 Loading model...")
        
        # Load model configuration
        config_path = os.path.join(self.model_dir, 'model_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load class mapping
        class_mapping_path = os.path.join(self.model_dir, 'class_mapping.pkl')
        with open(class_mapping_path, 'rb') as f:
            self.class_to_idx = pickle.load(f)
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Initialize model
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, config['num_classes'])
        
        # Load trained weights
        model_path = os.path.join(self.model_dir, 'sem_model.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((config['input_size'], config['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Classes: {list(self.class_to_idx.keys())}")
        print(f"🎯 Input size: {config['input_size']}x{config['input_size']}")
    
    def predict_image(self, image_path):
        """Predict on a single image file"""
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('L')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()
            
            predicted_class = self.idx_to_class[predicted_class_idx]
            
            return {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'all_probabilities': {
                    self.idx_to_class[i]: float(probabilities[0][i]) 
                    for i in range(len(self.idx_to_class))
                }
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def predict_base64(self, base64_string):
        """Predict on base64 encoded image"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data)).convert('L')
            
            # Apply transforms and predict
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()
            
            predicted_class = self.idx_to_class[predicted_class_idx]
            
            return {
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'all_probabilities': {
                    self.idx_to_class[i]: float(probabilities[0][i]) 
                    for i in range(len(self.idx_to_class))
                }
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def test_on_samples(self, num_samples=3):
        """Test the model on sample images from the dataset"""
        print("\n🧪 Testing model on sample images...")
        
        data_dir = os.path.join(self.model_dir, 'data')
        if not os.path.exists(data_dir):
            data_dir = './outputs/data'
        
        results = []
        
        for class_name in self.class_to_idx.keys():
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                images = [f for f in os.listdir(class_dir) if f.endswith('.png')][:num_samples]
                
                print(f"\n📁 Testing {class_name} samples:")
                for img_name in images:
                    img_path = os.path.join(class_dir, img_name)
                    result = self.predict_image(img_path)
                    
                    if 'error' not in result:
                        predicted = result['predicted_class']
                        confidence = result['confidence']
                        is_correct = predicted == class_name
                        status = "✅" if is_correct else "❌"
                        
                        print(f"  {status} {img_name}: {predicted} ({confidence:.3f})")
                        results.append({
                            'image': img_name,
                            'true_class': class_name,
                            'predicted_class': predicted,
                            'confidence': confidence,
                            'correct': is_correct
                        })
                    else:
                        print(f"  ❌ {img_name}: Error - {result['error']}")
        
        # Calculate accuracy
        if results:
            correct = sum(1 for r in results if r['correct'])
            total = len(results)
            accuracy = correct / total
            print(f"\n🎯 Sample Test Results: {correct}/{total} ({accuracy:.3f})")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Local SEM Model Inference Service')
    parser.add_argument('--model_dir', type=str, default='./outputs', help='Model directory')
    parser.add_argument('--test', action='store_true', help='Run tests on sample images')
    parser.add_argument('--image', type=str, help='Path to single image for prediction')
    
    args = parser.parse_args()
    
    # Initialize service
    service = SEMInferenceService(args.model_dir)
    
    if args.test:
        # Run sample tests
        service.test_on_samples()
    
    elif args.image:
        # Predict on single image
        result = service.predict_image(args.image)
        print(f"\n🔍 Prediction Result:")
        print(json.dumps(result, indent=2))
    
    else:
        # Interactive mode
        print("\n🎯 SEM Classification Service Ready!")
        print("Commands:")
        print("  test - Run tests on sample images")
        print("  predict <image_path> - Predict on single image")
        print("  quit - Exit")
        
        while True:
            try:
                command = input("\n> ").strip().split()
                if not command:
                    continue
                
                if command[0] == 'quit':
                    break
                elif command[0] == 'test':
                    service.test_on_samples()
                elif command[0] == 'predict' and len(command) > 1:
                    result = service.predict_image(command[1])
                    print(json.dumps(result, indent=2))
                else:
                    print("Unknown command. Use: test, predict <image_path>, or quit")
                    
            except KeyboardInterrupt:
                break
        
        print("👋 Service stopped.")

if __name__ == "__main__":
    main()