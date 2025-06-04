# evaluate_sem_model_cldai.py
# Comprehensive model evaluation script optimized for CPU

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import argparse
import logging
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    from azureml.core import Run
    azure_run = Run.get_context()
except:
    azure_run = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SEMDataset:
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.class_to_idx = {}
        
        with open(csv_file, 'r') as file:
            next(file)  # Skip header
            for line in file:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    filename = parts[0]
                    label = parts[1]
                    self.data.append((filename, label))
                    if label not in self.class_to_idx:
                        self.class_to_idx[label] = len(self.class_to_idx)
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, label = self.data[idx]
        img_path = os.path.join(self.data_dir, label, filename)
        
        try:
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(self.class_to_idx[label], dtype=torch.long)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            dummy_image = torch.zeros(1, 128, 128)
            return dummy_image, torch.tensor(0, dtype=torch.long)

def load_model_and_config(model_dir):
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    mapping_path = os.path.join(model_dir, 'class_mapping.pkl')
    with open(mapping_path, 'rb') as f:
        class_to_idx = pickle.load(f)
    
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
    
    model_path = os.path.join(model_dir, 'sem_model.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, class_to_idx, config

def comprehensive_evaluation(model, test_loader, class_names, device, output_dir):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)
    
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_eval.png'), dpi=300)
    plt.close()
    
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(np.array(all_preds)[class_mask] == np.array(all_labels)[class_mask])
            class_accuracies[class_name] = float(class_acc)
    
    eval_summary = {
        'overall_accuracy': float(np.mean(np.array(all_preds) == np.array(all_labels))),
        'per_class_accuracy': class_accuracies,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'total_samples': len(all_labels)
    }
    
    eval_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_summary, f, indent=2)
    
    return eval_summary

def main():
    parser = argparse.ArgumentParser(description='Comprehensive SEM Model Evaluation')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory with trained model')
    parser.add_argument('--output_dir', type=str, default='./evaluation_outputs', help='Output directory')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory (optional)')
    parser.add_argument('--csv_file', type=str, default=None, help='CSV file path (optional)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting comprehensive model evaluation...")
    
    try:
        model, class_to_idx, config = load_model_and_config(args.model_dir)
        class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]
        
        device = torch.device("cpu")
        model = model.to(device)
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        # Try different data locations
        data_locations = []
        if args.data_dir and args.csv_file:
            data_locations.append((args.data_dir, args.csv_file))
        
        # Default locations to try
        data_locations.extend([
            (os.path.join(args.model_dir, 'data'), os.path.join(args.model_dir, 'data', 'labels.csv')),
            (os.path.join(args.model_dir, '..', 'data'), os.path.join(args.model_dir, '..', 'data', 'labels.csv')),
            ('./outputs/data', './outputs/data/labels.csv'),
            ('./data', './data/labels.csv')
        ])
        
        test_dataset = None
        for data_dir, csv_file in data_locations:
            if os.path.exists(csv_file) and os.path.exists(data_dir):
                logger.info(f"Found data at: {data_dir}")
                try:
                    test_dataset = SEMDataset(csv_file, data_dir, transform)
                    break
                except Exception as e:
                    logger.warning(f"Failed to load dataset from {csv_file}: {e}")
                    continue
        
        if test_dataset is None:
            logger.error("Could not find test data. Searched locations:")
            for data_dir, csv_file in data_locations:
                logger.error(f"  - {csv_file} (exists: {os.path.exists(csv_file)})")
            return
        
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)
        
        # Perform evaluation
        eval_results = comprehensive_evaluation(model, test_loader, class_names, device, args.output_dir)
        
        logger.info("Evaluation completed!")
        logger.info(f"Overall accuracy: {eval_results['overall_accuracy']:.4f}")
        logger.info("Per-class accuracy:")
        for class_name, acc in eval_results['per_class_accuracy'].items():
            logger.info(f"  {class_name}: {acc:.4f}")
        
        # Log to Azure ML
        if azure_run:
            azure_run.log("eval_overall_accuracy", eval_results['overall_accuracy'])
            for class_name, acc in eval_results['per_class_accuracy'].items():
                azure_run.log(f"eval_accuracy_{class_name}", acc)
            
            try:
                azure_run.upload_file("confusion_matrix_eval.png", os.path.join(args.output_dir, 'confusion_matrix_eval.png'))
                azure_run.upload_file("evaluation_results.json", os.path.join(args.output_dir, 'evaluation_results.json'))
            except Exception as e:
                logger.warning(f"Could not upload files to Azure ML: {e}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
