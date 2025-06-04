# sem_train_model_cldai.py
# Enhanced SEM Training Model optimized for CPU compute

import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import logging
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Azure ML imports
try:
    from azureml.core import Run, Model, Workspace
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False
    print("Azure ML SDK not available. Running in standalone mode.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SEMDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None, validation_split=0.2, is_validation=False, random_seed=42):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Load data from CSV
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
                        self.idx_to_class[len(self.class_to_idx) - 1] = label
        
        # Split data for train/validation
        if validation_split > 0:
            np.random.seed(random_seed)
            indices = np.random.permutation(len(self.data))
            split_idx = int(len(self.data) * validation_split)
            
            if is_validation:
                self.data = [self.data[i] for i in indices[:split_idx]]
            else:
                self.data = [self.data[i] for i in indices[split_idx:]]
        
        logger.info(f"Dataset initialized with {len(self.data)} samples, {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, label = self.data[idx]
        img_path = os.path.join(self.data_dir, label, filename)

        if not os.path.exists(img_path):
            # Try alternative path structure
            alt_path = os.path.join(self.data_dir, filename)
            if os.path.exists(alt_path):
                img_path = alt_path
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")

        try:
            image = Image.open(img_path).convert('L')  # Grayscale
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise

        label_idx = self.class_to_idx[label]
        return image, torch.tensor(label_idx, dtype=torch.long)

class ModelTracker:
    def __init__(self, azure_run=None):
        self.azure_run = azure_run
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'epoch_time': []
        }
        self.start_time = None
        self.total_params = 0
        self.trainable_params = 0

    def count_parameters(self, model):
        self.total_params = sum(p.numel() for p in model.parameters())
        self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Log to Azure ML if available
        if self.azure_run:
            self.azure_run.log("total_parameters", self.total_params)
            self.azure_run.log("trainable_parameters", self.trainable_params)
            self.azure_run.log("model_size_mb", self.total_params * 4 / 1024**2)

    def log_epoch(self, epoch, train_loss, val_loss, val_acc, val_precision, val_recall, val_f1, epoch_time):
        self.training_history['epoch'].append(epoch)
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['val_accuracy'].append(val_acc)
        self.training_history['val_precision'].append(val_precision)
        self.training_history['val_recall'].append(val_recall)
        self.training_history['val_f1'].append(val_f1)
        self.training_history['epoch_time'].append(epoch_time)
        
        # Log to Azure ML if available
        if self.azure_run:
            self.azure_run.log("train_loss", train_loss)
            self.azure_run.log("val_loss", val_loss)
            self.azure_run.log("val_accuracy", val_acc)
            self.azure_run.log("val_precision", val_precision)
            self.azure_run.log("val_recall", val_recall)
            self.azure_run.log("val_f1", val_f1)
            self.azure_run.log("epoch_time", epoch_time)

    def save_metrics(self, filepath='training_metrics.json'):
        metrics_summary = {
            'model_info': {
                'total_parameters': self.total_params,
                'trainable_parameters': self.trainable_params,
                'total_training_time': sum(self.training_history['epoch_time']),
                'avg_epoch_time': np.mean(self.training_history['epoch_time']),
            },
            'training_history': self.training_history,
            'final_metrics': {
                'best_accuracy': max(self.training_history['val_accuracy']) if self.training_history['val_accuracy'] else 0,
                'best_f1': max(self.training_history['val_f1']) if self.training_history['val_f1'] else 0,
                'final_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else 0
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        return metrics_summary

def calculate_metrics(y_true, y_pred, num_classes):
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        return precision, recall, f1
    except Exception as e:
        logger.warning(f"Error calculating metrics: {e}")
        return 0.0, 0.0, 0.0

def create_transforms(img_size=128, augment=False):
    """Create data transforms - simplified for CPU"""
    if augment:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.3),  # Reduced probability
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    return transform

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs=10, tracker=None, early_stopping_patience=3):
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        train_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
            
            # Log batch progress for monitoring
            if batch_idx % 50 == 0:
                logger.info(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        epoch_loss = running_loss / train_samples
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        val_samples = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_samples += images.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / val_samples
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Calculate detailed metrics
        precision, recall, f1 = calculate_metrics(all_labels, all_preds, len(train_loader.dataset.class_to_idx))
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        if tracker:
            tracker.log_epoch(epoch + 1, epoch_loss, val_loss, val_acc, precision, recall, f1, epoch_time)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {epoch_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")
        print(f"  Val Precision: {precision:.4f}")
        print(f"  Val Recall: {recall:.4f}")
        print(f"  Val F1-Score: {f1:.4f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)
        
        # Early stopping check
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    return model, all_labels, all_preds

def save_model_artifacts(model, class_to_idx, output_dir, azure_run=None):
    """Save model and associated artifacts"""
    model_path = os.path.join(output_dir, 'sem_model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save class mapping
    class_mapping_path = os.path.join(output_dir, 'class_mapping.pkl')
    with open(class_mapping_path, 'wb') as f:
        pickle.dump(class_to_idx, f)
    
    # Create model configuration
    model_config = {
        'num_classes': len(class_to_idx),
        'input_size': 128,
        'architecture': 'resnet18',
        'class_to_idx': class_to_idx
    }
    
    config_path = os.path.join(output_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Register model in Azure ML if available and running in Azure ML context
    if azure_run and AZURE_ML_AVAILABLE:
        try:
            # Check if this is a real Azure ML run (not offline)
            if hasattr(azure_run, 'register_model') and not str(type(azure_run)).endswith('_OfflineRun'):
                model_azure = azure_run.register_model(
                    model_name='sem_classification_model',
                    model_path=output_dir,
                    description='SEM Image Classification Model (CPU trained)',
                    tags={'type': 'image_classification', 'framework': 'pytorch', 'compute': 'cpu'}
                )
                logger.info(f"Model registered in Azure ML: {model_azure.name}, Version: {model_azure.version}")
            else:
                logger.info("Running locally - model saved but not registered in Azure ML")
                logger.info("To register manually, use: az ml model register --name sem_classification_model --path ./outputs")
        except Exception as e:
            logger.warning(f"Could not register model in Azure ML: {e}")
            logger.info("Model saved locally - you can register manually later")
    
    return model_path, class_mapping_path, config_path

def main():
    parser = argparse.ArgumentParser(description='Train SEM Classification Model (CPU Optimized)')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with training images')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV file with image labels')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training (reduced for CPU)')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=128, help='Input image size')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save results')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--augment_data', action='store_true', help='Apply data augmentation')
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Azure ML run context
    azure_run = None
    if AZURE_ML_AVAILABLE:
        try:
            azure_run = Run.get_context()
            logger.info("Azure ML run context initialized")
            
            # Log hyperparameters
            azure_run.log("batch_size", args.batch_size)
            azure_run.log("learning_rate", args.lr)
            azure_run.log("num_epochs", args.num_epochs)
            azure_run.log("img_size", args.img_size)
            azure_run.log("validation_split", args.validation_split)
            azure_run.log("compute_type", "cpu")
            
        except Exception as e:
            logger.warning(f"Could not initialize Azure ML context: {e}")
    
    # Create transforms
    train_transform = create_transforms(args.img_size, args.augment_data)
    val_transform = create_transforms(args.img_size, False)
    
    # Load datasets
    try:
        csv_path = os.path.join(args.data_dir, args.csv_file)
        if not os.path.exists(csv_path):
            csv_path = args.csv_file  # Try direct path if relative doesn't work
        
        train_dataset = SEMDataset(
            csv_file=csv_path, 
            data_dir=args.data_dir, 
            transform=train_transform,
            validation_split=args.validation_split,
            is_validation=False
        )
        
        val_dataset = SEMDataset(
            csv_file=csv_path, 
            data_dir=args.data_dir, 
            transform=val_transform,
            validation_split=args.validation_split,
            is_validation=True
        )
        
        # Reduced num_workers for CPU
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=False)
        
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise

    # Initialize model
    device = torch.device("cpu")  # Force CPU
    logger.info(f"Using device: {device}")
    
    try:
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, len(train_dataset.class_to_idx))
        model = model.to(device)
        
        # Initialize tracker and count parameters
        tracker = ModelTracker(azure_run)
        tracker.count_parameters(model)
        
        logger.info(f"Model Parameters: Total: {tracker.total_params:,}, Trainable: {tracker.trainable_params:,}")
        logger.info(f"Model Size: {tracker.total_params * 4 / 1024**2:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise

    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    # Train model
    logger.info("Starting training...")
    start_time = time.time()
    
    try:
        model, final_labels, final_preds = train_model(
            model, criterion, optimizer, scheduler, train_loader, val_loader, 
            device, num_epochs=args.num_epochs, tracker=tracker, 
            early_stopping_patience=args.early_stopping_patience
        )
        
        total_time = time.time() - start_time
        logger.info(f"Total Training Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        
        if azure_run:
            azure_run.log("total_training_time", total_time)
            
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

    # Save all artifacts
    try:
        # Save model and artifacts
        model_path, class_mapping_path, config_path = save_model_artifacts(
            model, train_dataset.class_to_idx, args.output_dir, azure_run
        )
        logger.info(f"Model saved to: {model_path}")
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, 'training_metrics.json')
        metrics_summary = tracker.save_metrics(metrics_path)
        logger.info(f"Training metrics saved to: {metrics_path}")
        
        # Upload files to Azure ML if available
        if azure_run:
            try:
                azure_run.upload_file("training_metrics.json", metrics_path)
                logger.info("Files uploaded to Azure ML")
            except Exception as e:
                logger.error(f"Error uploading files to Azure ML: {e}")
        
        logger.info(f"All results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error saving artifacts: {e}")
        raise

    # Complete the run
    if azure_run:
        try:
            azure_run.complete()
            logger.info("Azure ML run completed successfully")
        except Exception as e:
            logger.error(f"Error completing Azure ML run: {e}")

if __name__ == "__main__":
    main()