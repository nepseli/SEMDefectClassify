# azure_ml_complete_pipeline_cldai.py
# Complete Azure ML CLI Pipeline for SEM Classification - CPU Optimized

import os
import json
import argparse
import logging
import time
from pathlib import Path
import yaml

# Azure ML imports
from azureml.core import (
    Workspace, Experiment, Environment, ScriptRunConfig, 
    Run, Model, Dataset
)
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice
from azureml.exceptions import WebserviceException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SEMAzureMLPipeline:
    def __init__(self, workspace_config_path="config.json"):
        """Initialize the Azure ML pipeline"""
        try:
            self.ws = Workspace.from_config(workspace_config_path)
            logger.info(f"Connected to workspace: {self.ws.name}")
        except Exception as e:
            logger.error(f"Failed to connect to workspace: {e}")
            raise
        
        self.datastore = self.ws.get_default_datastore()
        self.experiment_name = "sem-classification-complete-pipeline"
        self.model_name = "sem-classification-model"
        self.service_name = "sem-classification-service"
        
    def _create_run_config(self, compute_target, environment):
        """Create RunConfiguration with proper syntax for current SDK version"""
        run_config = RunConfiguration()
        run_config.target = compute_target
        run_config.environment = environment
        return run_config
    
    def create_or_get_compute_target(self, compute_name="cpu-cluster", vm_size="Standard_DS3_v2", max_nodes=2):
        """Create or get existing compute target with CPU fallback"""
        try:
            compute_target = ComputeTarget(workspace=self.ws, name=compute_name)
            logger.info(f'Found existing compute target: {compute_name}')
            return compute_target
        except ComputeTargetException:
            logger.info(f'Creating new compute target: {compute_name}')
            
            # List of VM sizes to try (CPU-focused for your region)
            vm_sizes_to_try = [
                "Standard_DS3_v2",    # 4 cores, 14GB RAM - good for CPU training
                "Standard_DS4_v2",    # 8 cores, 28GB RAM - better for larger models
                "Standard_D4s_v3",    # 4 cores, 16GB RAM - alternative
                "Standard_F4s_v2",    # 4 cores, 8GB RAM - compute optimized
                vm_size               # Original requested size
            ]
            
            for vm_size_attempt in vm_sizes_to_try:
                try:
                    logger.info(f'Trying VM size: {vm_size_attempt}')
                    compute_config = AmlCompute.provisioning_configuration(
                        vm_size=vm_size_attempt,
                        max_nodes=max_nodes,
                        min_nodes=0,
                        idle_seconds_before_scaledown=300
                    )
                    
                    compute_target = ComputeTarget.create(self.ws, compute_name, compute_config)
                    compute_target.wait_for_completion(show_output=True)
                    logger.info(f'Successfully created compute target with VM size: {vm_size_attempt}')
                    return compute_target
                    
                except Exception as e:
                    logger.warning(f'Failed to create compute with {vm_size_attempt}: {str(e)}')
                    continue
            
            # If all VM sizes fail, try to use existing compute instance
            logger.info('All VM sizes failed, checking for existing compute instances...')
            compute_targets = self.ws.compute_targets
            for name, compute in compute_targets.items():
                if compute.type == 'ComputeInstance' or compute.type == 'AmlCompute':
                    logger.info(f'Using existing compute: {name}')
                    return compute
            
            raise Exception("Could not create or find any suitable compute target")
    
    def create_environment(self, env_name="sem-classification-env"):
        """Create or get environment for the pipeline"""
        try:
            env = Environment.get(workspace=self.ws, name=env_name)
            logger.info(f"Using existing environment: {env_name}")
        except:
            logger.info(f"Creating new environment: {env_name}")
            
            # Create conda dependencies
            conda_deps = {
                'name': env_name,
                'dependencies': [
                    'python=3.8',
                    'pip',
                    {
                        'pip': [
                            'torch>=1.12.0',
                            'torchvision>=0.13.0',
                            'pillow>=8.0.0',
                            'numpy>=1.21.0',
                            'scikit-learn>=1.0.0',
                            'matplotlib>=3.5.0',
                            'seaborn>=0.11.0',
                            'scipy>=1.7.0',
                            'azureml-core>=1.45.0',
                            'azureml-defaults>=1.45.0'
                        ]
                    }
                ]
            }
            
            # Save conda file temporarily
            conda_file = "temp_conda_deps.yml"
            with open(conda_file, 'w') as f:
                yaml.dump(conda_deps, f)
            
            env = Environment.from_conda_specification(name=env_name, file_path=conda_file)
            env.register(workspace=self.ws)
            
            # Clean up temp file
            os.remove(conda_file)
        
        return env
    
    def step_1_generate_dataset(self, compute_target, environment, output_data):
        """Step 1: Generate SEM dataset"""
        logger.info("Creating dataset generation step...")
        
        generation_step = PythonScriptStep(
            name="generate_sem_dataset",
            script_name="generate_sem_dataset_azure_cldai.py",
            arguments=[
                "--output_dir", output_data,
                "--img_size", 128,
                "--num_images", 200,
                "--num_lines", 8,
                "--line_width", 6,
                "--gap_width", 6,
                "--seed", 42
            ],
            outputs=[output_data],
            compute_target=compute_target,
            runconfig=self._create_run_config(compute_target, environment),
            allow_reuse=False
        )
        
        return generation_step
    
    def step_2_train_model(self, compute_target, environment, input_data, model_output):
        """Step 2: Train the classification model"""
        logger.info("Creating model training step...")
        
        training_step = PythonScriptStep(
            name="train_sem_model",
            script_name="sem_train_model_cldai.py",
            arguments=[
                "--data_dir", input_data,
                "--csv_file", "data/labels.csv",
                "--batch_size", 16,
                "--num_epochs", 15,
                "--lr", 0.001,
                "--img_size", 128,
                "--validation_split", 0.2,
                "--early_stopping_patience", 3,
                "--output_dir", model_output
            ],
            inputs=[input_data],
            outputs=[model_output],
            compute_target=compute_target,
            runconfig=self._create_run_config(compute_target, environment),
            allow_reuse=False
        )
        
        return training_step
    
    def step_3_evaluate_model(self, compute_target, environment, model_input, evaluation_output):
        """Step 3: Comprehensive model evaluation"""
        logger.info("Creating model evaluation step...")
        
        evaluation_step = PythonScriptStep(
            name="evaluate_sem_model",
            script_name="evaluate_sem_model_cldai.py",
            arguments=[
                "--model_dir", model_input,
                "--output_dir", evaluation_output
            ],
            inputs=[model_input],
            outputs=[evaluation_output],
            compute_target=compute_target,
            runconfig=self._create_run_config(compute_target, environment),
            allow_reuse=False
        )
        
        return evaluation_step
    
    def create_complete_pipeline(self, compute_name="cpu-cluster"):
        """Create the complete pipeline with all steps"""
        logger.info("Creating complete SEM classification pipeline...")
        
        # Get compute target and environment with quota-aware sizing
        compute_target = self.create_or_get_compute_target(compute_name, vm_size="Standard_DS1_v2", max_nodes=1)
        environment = self.create_environment()
        
        # Define pipeline data
        dataset_output = PipelineData("dataset_output", datastore=self.datastore)
        model_output = PipelineData("model_output", datastore=self.datastore)
        evaluation_output = PipelineData("evaluation_output", datastore=self.datastore)
        
        # Create pipeline steps
        step1 = self.step_1_generate_dataset(compute_target, environment, dataset_output)
        step2 = self.step_2_train_model(compute_target, environment, dataset_output, model_output)
        step3 = self.step_3_evaluate_model(compute_target, environment, model_output, evaluation_output)
        
        # Create pipeline
        pipeline = Pipeline(
            workspace=self.ws,
            steps=[step1, step2, step3],
            description="Complete SEM Classification Pipeline: Generation -> Training -> Evaluation (CPU)"
        )
        
        return pipeline
    
    def run_complete_pipeline(self, compute_name="ml-primary-cpu-4c"):
        """Run the complete pipeline using existing compute"""
        logger.info("Starting complete SEM classification pipeline...")
        
        # Get existing compute target directly - don't try to create new one
        try:
            compute_target = ComputeTarget(workspace=self.ws, name=compute_name)
            logger.info(f'Using existing compute target: {compute_name}')
        except ComputeTargetException:
            logger.error(f'Compute target {compute_name} not found!')
            # List available compute targets
            compute_targets = self.ws.compute_targets
            logger.info("Available compute targets:")
            for name, compute in compute_targets.items():
                logger.info(f"  - {name}: {compute.type}")
            raise Exception(f"Compute target {compute_name} not found. Use one of the available compute targets above.")
        
        environment = self.create_environment()
        
        # Define pipeline data
        dataset_output = PipelineData("dataset_output", datastore=self.datastore)
        model_output = PipelineData("model_output", datastore=self.datastore)
        evaluation_output = PipelineData("evaluation_output", datastore=self.datastore)
        
        # Create pipeline steps
        step1 = self.step_1_generate_dataset(compute_target, environment, dataset_output)
        step2 = self.step_2_train_model(compute_target, environment, dataset_output, model_output)
        step3 = self.step_3_evaluate_model(compute_target, environment, model_output, evaluation_output)
        
        # Create pipeline
        pipeline = Pipeline(
            workspace=self.ws,
            steps=[step1, step2, step3],
            description="Complete SEM Classification Pipeline: Generation -> Training -> Evaluation (CPU)"
        )
        
        # Submit pipeline
        experiment = Experiment(workspace=self.ws, name=self.experiment_name)
        pipeline_run = experiment.submit(pipeline)
        logger.info(f"Pipeline submitted: {pipeline_run.id}")
        logger.info(f"Monitor at: {pipeline_run.get_portal_url()}")
        
        return pipeline_run
    
    def deploy_latest_model(self, local_model_path=None):
        """Deploy the latest trained model"""
        logger.info("Deploying latest model...")
        
        model = None
        
        # First try to find registered model in Azure ML
        try:
            model = Model(self.ws, self.model_name)
            logger.info(f"Found registered model: {model.name}, version: {model.version}")
        except:
            logger.info("No registered model found in Azure ML. Checking for local model...")
            
            # Try to register local model if it exists
            local_paths_to_try = [
                local_model_path,
                './outputs',
                './outputs/sem_model.pth'
            ]
            
            model_dir = None
            for path in local_paths_to_try:
                if path and os.path.exists(path):
                    if os.path.isdir(path):
                        # Check if it contains model files
                        if os.path.exists(os.path.join(path, 'sem_model.pth')):
                            model_dir = path
                            break
                    elif path.endswith('.pth'):
                        model_dir = os.path.dirname(path)
                        break
            
            if model_dir:
                logger.info(f"Found local model at: {model_dir}")
                try:
                    # Register the local model
                    model = Model.register(
                        workspace=self.ws,
                        model_name=self.model_name,
                        model_path=model_dir,
                        description='SEM Image Classification Model (Local Training)',
                        tags={'type': 'image_classification', 'framework': 'pytorch', 'source': 'local'}
                    )
                    logger.info(f"Successfully registered local model: {model.name}, version: {model.version}")
                except Exception as e:
                    logger.error(f"Failed to register local model: {e}")
                    return None
            else:
                logger.error("No model found. Please train a model first or provide model path.")
                logger.info("Available files in ./outputs:")
                if os.path.exists('./outputs'):
                    for file in os.listdir('./outputs'):
                        logger.info(f"  - {file}")
                return None
        
        if not model:
            logger.error("Could not find or register model for deployment")
            return None
        
        # Create scoring script for deployment
        scoring_script_content = '''
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pickle
import base64
from io import BytesIO
import os

def init():
    global model, class_to_idx, idx_to_class, transform, device
    
    # Load model configuration
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    
    # Load class mapping
    with open('class_mapping.pkl', 'rb') as f:
        class_to_idx = pickle.load(f)
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Initialize model
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
    
    # Load trained weights
    device = torch.device('cpu')
    model.load_state_dict(torch.load('sem_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

def run(raw_data):
    try:
        # Parse input data
        data = json.loads(raw_data)
        
        if isinstance(data, str):
            # Base64 encoded image
            image_data = base64.b64decode(data)
            image = Image.open(BytesIO(image_data)).convert('L')
        else:
            return {"error": "Invalid input format. Expected base64 encoded image."}
        
        # Apply transforms
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        predicted_class = idx_to_class[predicted_class_idx]
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'all_probabilities': {idx_to_class[i]: float(probabilities[0][i]) for i in range(len(idx_to_class))}
        }
    
    except Exception as e:
        return {'error': str(e)}
'''
        
        # Save scoring script
        scoring_script_path = 'score.py'
        with open(scoring_script_path, 'w') as f:
            f.write(scoring_script_content)
        
        # Create inference configuration
        inference_config = InferenceConfig(
            entry_script=scoring_script_path,
            environment=self.create_environment("inference-env")
        )
        
        # Configure deployment
        deployment_config = AciWebservice.deploy_configuration(
            cpu_cores=1,
            memory_gb=2,
            auth_enabled=True,
            enable_app_insights=True,
            description="SEM Image Classification Service"
        )
        
        try:
            # Check if service already exists
            service = Webservice(self.ws, self.service_name)
            logger.info(f"Updating existing service: {self.service_name}")
            service.update(models=[model], inference_config=inference_config)
        except:
            logger.info(f"Creating new service: {self.service_name}")
            service = Model.deploy(
                workspace=self.ws,
                name=self.service_name,
                models=[model],
                inference_config=inference_config,
                deployment_config=deployment_config
            )
        
        service.wait_for_deployment(show_output=True)
        logger.info(f"Service deployed successfully!")
        logger.info(f"Scoring URI: {service.scoring_uri}")
        
        # Clean up scoring script
        if os.path.exists(scoring_script_path):
            os.remove(scoring_script_path)
        
        return service
    
    def run_single_step(self, step_name, compute_name="ml-primary-cpu-4c", **kwargs):
        """Run a single step of the pipeline"""
        # Use existing compute directly - don't try to create
        try:
            compute_target = ComputeTarget(workspace=self.ws, name=compute_name)
            logger.info(f'Using existing compute target: {compute_name}')
        except ComputeTargetException:
            logger.error(f'Compute target {compute_name} not found!')
            compute_targets = self.ws.compute_targets
            logger.info("Available compute targets:")
            for name, compute in compute_targets.items():
                logger.info(f"  - {name}: {compute.type}")
            raise Exception(f"Compute target {compute_name} not found")
            
        environment = self.create_environment()
        experiment = Experiment(workspace=self.ws, name=f"{self.experiment_name}-{step_name}")
        
        if step_name == "generate":
            run_config = self._create_run_config(compute_target, environment)
            config = ScriptRunConfig(
                source_directory='.',
                script='generate_sem_dataset_azure_cldai.py',
                arguments=[
                    '--output_dir', './outputs',
                    '--img_size', kwargs.get('img_size', 128),
                    '--num_images', kwargs.get('num_images', 200),
                    '--num_lines', kwargs.get('num_lines', 8),
                    '--line_width', kwargs.get('line_width', 6),
                    '--gap_width', kwargs.get('gap_width', 6),
                    '--seed', kwargs.get('seed', 42)
                ],
                run_config=run_config
            )
        
        elif step_name == "train":
            run_config = self._create_run_config(compute_target, environment)
            config = ScriptRunConfig(
                source_directory='.',
                script='sem_train_model_cldai.py',
                arguments=[
                    '--data_dir', kwargs.get('data_dir', './outputs/data'),
                    '--csv_file', kwargs.get('csv_file', 'labels.csv'),
                    '--batch_size', kwargs.get('batch_size', 16),
                    '--num_epochs', kwargs.get('num_epochs', 15),
                    '--lr', kwargs.get('lr', 0.001),
                    '--img_size', kwargs.get('img_size', 128),
                    '--validation_split', kwargs.get('validation_split', 0.2),
                    '--early_stopping_patience', kwargs.get('early_stopping_patience', 3),
                    '--output_dir', './outputs'
                ] + (['--augment_data'] if kwargs.get('augment_data', False) else []),
                run_config=run_config
            )
        
        elif step_name == "evaluate":
            run_config = self._create_run_config(compute_target, environment)
            config = ScriptRunConfig(
                source_directory='.',
                script='evaluate_sem_model_cldai.py',
                arguments=[
                    '--model_dir', kwargs.get('model_dir', './outputs'),
                    '--output_dir', kwargs.get('output_dir', './outputs/evaluation')
                ],
                run_config=run_config
            )
        
        else:
            raise ValueError(f"Unknown step: {step_name}")
        
        run = experiment.submit(config)
        logger.info(f"Step '{step_name}' submitted: {run.id}")
        logger.info(f"Monitor at: {run.get_portal_url()}")
        
        return run

def create_evaluation_script():
    """Create the evaluation script for step 3"""
    
    eval_script_content = """# evaluate_sem_model_cldai.py
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
"""
    
    with open('evaluate_sem_model_cldai.py', 'w') as f:
        f.write(eval_script_content)
    
    logger.info("Created evaluation script: evaluate_sem_model_cldai.py")

def main():
    parser = argparse.ArgumentParser(description='Complete Azure ML Pipeline for SEM Classification')
    parser.add_argument('--action', type=str, required=True, 
                       choices=['full-pipeline', 'generate', 'train', 'evaluate', 'deploy', 'create-scripts'],
                       help='Action to perform')
    parser.add_argument('--compute_name', type=str, default='cpu-cluster', help='Compute target name')
    parser.add_argument('--wait', action='store_true', help='Wait for completion')
    
    # Generation parameters
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--num_images', type=int, default=200, help='Images per class (reduced for CPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Training parameters
    parser.add_argument('--data_dir', type=str, default='./outputs/data', help='Data directory (reduced for CPU)')
    parser.add_argument('--csv_file', type=str, default='./outputs/data/labels.csv', help='CSV file name (reduced for CPU)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (reduced for CPU)')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs (reduced for CPU)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SEMAzureMLPipeline()
    
    if args.action == 'create-scripts':
        create_evaluation_script()
        logger.info("All required scripts created!")
        logger.info("Note: Scripts are optimized for CPU compute")
        return
    
    elif args.action == 'full-pipeline':
        logger.info("Running complete pipeline (CPU optimized)...")
        pipeline_run = pipeline.run_complete_pipeline()
        
        if args.wait:
            logger.info("Waiting for pipeline completion...")
            pipeline_run.wait_for_completion(show_output=True)
            
            # Auto-deploy if successful
            if pipeline_run.get_status() == 'Completed':
                logger.info("Pipeline completed successfully! Deploying model...")
                try:
                    pipeline.deploy_latest_model()
                except Exception as e:
                    logger.warning(f"Deployment failed: {e}")
                    logger.info("You can deploy manually later using --action deploy")
    
    elif args.action == 'generate':
        logger.info("Running dataset generation...")
        run = pipeline.run_single_step('generate', 
                                     compute_name=args.compute_name,
                                     img_size=args.img_size,
                                     num_images=args.num_images,
                                     seed=args.seed)
        if args.wait:
            run.wait_for_completion(show_output=True)
    
    elif args.action == 'train':
        logger.info("Running model training (CPU optimized)...")
        run = pipeline.run_single_step('train',
                                     compute_name=args.compute_name,
                                     data_dir=args.data_dir,
                                     csv_file=args.csv_file,
                                     batch_size=args.batch_size,
                                     num_epochs=args.num_epochs,
                                     lr=args.lr,
                                     augment_data=False)
        if args.wait:
            run.wait_for_completion(show_output=True)
    
    elif args.action == 'evaluate':
        logger.info("Running model evaluation...")
        run = pipeline.run_single_step('evaluate',
                                     compute_name=args.compute_name,
                                     model_dir='./outputs',
                                     output_dir='./outputs/evaluation')
        if args.wait:
            run.wait_for_completion(show_output=True)
    
    elif args.action == 'deploy':
        logger.info("Deploying model...")
        try:
            service = pipeline.deploy_latest_model()
            if service:
                logger.info(f"Service deployed: {service.scoring_uri}")
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            logger.info("Make sure you have a trained model first by running --action train")

if __name__ == "__main__":
    main()