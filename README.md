# SEM Classification Pipeline

Generated on: 2025-06-04 12:35:26

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate dataset:
   ```bash
   python generate_sem_dataset_azure_cldai.py --output_dir ./outputs --num_images 100
   ```

3. Train model:
   ```bash
   python sem_train_model_cldai.py --data_dir ./outputs/data --csv_file ./outputs/data/labels.csv --batch_size 8 --num_epochs 10 --output_dir ./outputs
   ```

4. Evaluate model:
   ```bash
   python evaluate_sem_model_cldai.py --model_dir ./outputs --output_dir ./outputs/evaluation
   ```

5. Test inference:
   ```bash
   python local_inference_test.py --test
   ```

## Results Achieved
- Overall Accuracy: 89.33%
- Gap Detection: 100%
- Bridge Detection: 100%
- Training Time: 36 seconds

## Files Included
- azure_ml_complete_pipeline_cldai.py
- generate_sem_dataset_azure_cldai.py
- sem_train_model_cldai.py
- evaluate_sem_model_cldai.py
- local_inference_test_cldai.py
- register_model_cldai.py
- quick_test_model_cldai.py
- config.json

For complete documentation and advanced usage, see the full GitHub repository.
