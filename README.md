# README Template
------------------------------------------------------------
## 1. Project Overview
Project Title: Partial Retinal Image Reconstruction with CVAE   
Model Type: Conditional Variational Autoencoder   
Objective: Reconstruction   
Dataset Used: OCTA 500 6mm 64x64x64 Filtered (https://www.kaggle.com/datasets/humbgruland/octa-500-6mm-64x64x64-filtered)   
Expected test evaluation for sanity check: {{Loss=1.90E-04, PSNR=39.106, SSIM=0.9812}}   
------------------------------------------------------------
## 2. Repository Structure
```
project_root/
README.md
CVAE.py  – Defines the 3D Conditional Variational Autoencoder architecture and loss function.
metrics.py – Implements PSNR and 3D SSIM evaluation metrics.
requirements.txt – Lists Python dependencies required to run the project.
OCTADataset.py – Provides dataset loading, splitting, and sampling-mask generation for OCTA volumes.
train_CVAE.py – Training script with validation, checkpointing, and metric logging.
test_CVAE.py – Inference and evaluation script for reconstructing and assessing trained models.
models/best_model.pt (contains the best trained model)
comparisons/ (shows test performance against ground truth after running test_CVAE.py)
logs/output.out (Shows the train, evaluation and test performance of the best model)
```
------------------------------------------------------------
## 3. Dataset
## OPTION A — PUBLIC DATASET SPLITS
Dataset Link: 
https://www.kaggle.com/datasets/humbgruland/octa-500-6mm-64x64x64-filtered

Where to place the downloaded dataset:
You can store the dataset anywhere on your machine (it is ∼32 GB), so it is best to keep it in its download location instead of copying it into the project directory. Extract the .zip, then locate the folder named processed_small_filtered and copy its full path. Use that path as the value for --dataset-path.

------------------------------------------------------------
## 4. Model Checkpoint
The best model can already be found in models/best_model, however it can also be found in the following box link:
https://usf.box.com/s/s3k8pxkh8fs1d5j16u0joraa5ts3p7bk

Place the model in:
```
models/
best_model.pt
```
------------------------------------------------------------
## 5. Requirements (Dependencies)
Using pip:
```
pip install -r requirements.txt
```
------------------------------------------------------------
## 6. Running the Test Script (DO)
Command to run testing:
```
python .\test_CVAE.py --dataset-path "Your_location_of_dataset (copy as path)" --save-comparisons-dir ".\comparisons" --num-comparisons 20

# PowerShell multi-line version:
python .\test_CVAE.py `
	--dataset-path "Your_location_of_dataset (copy as path)" `
	--save-comparisons-dir ".\comparisons" `
	--num-comparisons 20
```
Note: You must copy the dataset path from the folder "processed_small_filtered" that is inside the decompressed file.
The parameter num-comparisons determines the number of reconstructed vs ground truth test images that will be printed in the "comparisons" folder after testing.
------------------------------------------------------------
## 7. Running the Training Script
Command to run training:
```
python train_CVAE.py --dataset-path "Include dataset path here"
```
Note: You must copy the dataset path from the folder "processed_small_filtered" that is inside the decompressed file.
This command automatically runs the training script using the same parameters used for the best model explained in the document:
```
seed =42, train-batch-size = 8, val-batch-size=4, epochs = 10, sampling_rate = 0.6, beta = 0.5
```
To modify, please modify the values in train_CVAE.py or specify this values in the command.
Training requires a GPU. Running the model on a CPU may lead to memory overflow errors due to the high computational and memory demands of the architecture.
------------------------------------------------------------
