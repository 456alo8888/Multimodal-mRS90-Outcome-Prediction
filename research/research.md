---
date: 2026-04-17T00:00:00+00:00
researcher: GitHub Copilot
git_commit: 61c0e77604e642adfc9009e2f2d9d3891085b0f6
branch: main
repository: 4gpus-Stroke-outcome-prediction-code
topic: "Multimodal-mRS90 model: model, input, data, and how to run"
tags: [research, codebase, multimodal-mrs90, tensorflow, data-pipeline]
status: complete
last_updated: 2026-04-17
last_updated_by: GitHub Copilot
---

# Research: Multimodal-mRS90 model

Date: 2026-04-17
Researcher: GitHub Copilot
Git Commit: 61c0e77604e642adfc9009e2f2d9d3891085b0f6
Branch: main
Repository: 4gpus-Stroke-outcome-prediction-code

## Research Question
Read the codebase inside code/baseline/Multimodal-mRS90-Outcome-Prediction/python/model and document everything about the model, inputs, data, and how to run the model.

## Summary
The implementation under python/model defines a TensorFlow multimodal classifier for binary 90-day mRS outcome prediction. It combines:
- 4D CTP imaging processed timepoint-by-timepoint using a shared 3D CNN encoder.
- Clinical tabular metadata tokenized into categorical embeddings and continuous dense projections.
- Transformer-style self-attention per modality plus bidirectional cross-attention fusion.
- A final MLP and sigmoid output for binary prediction.

Training, validation, and testing are orchestrated in the main script, with data supplied by a Keras Sequence generator that reads per-patient NPZ volumes and clinical metadata indexed by PATIENT_ID.

## Components And Files

### Model package files
- python/model/__init__.py: empty module initializer.
- python/model/config.py: central static configuration for transformer/training/data/feature settings.
- python/model/data_generator.py: Keras Sequence for multimodal batch loading and formatting.
- python/model/lr_scheduler.py: custom cosine decay with restarts schedule.
- python/model/utils.py: model building blocks (ENCODER, SelfAttention, MultimodalFusion, AddCLSToken).
- python/model/main.py: executable training/evaluation pipeline.

### Project-level run and dependency docs
- README.md: install, data preparation, and training command.
- requirements.txt: dependency list.

## Configuration As Implemented
Source: python/model/config.py

### Transformer parameters
- n_layers: 1
- n_heads: 8
- dropout_rate: 0.2

### Clinical metadata definition
- Categorical feature names: Sex, Hypertension, Smoking, Atrial_Fibrillation, mTICI
- Categorical feature category sizes: [2, 2, 2, 2, 3]
- Continuous feature names: Age, Onset2CTP, CTP2Recanalization, NIHSS00
- Target feature name: mRs90_binary (binary target)

### Training parameters
- n_epochs: 100
- learning_rate: 0.001

### Generator/runtime parameters
- imagePath: ./datasets/
- dictFile: ./datasets/patient_dictionary.pickle
- clinicalFile: ./datasets/clinical_metadata.csv
- resultsPath: ./results/
- dim: (512, 512, 16)
- batch_size: 1
- timepoints: 32
- n_classes: 2
- features: [continuous_features, categorical_feature_names]

## Input And Data Contract

### Imaging input format
Source: python/model/data_generator.py
- Expected storage per patient: imagePath + PatientID + /preprocessed.npz.
- Expected NPZ key: img.
- Expected image tensor shape in NPZ: (height, width, n_slices, n_timepoints).
- Runtime tensor allocation in generator: (batch_size, dim_x, dim_y, dim_z, timepoints).

### Clinical metadata input format
Sources: python/model/main.py, python/model/config.py
- Clinical CSV path from config: ./datasets/clinical_metadata.csv.
- CSV is loaded with index_col=PATIENT_ID.
- Model reads continuous + categorical columns listed in config.
- Target is read from column mRs90_binary.

### Patient split dictionary
Sources: README.md, python/model/main.py
- README documents a pickle split dictionary structure.
- main.py loads a pickle dictionary and uses partition keys training and testing for generator creation.

### Generator output structure
Source: python/model/data_generator.py
For each batch, generator returns:
- X: list of timepoint tensors, formed by splitting the 5D image tensor along time axis.
- C_categorical: list of per-feature categorical tensors.
- C_continuous: list of per-feature continuous tensors.
- y: batch of scalar binary labels.

Return signature:
- inputs: [X, C_categorical, C_continuous]
- targets: y

## Model Architecture Documentation

### 1) Imaging encoder
Sources: python/model/utils.py, python/model/main.py
- ENCODER.build creates a shared 3D CNN encoder with stacked Conv3D blocks and downsampling Conv3D layers.
- For each of 32 timepoints, main.py creates an Input of shape (512, 512, 16, 1) and applies the same encoder.
- Timepoint latent vectors are concatenated along sequence axis to form imaging_encoded.

### 2) Clinical tokenization
Source: python/model/main.py
- Categorical features:
  - One Input(shape=(1,), dtype=int32) per categorical field.
  - Embedding(input_dim=category_size, output_dim=embed_dim) per field.
  - Concatenated into categorical token sequence.
- Continuous features:
  - One Input(shape=(1,), dtype=float32) per continuous field.
  - Dense(embed_dim, relu) projection per scalar.
  - Expanded along token axis and concatenated into continuous token sequence.
- Continuous and categorical token sequences are concatenated into metadata_encoded.

### 3) Self-attention per modality
Sources: python/model/utils.py, python/model/main.py
- SelfAttention.build is applied to imaging_encoded and metadata_encoded.
- Adds positional embeddings based on sequence length.
- Prepends a learnable CLS token via AddCLSToken.
- Applies MultiHeadAttention + residual LayerNormalization for configured number of layers.

### 4) Cross-modal fusion
Source: python/model/utils.py
- MultimodalFusion.build performs two cross-attention passes:
  - Imaging query over clinical key/value.
  - Clinical query over imaging key/value.
- Each branch applies residual + LayerNorm + Dense(GELU) + Dropout.
- Extracts CLS token from each branch output.
- Concatenates both CLS vectors into fused feature vector.

### 5) Prediction head
Source: python/model/main.py
- Applies MLP blocks with hidden sizes derived from fused feature width using factors [2, 1].
- Each hidden layer uses Dense(relu) + Dropout(0.2).
- Final output layer: Dense(1, sigmoid).

## Training, Optimization, And Outputs

### Preprocessing in main
Source: python/model/main.py
- Sets random seeds for reproducibility.
- Loads clinical metadata and target.
- Applies MinMaxScaler to continuous variables and writes normalized values back into metadata dataframe.
- Loads split dictionary from pickle.
- Instantiates train/validation/test generators.

### Learning-rate schedule
Sources: python/model/main.py, python/model/lr_scheduler.py
- Uses custom CosineDecayRestarts schedule.
- total_steps computed from training partition size, batch_size, and n_epochs.
- Scheduler used in Adam optimizer as learning_rate.

### Compilation and fitting
Source: python/model/main.py
- Loss: BinaryCrossentropy.
- Optimizer: Adam(learning_rate=scheduler).
- Metrics: accuracy.
- Training call: model.fit with train_generator and val_generator for configured epochs.

### Evaluation and persistence
Source: python/model/main.py
- Runs predict on test_generator.
- Saves predictions with np.savez_compressed to resultsPath + predictions.
- Code also includes commented save lines for model weights and train history.

## How To Run (Documented Workflow)

### 1) Install
Source: README.md
- Clone repository.
- Change into project directory.
- Install dependencies from requirements.txt.

### 2) Prepare data
Sources: README.md, python/model/data_generator.py, python/model/config.py
- Preprocess images to default spatial size 512 x 512 x 16.
- Save per-patient NPZs with image array under key img.
- Place data under configured datasets path.
- Prepare clinical_metadata.csv with PATIENT_ID as index column and expected feature/target columns.
- Prepare patient split pickle dictionary and place at configured dict file path.

### 3) Configure
Source: README.md, python/model/config.py
- Adjust values in python/model/config.py for paths, dimensions, feature definitions, and training parameters.

### 4) Train
Source: README.md
- Run: python ./python/model/main.py

### 5) Evaluate artifacts
Source: python/model/main.py
- Predictions are produced on test generator and saved under configured results path.

## Code References
- python/model/config.py:1 - Configuration dictionaries for transformer, features, training, and runtime paths.
- python/model/data_generator.py:9 - Data generator class and data format contract.
- python/model/data_generator.py:53 - Batch tensor allocation for image/metadata/labels.
- python/model/data_generator.py:60 - Per-patient NPZ loading from PatientID/preprocessed.npz and key img.
- python/model/data_generator.py:66 - Conversion of tensors to list-based multi-input format.
- python/model/utils.py:11 - ENCODER class (3D CNN feature extractor).
- python/model/utils.py:43 - SelfAttention with positional embedding, CLS token, and MHSA blocks.
- python/model/utils.py:66 - MultimodalFusion bidirectional cross-attention and CLS pooling.
- python/model/utils.py:92 - AddCLSToken learnable class token layer.
- python/model/lr_scheduler.py:10 - CosineDecayRestarts learning rate schedule.
- python/model/main.py:44 - Clinical metadata and target loading.
- python/model/main.py:64 - Generator construction for train/val/test.
- python/model/main.py:87 - Model assembly from encoder through fusion and classifier.
- python/model/main.py:162 - Compile configuration.
- python/model/main.py:172 - Training with model.fit.
- python/model/main.py:187 - Inference and prediction export.
- README.md:20 - Environment and installation notes.
- README.md:33 - Data preparation instructions.
- README.md:60 - Model training command.
- requirements.txt:1 - Project dependencies.

## Architecture And Interaction Map
1. config.py defines all static hyperparameters, feature schemas, and paths.
2. main.py loads config and data, scales continuous metadata, loads split dictionary, and builds generators.
3. data_generator.py reads images and tabular entries per patient ID and formats a multi-input batch.
4. utils.py blocks are used by main.py to construct encoder, attention, and fusion pipeline.
5. lr_scheduler.py is instantiated in main.py and passed to Adam.
6. main.py executes training and evaluation and writes prediction artifacts.
