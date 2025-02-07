# MultiWave

The code for the paper: Deznabi, Iman, and Madalina Fiterau. "MultiWave: Multiresolution Deep Architectures through Wavelet Decomposition for Multivariate Time Series Prediction." Conference on Health, Inference, and Learning. PMLR, 2023. https://proceedings.mlr.press/v209/deznabi23a.html

Please cite this paper if you use the code in this repository as part of a published research project.

## Overview

MultiWave implements multiresolution deep architectures for time series prediction using wavelet decomposition. The repository includes code for data processing, model definitions, training routines, and evaluation utilities.

## Repository Structure

- **MultiWave/main.py**: Entry point for training and evaluation. See [main.py](main.py).
- **MultiWave/DownloadProcessedWESADdata.py**: Script to download and extract the processed WESAD dataset.
- **MultiWave/Models/**: Contains model definitions, fusion functions, and training routines:
  - [`Fusions.py`](Models/Fusions.py)
  - [`Routines.py`](Models/Routines.py)
  - Other model-specific classes and wrappers.
- **MultiWave/utils/**: Provides utility functions for dataset handling, model training, loss computation, and wavelet transformations:
  - [`Dataset.py`](utils/Dataset.py)
  - [`ModelUtils.py`](utils/ModelUtils.py)
  - [`pytorchtools.py`](utils/pytorchtools.py)
  - [`WaveletUtils.py`](utils/WaveletUtils.py)

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/username/MultiWave.git
   cd MultiWave
   ```

2. **Install required packages**: Make sure you have conda installed. Then, create and activate a new environment with the dependencies:
   ```sh
   conda env create -f environment.yml
   conda activate MultiWave
   ```

3. **Set up dataset**: Run the provided script to download and extract the WESAD dataset:
   ```sh
   python DownloadProcessedWESADdata.py
   ```

## Usage
**Training a Model**
To train a model, run the main script with appropriate arguments. For example:
```sh
python main.py --hs "[8, 8, 8, 8, 8, 0]" --d 32 --seed 123 --Fusion LinearFusion --Routine FeatNormLossWrapper --SubRoutine OnlyLastLoss --UseExtraLinear False --epochstotrain -1 --LW 0.1 --InitWs 0.5 --InitTemp 10.0 --Model Modelfreq_featMasks --Comp FCN_perchannel --NumLayers 1 --WaveletType db1 --LR 0.001
```
Refer to the argument definitions in main.py for available options.

## Evaluation
Model evaluation and logging are integrated with WandB. Once training completes, performance metrics such as accuracy, AUC, and confusion matrices (for classification) or MSE, MAE, and R2 (for regression) are printed and logged.

## Code Details

### Models and Routines
The `Models` folder defines various routines (e.g., `LossSwitches`, `CosineLosses`, `ResetModuleWrapper`) which handle training dynamics, loss weighting, and model optimization. These routines also interface with WandB for logging, as shown in the `wandblog` methods.

### Utilities
The `utils` folder contains helper functions:
- **Dataset.py**: Handles data loading and tensor conversion.
- **ModelUtils.py**: Contains training loops, loss functions, and evaluation functions.
- **WaveletUtils.py**: Provides utilities for wavelet transformations.

### Logging with WandB
Logging is integrated across training and evaluation routines. Functions such as `wandblog` and `wandbLossLogs` are used to track training progress and model performance.

## Citation
If you use this code in your research, please cite:

Deznabi, Iman, and Madalina Fiterau. "MultiWave: Multiresolution Deep Architectures through Wavelet Decomposition for Multivariate Time Series Prediction." Conference on Health, Inference, and Learning. PMLR, 2023.
