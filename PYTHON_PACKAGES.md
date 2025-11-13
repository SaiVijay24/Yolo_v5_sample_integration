# Python Packages Used in Yolo_v5_sample_integration

This document provides a comprehensive list of all Python packages used in this YOLOv5 sample integration repository.

## Core Dependencies (requirements.txt)

### Base Packages
- **matplotlib** (>=3.2.2) - Plotting library for creating visualizations
- **numpy** (>=1.18.5) - Fundamental package for numerical computing
- **opencv-python** (>=4.1.2) - Computer vision library for image processing
- **Pillow** (>=7.1.2) - Python Imaging Library for image manipulation
- **PyYAML** (>=5.3.1) - YAML parser and emitter for configuration files
- **requests** (>=2.23.0) - HTTP library for making web requests
- **scipy** (>=1.4.1) - Scientific computing library
- **torch** (>=1.7.0) - PyTorch deep learning framework
- **torchvision** (>=0.8.1) - Computer vision models and utilities for PyTorch
- **tqdm** (>=4.41.0) - Progress bar library for loops and iterations

### Logging Packages
- **tensorboard** (>=2.4.1) - TensorFlow's visualization toolkit for ML experiments
- **wandb** (commented out in requirements.txt, but used in Dockerfile) - Weights & Biases for experiment tracking

### Plotting Packages
- **pandas** (>=1.1.4) - Data manipulation and analysis library
- **seaborn** (>=0.11.0) - Statistical data visualization library

### Export Packages (Optional/Commented)
- **coremltools** (>=4.1) - Tools for converting models to Core ML format
- **onnx** (>=1.9.0) - Open Neural Network Exchange format
- **onnx-simplifier** (>=0.3.6) - Simplify ONNX models
- **scikit-learn** (==0.19.2) - Machine learning library (for CoreML quantization)
- **tensorflow** (>=2.4.1) - TensorFlow framework for TFLite export
- **tensorflowjs** (>=3.9.0) - TensorFlow.js for JavaScript export
- **openvino-dev** - Intel OpenVINO toolkit for model optimization

### Extra Packages (Optional/Commented)
- **albumentations** (>=1.0.3) - Image augmentation library (installed in Dockerfile)
- **Cython** - C-extensions for Python (for pycocotools)
- **pycocotools** (>=2.0) - COCO dataset API for metrics computation
- **roboflow** - Computer vision dataset management
- **thop** - PyTorch FLOPs (floating point operations) computation

### Additional Packages from Dockerfile
- **notebook** - Jupyter notebook environment
- **gsutil** - Google Cloud Storage command-line tool
- **wandb** - Weights & Biases (experiment tracking)
- **albumentations** - Image augmentation library

## Python Standard Library Modules Used

The following standard library modules are imported in the codebase:

- **argparse** - Command-line argument parsing
- **copy** - Shallow and deep copy operations
- **datetime** - Date and time handling
- **json** - JSON encoding and decoding
- **math** - Mathematical functions
- **os** - Operating system interface
- **pathlib** - Object-oriented filesystem paths
- **platform** - Access to platform-identifying data
- **pdb** - Python debugger
- **random** - Generate pseudo-random numbers
- **subprocess** - Subprocess management
- **sys** - System-specific parameters and functions
- **threading** - Thread-based parallelism
- **time** - Time access and conversions
- **warnings** - Warning control

## Package Categories

### Deep Learning & Neural Networks
- torch
- torchvision
- torch.nn
- torch.cuda.amp
- torch.distributed
- torch.optim
- torch.utils.mobile_optimizer

### Computer Vision
- opencv-python (cv2)
- Pillow
- albumentations

### Data Processing & Scientific Computing
- numpy
- pandas
- scipy

### Visualization & Plotting
- matplotlib
- seaborn
- tensorboard

### Model Export & Deployment
- onnx
- onnx-simplifier
- coremltools
- tensorflow
- tensorflowjs
- openvino-dev

### Utilities
- PyYAML
- requests
- tqdm
- thop

### Experiment Tracking
- tensorboard
- wandb

### Dataset & Annotations
- pycocotools
- roboflow

## Docker Environment Specific

The Dockerfile installs the following versions:
- **torch** (1.11.0+cu113) - CUDA 11.3 version
- **torchvision** (0.12.0+cu113) - CUDA 11.3 version

Base image: `nvcr.io/nvidia/pytorch:21.10-py3`

## Installation

To install all required packages:

```bash
pip install -r requirements.txt
```

To install with optional packages for export:
```bash
pip install -r requirements.txt coremltools onnx onnx-simplifier tensorflow tensorflowjs
```

To install with all extras:
```bash
pip install -r requirements.txt albumentations pycocotools roboflow wandb
```

## Notes

1. Some packages in requirements.txt are commented out and marked as optional
2. The Dockerfile includes additional packages (albumentations, wandb, gsutil, notebook) not in requirements.txt
3. PyTorch versions may vary - requirements.txt specifies >=1.7.0, but Dockerfile uses 1.11.0+cu113
4. The project uses custom modules from `models/` and `utils/` directories
