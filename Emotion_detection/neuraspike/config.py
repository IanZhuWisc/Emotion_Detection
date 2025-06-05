"""
Configuration Settings for Emotion Recognition System

This module contains all configuration parameters and hyperparameters for the
emotion recognition system. It provides a centralized location for managing
training parameters, data paths, and model configurations.

Key Configuration Areas:
- Dataset paths and splits
- Training hyperparameters
- Model architecture settings
- Data preprocessing parameters
- System optimization settings

Author: CS55100 Student
Date: 2024
"""

import os
from pathlib import Path

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# Dataset directory structure
DATASET_FOLDER = 'FER2013'  # Main dataset directory
TRAIN_DIRECTORY = os.path.join(DATASET_FOLDER, "train")
TEST_DIRECTORY = os.path.join(DATASET_FOLDER, "test")

# Data split ratios for training/validation
TRAIN_SIZE = 0.90  # 90% of training data for actual training
VAL_SIZE = 0.10    # 10% of training data for validation

# Emotion class configuration
NUM_EMOTION_CLASSES = 7
EMOTION_LABELS = {
    0: 'angry',
    1: 'disgust', 
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Batch processing
BATCH_SIZE = 16        # Number of samples per training batch
NUM_WORKERS = 4        # Number of data loading workers

# Training duration
NUM_OF_EPOCHS = 50     # Total training epochs
EARLY_STOPPING_PATIENCE = 10  # Epochs to wait before early stopping

# Learning rate configuration
LR = 1e-1              # Initial learning rate
MIN_LR = 1e-6          # Minimum learning rate for scheduler
LR_SCHEDULER_PATIENCE = 5     # Epochs to wait before reducing LR
LR_SCHEDULER_FACTOR = 0.5     # Factor to multiply LR by when reducing

# =============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# =============================================================================

# Input image specifications
INPUT_CHANNELS = 1     # 1 for grayscale, 3 for RGB
INPUT_HEIGHT = 48      # Model input height
INPUT_WIDTH = 48       # Model input width

# ResNet configuration (if using transfer learning)
RESNET_VERSION = 'resnet18'  # Available: resnet18, resnet34, resnet50
PRETRAINED = True            # Whether to use ImageNet pretrained weights
FREEZE_BACKBONE = False      # Whether to freeze ResNet backbone layers

# =============================================================================
# DATA PREPROCESSING CONFIGURATION
# =============================================================================

# Image augmentation settings
AUGMENTATION_ENABLED = True
ROTATION_RANGE = 20          # Random rotation range in degrees
WIDTH_SHIFT_RANGE = 0.1      # Horizontal shift range (fraction of width)
HEIGHT_SHIFT_RANGE = 0.1     # Vertical shift range (fraction of height)
SHEAR_RANGE = 0.2            # Shear transformation range
ZOOM_RANGE = 0.2             # Zoom range (fraction of original size)
HORIZONTAL_FLIP = True       # Whether to apply horizontal flipping

# Normalization settings
NORMALIZE_IMAGES = True      # Whether to normalize pixel values to [0,1]
USE_IMAGENET_STATS = False   # Whether to use ImageNet normalization

# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================

# Optimizer settings
OPTIMIZER = 'adam'           # Available: 'adam', 'sgd', 'adamw'
WEIGHT_DECAY = 1e-4          # L2 regularization factor
MOMENTUM = 0.9               # Momentum for SGD optimizer

# Loss function
LOSS_FUNCTION = 'cross_entropy'  # Available: 'cross_entropy', 'focal_loss'

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Hardware acceleration
USE_GPU = True               # Whether to use GPU if available
MIXED_PRECISION = True       # Whether to use mixed precision training

# Checkpointing and logging
SAVE_CHECKPOINTS = True      # Whether to save model checkpoints
CHECKPOINT_INTERVAL = 5      # Save checkpoint every N epochs
LOG_INTERVAL = 100           # Log training progress every N batches

# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

# Validation settings
VALIDATION_FREQUENCY = 1     # Validate every N epochs
METRICS = ['accuracy', 'f1_score', 'precision', 'recall']

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_dataset_paths():
    """
    Get absolute paths to dataset directories.
    
    Returns:
        dict: Dictionary containing dataset paths
    """
    base_path = Path.cwd()
    
    return {
        'dataset_folder': base_path / DATASET_FOLDER,
        'train_directory': base_path / TRAIN_DIRECTORY,
        'test_directory': base_path / TEST_DIRECTORY
    }

def get_training_config():
    """
    Get training configuration as a dictionary.
    
    Returns:
        dict: Training configuration parameters
    """
    return {
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_OF_EPOCHS,
        'learning_rate': LR,
        'min_lr': MIN_LR,
        'lr_scheduler_patience': LR_SCHEDULER_PATIENCE,
        'lr_scheduler_factor': LR_SCHEDULER_FACTOR,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'num_workers': NUM_WORKERS
    }

def get_model_config():
    """
    Get model configuration as a dictionary.
    
    Returns:
        dict: Model configuration parameters
    """
    return {
        'input_channels': INPUT_CHANNELS,
        'input_height': INPUT_HEIGHT,
        'input_width': INPUT_WIDTH,
        'num_classes': NUM_EMOTION_CLASSES,
        'resnet_version': RESNET_VERSION,
        'pretrained': PRETRAINED,
        'freeze_backbone': FREEZE_BACKBONE
    }

def get_augmentation_config():
    """
    Get data augmentation configuration as a dictionary.
    
    Returns:
        dict: Augmentation configuration parameters
    """
    return {
        'enabled': AUGMENTATION_ENABLED,
        'rotation_range': ROTATION_RANGE,
        'width_shift_range': WIDTH_SHIFT_RANGE,
        'height_shift_range': HEIGHT_SHIFT_RANGE,
        'shear_range': SHEAR_RANGE,
        'zoom_range': ZOOM_RANGE,
        'horizontal_flip': HORIZONTAL_FLIP
    }

def validate_config():
    """
    Validate configuration parameters for consistency.
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate data split ratios
    if TRAIN_SIZE + VAL_SIZE != 1.0:
        raise ValueError("TRAIN_SIZE + VAL_SIZE must equal 1.0")
    
    # Validate learning rate
    if LR <= 0 or MIN_LR <= 0:
        raise ValueError("Learning rates must be positive")
    
    if MIN_LR >= LR:
        raise ValueError("MIN_LR must be less than initial LR")
    
    # Validate batch size
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive")
    
    # Validate image dimensions
    if INPUT_HEIGHT <= 0 or INPUT_WIDTH <= 0:
        raise ValueError("Image dimensions must be positive")
    
    print("[INFO] Configuration validation passed")


# =============================================================================
# CONFIGURATION INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    print("Emotion Recognition System Configuration")
    print("=" * 50)
    
    # Validate configuration
    validate_config()
    
    # Display key configuration
    print(f"Dataset: {DATASET_FOLDER}")
    print(f"Training split: {TRAIN_SIZE*100:.0f}% / {VAL_SIZE*100:.0f}%")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Epochs: {NUM_OF_EPOCHS}")
    print(f"Input size: {INPUT_HEIGHT}x{INPUT_WIDTH}")
    print(f"Emotion classes: {NUM_EMOTION_CLASSES}")
    
    # Display paths
    paths = get_dataset_paths()
    print(f"\nDataset paths:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    
    print("\nConfiguration loaded successfully!")
