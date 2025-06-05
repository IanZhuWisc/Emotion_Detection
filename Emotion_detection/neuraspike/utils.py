"""
Utility Functions for Emotion Recognition System

This module provides essential utility functions and classes for the emotion
recognition pipeline, including learning rate scheduling, early stopping,
and image processing utilities.

Key Components:
- LRScheduler: Adaptive learning rate scheduling based on validation loss
- EarlyStopping: Prevents overfitting by monitoring validation performance
- Image utilities: Efficient image resizing and preprocessing functions

Author: CS55100 Student
Date: 2024
"""

from torch.optim import lr_scheduler
import cv2
import numpy as np


class LRScheduler:
    """
    Adaptive Learning Rate Scheduler for Training Optimization.
    
    This class implements a ReduceLROnPlateau scheduler that automatically
    reduces the learning rate when the validation loss plateaus. This helps
    with convergence and prevents the model from getting stuck in local minima.
    
    The scheduler monitors validation loss and reduces the learning rate by
    a specified factor when the loss doesn't improve for a given number of
    epochs (patience).
    
    Attributes:
        optimizer: PyTorch optimizer to schedule
        patience: Number of epochs to wait before reducing LR
        min_lr: Minimum learning rate threshold
        factor: Multiplicative factor for LR reduction
        lr_scheduler: PyTorch ReduceLROnPlateau scheduler
    """
    
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        """
        Initialize the learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer to schedule
            patience (int): Number of epochs to wait before reducing LR
            min_lr (float): Minimum learning rate value
            factor (float): Factor by which to multiply the learning rate
            
        Note:
            The new learning rate is calculated as: new_lr = old_lr * factor
            This continues until the learning rate reaches min_lr.
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        
        # Initialize PyTorch's ReduceLROnPlateau scheduler
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min",           # Monitor validation loss (minimize)
            patience=patience,    # Wait this many epochs
            factor=factor,        # Multiply LR by this factor
            min_lr=min_lr,       # Don't go below this LR
            verbose=True          # Print LR changes
        )

    def __call__(self, validation_loss):
        """
        Update learning rate based on validation loss.
        
        Args:
            validation_loss (float): Current validation loss value
            
        Note:
            This method should be called after each validation step.
            The scheduler will automatically reduce the learning rate
            if the validation loss doesn't improve for 'patience' epochs.
        """
        self.lr_scheduler.step(validation_loss)
    
    def get_last_lr(self):
        """
        Get the current learning rate.
        
        Returns:
            list: Current learning rate for each parameter group
        """
        return self.lr_scheduler.optimizer.param_groups[0]['lr']
    
    def state_dict(self):
        """
        Get the scheduler state for checkpointing.
        
        Returns:
            dict: Scheduler state dictionary
        """
        return self.lr_scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        """
        Load scheduler state from checkpoint.
        
        Args:
            state_dict (dict): Scheduler state dictionary
        """
        self.lr_scheduler.load_state_dict(state_dict)


class EarlyStopping:
    """
    Early Stopping Mechanism for Training Optimization.
    
    This class implements early stopping to prevent overfitting by monitoring
    validation loss and stopping training when the model stops improving.
    
    Early stopping is crucial for deep learning models as it:
    - Prevents overfitting by stopping training at optimal point
    - Saves computational resources
    - Improves model generalization
    - Provides automatic model selection
    
    Attributes:
        early_stop_enabled (bool): Whether early stopping has been triggered
        min_delta (float): Minimum improvement threshold
        patience (int): Number of epochs to wait before stopping
        best_loss (float): Best validation loss seen so far
        counter (int): Counter for epochs without improvement
    """
    
    def __init__(self, patience=10, min_delta=0):
        """
        Initialize the early stopping mechanism.
        
        Args:
            patience (int): Number of epochs to wait before stopping training
            min_delta (float): Minimum difference in loss to consider improvement
                
        Note:
            min_delta helps prevent stopping due to very small fluctuations
            in validation loss. A value of 0 means any improvement is considered.
        """
        self.early_stop_enabled = False
        self.min_delta = min_delta
        self.patience = patience
        self.best_loss = None
        self.counter = 0

    def __call__(self, validation_loss):
        """
        Check if training should be stopped based on validation loss.
        
        Args:
            validation_loss (float): Current validation loss
            
        Returns:
            bool: True if training should be stopped, False otherwise
            
        Note:
            This method should be called after each validation step.
            It tracks the best loss and counts epochs without improvement.
        """
        # First validation loss - set as baseline
        if self.best_loss is None:
            self.best_loss = validation_loss
            return False

        # Check if current loss is better than best loss
        improvement = self.best_loss - validation_loss
        
        if improvement < self.min_delta:
            # No significant improvement - increment counter
            self.counter += 1
            print(f"[INFO] Early stopping: {self.counter}/{self.patience} epochs without improvement")
            
            # Check if patience exceeded
            if self.counter >= self.patience:
                self.early_stop_enabled = True
                print(f"[INFO] Early stopping triggered after {self.patience} epochs")
                return True
        else:
            # Significant improvement - reset counter and update best loss
            self.best_loss = validation_loss
            self.counter = 0
            
        return False
    
    def reset(self):
        """Reset the early stopping state."""
        self.early_stop_enabled = False
        self.best_loss = None
        self.counter = 0
    
    def get_best_loss(self):
        """
        Get the best validation loss seen so far.
        
        Returns:
            float: Best validation loss value
        """
        return self.best_loss


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize image while maintaining aspect ratio.
    
    This function resizes an image to specified dimensions while preserving
    the original aspect ratio. It's optimized for downsampling using
    INTER_AREA interpolation which is ideal for reducing image size.
    
    Args:
        image (numpy.ndarray): Input image to resize
        width (int, optional): Target width in pixels
        height (int, optional): Target height in pixels
        inter (int): OpenCV interpolation method
        
    Returns:
        numpy.ndarray: Resized image
        
    Note:
        - If both width and height are None, original image is returned
        - If only one dimension is specified, the other is calculated
          to maintain aspect ratio
        - INTER_AREA is recommended for downsampling (better quality)
        - INTER_LINEAR is better for upsampling
        
    Example:
        >>> # Resize to width 300, maintain aspect ratio
        >>> resized = resize_image(img, width=300)
        >>> 
        >>> # Resize to height 200, maintain aspect ratio
        >>> resized = resize_image(img, height=200)
        >>> 
        >>> # Resize to exact dimensions (may distort)
        >>> resized = resize_image(img, width=300, height=200)
    """
    # Return original image if no dimensions specified
    if width is None and height is None:
        return image

    # Get original dimensions
    (h, w) = image.shape[:2]
    
    # Calculate new dimensions maintaining aspect ratio
    if width is None:
        # Height specified - calculate width
        ratio = height / float(h)
        dimension = (int(w * ratio), height)
    elif height is None:
        # Width specified - calculate height
        ratio = width / float(w)
        dimension = (width, int(h * ratio))
    else:
        # Both dimensions specified - use exact values
        dimension = (width, height)

    # Resize image using specified interpolation
    resized_image = cv2.resize(image, dimension, interpolation=inter)
    
    return resized_image


def normalize_image(image, mean=None, std=None):
    """
    Normalize image pixel values.
    
    Args:
        image (numpy.ndarray): Input image
        mean (tuple, optional): Mean values for normalization
        std (tuple, optional): Standard deviation values for normalization
        
    Returns:
        numpy.ndarray: Normalized image
        
    Note:
        If mean/std not provided, uses ImageNet statistics:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet mean
    if std is None:
        std = [0.229, 0.224, 0.225]   # ImageNet std
    
    # Convert to float32 for normalization
    image = image.astype(np.float32)
    
    # Normalize each channel
    for i in range(image.shape[2]):
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
    
    return image


def preprocess_for_model(image, target_size=(48, 48), normalize=True):
    """
    Complete preprocessing pipeline for model input.
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Target dimensions (width, height)
        normalize (bool): Whether to normalize pixel values
        
    Returns:
        numpy.ndarray: Preprocessed image ready for model input
    """
    # Resize image
    resized = resize_image(image, width=target_size[0], height=target_size[1])
    
    # Normalize if requested
    if normalize:
        resized = normalize_image(resized)
    
    return resized


# Example usage and testing
if __name__ == "__main__":
    print("Emotion Recognition Utilities Test")
    print("=" * 40)
    
    # Test image resizing
    test_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    print(f"Original image shape: {test_image.shape}")
    
    resized_width = resize_image(test_image, width=75)
    print(f"Resized to width 75: {resized_width.shape}")
    
    resized_height = resize_image(test_image, height=50)
    print(f"Resized to height 50: {resized_height.shape}")
    
    # Test preprocessing
    preprocessed = preprocess_for_model(test_image, target_size=(48, 48))
    print(f"Preprocessed shape: {preprocessed.shape}")
    print(f"Preprocessed dtype: {preprocessed.dtype}")
    
    print("\nUtility functions working correctly!")
