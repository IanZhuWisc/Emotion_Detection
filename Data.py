"""
Data Processing Pipeline for Emotion Detection System

This module provides comprehensive data loading, preprocessing, and augmentation
capabilities for the FER2013 emotion recognition dataset. It implements
industry-standard data pipeline practices including normalization, augmentation,
and efficient batch processing.

Author: CS55100 Student
Date: 2024
"""

import os
import cv2
import numpy as np

# Prefer TensorFlow Keras; fall back to standalone Keras if available
try:
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception:  # pragma: no cover - fallback for environments with standalone Keras
    from keras.utils import to_categorical
    from keras.preprocessing.image import ImageDataGenerator

# Dataset paths
TRAIN_PATH = 'FER2013/train'
TEST_PATH = 'FER2013/test'

# Emotion label mapping for consistent encoding
EMOTION_LABELS = {
    'angry': 0, 
    'disgust': 1, 
    'fear': 2, 
    'happy': 3, 
    'neutral': 4, 
    'sad': 5, 
    'surprise': 6
}

def load_data(dataset_path):
    """
    Load images and labels from the specified dataset directory.
    
    Args:
        dataset_path (str): Path to the dataset directory containing emotion subfolders
        
    Returns:
        tuple: (images_array, labels_array) where images are numpy arrays and labels are emotion names
        
    Note:
        Each emotion subfolder should contain images for that emotion category.
        Images are loaded using OpenCV and stored as numpy arrays.
    """
    data = []
    labels = []

    for emotion_folder in os.listdir(dataset_path):
        emotion_path = os.path.join(dataset_path, emotion_folder)
        if os.path.isdir(emotion_path):
            for img_name in os.listdir(emotion_path):
                img_path = os.path.join(emotion_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:  # Ensure image loaded successfully
                    data.append(img)
                    labels.append(emotion_folder)

    return np.array(data), np.array(labels)

def extract_labels(labels):
    """
    Convert emotion string labels to integer encodings.
    
    Args:
        labels (array): Array of emotion string labels
        
    Returns:
        numpy.ndarray: Integer-encoded labels using the EMOTION_LABELS mapping
    """
    return np.array([EMOTION_LABELS[label] for label in labels])

def resize_images(images, target_size=(64, 64)):
    """
    Resize all images to a consistent target size.
    
    Args:
        images (numpy.ndarray): Array of input images
        target_size (tuple): Target dimensions (width, height)
        
    Returns:
        numpy.ndarray: Resized images array
    """
    return np.array([cv2.resize(img, target_size) for img in images])

def normalize_images(images):
    """
    Normalize pixel values to the range [0, 1].
    
    Args:
        images (numpy.ndarray): Input images with pixel values in [0, 255]
        
    Returns:
        numpy.ndarray: Normalized images with pixel values in [0, 1]
    """
    return images.astype(np.float32) / 255.0

def augment_data(images):
    """
    Apply data augmentation techniques to increase dataset diversity.
    
    This function implements several augmentation strategies:
    - Random rotation (±20 degrees)
    - Width/height shifts (±10%)
    - Shear transformation (±20%)
    - Zoom scaling (±20%)
    - Horizontal flipping
    
    Args:
        images (numpy.ndarray): Input images to augment
        
    Returns:
        numpy.ndarray: Augmented images array
        
    Note:
        Augmentation helps improve model generalization and robustness
        by simulating real-world variations in facial expressions.
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    augmented_images = []
    for img in images:
        img = img.reshape((1,) + img.shape)
        augmented_batch = datagen.flow(img, batch_size=1)
        augmented_images.append(augmented_batch[0][0])

    return np.array(augmented_images)

def preprocess_data(dataset_path, augment=False, target_size=(64, 64)):
    """
    Complete data preprocessing pipeline for emotion recognition.
    
    This function orchestrates the entire data preparation workflow:
    1. Load raw images and labels
    2. Convert labels to integer encodings
    3. Resize images to target dimensions
    4. Normalize pixel values
    5. Optionally apply data augmentation
    
    Args:
        dataset_path (str): Path to the dataset directory
        augment (bool): Whether to apply data augmentation
        target_size (tuple): Target image dimensions (width, height)
        
    Returns:
        tuple: (processed_images, processed_labels) ready for model training
        
    Example:
        >>> train_data, train_labels = preprocess_data('FER2013/train', augment=True)
        >>> print(f"Training data shape: {train_data.shape}")
        >>> print(f"Training labels shape: {train_labels.shape}")
    """
    # Step 1: Load the Data
    data, labels = load_data(dataset_path)

    # Step 2: Extract Labels
    labels = extract_labels(labels)

    # Step 3: Resize Images
    data = resize_images(data, target_size)

    # Step 4: Normalize Pixel Values
    data = normalize_images(data)

    # Step 5: Data Augmentation (optional)
    if augment:
        augmented_data = augment_data(data)
        data = np.concatenate((data, augmented_data), axis=0)
        labels = np.concatenate((labels, labels), axis=0)

    return data, labels

def get_data_loaders(batch_size=32, target_size=(64, 64), augment_train=True):
    """
    Create data loaders for training and testing with proper preprocessing.
    
    Args:
        batch_size (int): Batch size for data loading
        target_size (tuple): Target image dimensions
        augment_train (bool): Whether to augment training data
        
    Returns:
        tuple: (train_data, train_labels, test_data, test_labels) with one-hot encoding
    """
    # Get absolute paths
    train_path = os.path.abspath(TRAIN_PATH)
    test_path = os.path.abspath(TEST_PATH)
    
    # Preprocess data
    train_data, train_labels = preprocess_data(train_path, augment=augment_train, target_size=target_size)
    test_data, test_labels = preprocess_data(test_path, augment=False, target_size=target_size)
    
    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels, num_classes=len(EMOTION_LABELS))
    test_labels = to_categorical(test_labels, num_classes=len(EMOTION_LABELS))
    
    return train_data, train_labels, test_data, test_labels

# Example usage and demonstration
if __name__ == "__main__":
    print("Emotion Detection Data Pipeline")
    print("=" * 40)
    
    # Load and preprocess training data
    train_data, train_labels, test_data, test_labels = get_data_loaders(
        batch_size=32, 
        target_size=(64, 64), 
        augment_train=True
    )
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"Number of emotion classes: {len(EMOTION_LABELS)}")
    print(f"Emotion labels: {list(EMOTION_LABELS.keys())}")
