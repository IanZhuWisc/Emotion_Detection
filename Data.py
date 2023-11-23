import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pkg_resources

print(pkg_resources.get_distribution("numpy").version)
print(pkg_resources.get_distribution("keras").version)
print(pkg_resources.get_distribution("tensorflow").version)
print(pkg_resources.get_distribution("opencv-python").version)


train_path = 'FER2013/train'
absolute_train_path = os.path.abspath(train_path)

test_path = 'FER2013/test'
absolute_test_path = os.path.abspath(test_path)

# Step 1: Load the Data
def load_data(dataset_path):
    data = []
    labels = []

    for emotion_folder in os.listdir(dataset_path):
        emotion_path = os.path.join(dataset_path, emotion_folder)
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            img = cv2.imread(img_path)
            data.append(img)
            labels.append(emotion_folder)

    return np.array(data), np.array(labels)

def extract_labels(labels):
    # Map emotion labels to integers
    label_mapping = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    return np.array([label_mapping[label] for label in labels])

# Step 3: Resize Images
def resize_images(images, target_size=(64, 64)):
    # Resize the images to a consistent size
    return np.array([cv2.resize(img, target_size) for img in images])

# Step 4: Normalize Pixel Values
def normalize_images(images):
    # Normalize pixel values to the range [0, 1]
    return images / 255.0

# Step 5: Data Augmentation
def augment_data(images):
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

# Main preprocessing function
def preprocess_data(dataset_path, augment=False):
    # Step 1: Load the Data
    data, labels = load_data(dataset_path)

    # Step 2: Extract Labels
    labels = extract_labels(labels)

    # Step 3: Resize Images
    data = resize_images(data)

    # Step 4: Normalize Pixel Values
    data = normalize_images(data)

    # # Step 5: Data Augmentation
    if augment:
        augmented_data = augment_data(data)
        data = np.concatenate((data, augmented_data), axis=0)
        labels = np.concatenate((labels, labels), axis=0)

    return data, labels

# Example usage
train_data, train_labels = preprocess_data(absolute_train_path, augment=True)
test_data, test_labels = preprocess_data(absolute_test_path)

# Convert labels to one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print("Train data shape:", train_data.shape)
print("Train labels shape:", train_labels.shape)
print("Test data shape:", test_data.shape)
print("Test labels shape:", test_labels.shape)
