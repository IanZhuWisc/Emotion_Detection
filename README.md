# Emotion Detection System

A comprehensive deep learning system for real-time facial emotion recognition using convolutional neural networks (CNNs) and transfer learning approaches.

## 🎯 Project Overview

This project implements a state-of-the-art emotion detection system capable of recognizing 7 distinct emotions from facial expressions:
- 😠 Angry
- 🤢 Disgust  
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprised

The system leverages the FER2013 dataset and implements multiple CNN architectures including ResNet variants for robust emotion classification.

## 🏗️ Architecture & Technical Implementation

### Core Components

1. **Data Processing Pipeline** (`Data.py`)
   - Automated image loading and preprocessing
   - Data augmentation techniques (rotation, scaling, flipping)
   - Label encoding and one-hot conversion
   - Image normalization and resizing

2. **Neural Network Architectures**
   - **Custom CNN** (`Emotion_detection/neuraspike/emotionNet.py`): Lightweight architecture with ELU activation and dropout
   - **ResNet Variants**: Transfer learning with pre-trained ResNet models
   - **VGG**: Alternative architecture for comparison studies

3. **Real-time Inference** (`Emotion_detection/emotion_recognition.py`)
   - Face detection using OpenCV DNN
   - Real-time video processing
   - GPU acceleration support
   - Confidence-based filtering

### Model Training & Evaluation

The project includes comprehensive training notebooks demonstrating:
- **Transfer Learning**: Leveraging pre-trained ImageNet weights
- **Data Augmentation**: Improving model robustness
- **Hyperparameter Optimization**: Learning rate scheduling, optimizer selection
- **Performance Comparison**: Multi-architecture benchmarking

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
PyTorch 1.9+
OpenCV 4.8+
NumPy 1.24+
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Emotion_Detection
```

2. Install dependencies:
```bash
pip install torch torchvision opencv-python numpy
```

### Usage

#### Real-time Video Processing

```bash
cd Emotion_detection
python emotion_recognition.py \
    -i video/input_video.mp4 \
    --model output/model_resNet_30.pth \
    --prototxt model/deploy.prototxt.txt \
    --caffemodel model/res10_300x300_ssd_iter_140000_fp16.caffemodel
```

#### Model Training

```bash
# Open Jupyter notebooks for training different architectures
jupyter notebook Model_training_resNet-30_comparison.ipynb
```

## 📊 Performance & Results

The system achieves competitive performance on the FER2013 dataset:
- **ResNet-30**: Best performing architecture with optimized training
- **Real-time Processing**: 30+ FPS on modern hardware
- **Robust Detection**: Handles various lighting conditions and face angles

## 🔬 Technical Highlights

### Advanced Features
- **Multi-scale Face Detection**: Robust face localization across different image sizes
- **Ensemble Methods**: Combines multiple model predictions for improved accuracy
- **GPU Optimization**: CUDA support for accelerated inference
- **Data Pipeline**: Efficient batch processing and memory management

### Research Contributions
- Comparative analysis of CNN architectures for emotion recognition
- Transfer learning strategies for small emotion datasets
- Real-time deployment optimization techniques

## 📁 Project Structure

```
Emotion_Detection/
├── Data.py                          # Data preprocessing pipeline
├── Emotion_detection/               # Core emotion recognition system
│   ├── emotion_recognition.py      # Real-time video processing
│   ├── neuraspike/                 # Custom neural network implementations
│   ├── model/                      # Pre-trained face detection models
│   └── output/                     # Trained emotion classification models
├── FER2013/                        # Dataset (train/test splits)
├── Model_training_*.ipynb          # Comprehensive training notebooks
└── README.md                       # This file
```

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- **Deep Learning**: CNN architectures, transfer learning, data augmentation
- **Computer Vision**: Face detection, image processing, real-time video analysis
- **Software Engineering**: Modular code design, performance optimization
- **Machine Learning**: Model evaluation, hyperparameter tuning, dataset management

## 🔮 Future Enhancements

- [ ] Real-time webcam integration
- [ ] Mobile deployment optimization
- [ ] Multi-person emotion detection
- [ ] Temporal emotion analysis
- [ ] API development for cloud deployment

## 📚 References

- FER2013 Dataset: [Paper](https://arxiv.org/abs/1608.01041)
- ResNet Architecture: [Paper](https://arxiv.org/abs/1512.03385)
- OpenCV DNN: [Documentation](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)

## 👨‍💻 Author

Developed as part of CS55100 coursework, demonstrating advanced machine learning and computer vision skills.

---

*This project showcases practical implementation of state-of-the-art deep learning techniques for real-world computer vision applications.*
# Emotion Detection through Face Recognition

## Team 23
- Yanzhang Zhu (zhu.yanzh@northeastern.edu)
- Nishanth Chockalingam Veerapandian (chockalingamveerap.n@northeastern.edu)
- Sai Nithish Mahadeva Rao (mahadevarao.s@northeastern.edu)
- Varun Vikram Sha (vikramsha.v@northeastern.edu)

## Overview
This repository contains the code and documentation for our project on Emotion Detection through Face Recognition. We explored various deep learning models, including custom Convolutional Neural Network (CNN), ResNet18, ResNet50, and VGG, to detect human emotions from facial expressions. Our primary focus was on the FER-2013 dataset, and we aimed to contribute to the iterative development of emotion recognition technology.

3. Main.py: do training and testing

## Project Structure
- `Cnn.py`: Python script for createing the cnn which we want to use and may make change on.
- `Data.py`: Python script for data loading and preprocessing.
- `Main.py`: Python script for loading data and basic files.
- `Model_training.ipynb`: Jupyter Notebook for Resnet18 model training.
- `Model_training_CNN.ipynb`: Jupyter Notebook for CNN model training.
- `Model_training_face.ipynb`: Jupyter Notebook to fine-tune ResNet model using transfer learning with a pre-trained InceptionResnetV1.
- `Model_training_resNet-10_adam.ipynb`: Jupyter Notebook for training ResNet with 10 epochs using Adam optimizer.
- `Model_training_resNet-30.ipynb`: Jupyter Notebook for training ResNet with 30 epochs.
- `Model_training_resNet-30_comparison.ipynb`: Jupyter Notebook for comparing ResNet models trained for 30 epochs.
- `Model_training_resNet.ipynb`: Jupyter Notebook for ResNet model training.
- `Model_training_vgg.ipynb`: Jupyter Notebook for VGG model training.
- `model_resNet_30.pth`: PyTorch model checkpoint file for the ResNet model after 30 epochs.

## Dependencies Used
Python: 3.8.10
numpy: 1.24.4
keeras: 2.10.0
tensorflow: 2.10.1
opencv: 4.8.1.78

## Results
Check the experiment results and discussions in the `Experiments/Results` section of our [project report](https://docs.google.com/document/d/1a2VspnAUc46IqVgEIwtVgn97a5JOlv4zhX-AIEeksZo/edit?usp=sharing).

## Getting Started
1. Clone the repository: `git clone https://github.com/your-username/Emotion_Detection.git`
2. Navigate to the project directory: `cd Emotion_Detection`
3. Set up your environment and install dependencies as specified in the project documentation.

## Usage

Apply the model to the video(Learned from https://neuraspike.com/blog/realtime-emotion-detection-system-pytorch-opencv/)

cd Video

python3 emotion_recognition.py -i video/Sai.mov --model output/model_resNet_10.pth  --prototxt model/deploy.prototxt.txt  --caffemodel model/res10_300x300_ssd_iter_140000_fp16.caffemodel


