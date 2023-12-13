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
commands


