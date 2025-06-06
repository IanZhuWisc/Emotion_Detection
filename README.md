# Emotion Detection System

A deep learning system for real-time facial emotion recognition using CNNs and transfer learning. The project targets 7 emotions on FER2013 and includes a production-style video demo, training notebooks, and clear documentation for both technical reviewers and hiring managers.

## 🎯 Project Overview

Recognizes the following emotions:
- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

The runtime demo uses OpenCV DNN for face detection and a PyTorch ResNet-based classifier for emotions.

## 🏗️ Architecture

- **Data pipeline** (`Data.py`): image loading, preprocessing, augmentation (standalone; used for experiments)
- **Models** (`Video/neuraspike/emotionNet.py`, ResNet backbones): custom CNN and transfer learning
- **Real-time demo** (`Video/emotion_recognition.py`): face detection + classification at 30+ FPS

## 🚀 Quick Start

### Prerequisites
- Python 3.8+

### Install
```bash
git clone <repository-url>
cd Emotion_Detection
pip install -r requirements.txt
```

### Run the demo
```bash
cd Video
python emotion_recognition.py -i 0 \
  --model output/model_resNet_30.pth \
  --prototxt model/deploy.prototxt.txt \
  --caffemodel model/res10_300x300_ssd_iter_140000_fp16.caffemodel
```
Replace `-i 0` with a video file path to process a file (e.g., `-i Sai.mov`).

## 📊 Results

- Real-time: 30+ FPS on a modern GPU
- Robust to lighting and moderate pose variations
- Trained and compared multiple backbones (CNN, ResNet, VGG) in provided notebooks

## 🔬 Technical Highlights

- OpenCV DNN SSD face detector (ResNet-10) for accurate, fast face localization
- PyTorch ResNet classifier with custom final layer for 7-class prediction
- GPU acceleration with graceful CPU fallback
- Clear separation of concerns: data, model, inference, and configuration

## 📁 Project Structure
```
Emotion_Detection/
├── Data.py                      # Standalone data pipeline (experiments)
├── FER2013/                     # Dataset (train/test splits)
├── Video/                       # Main runtime demo
│   ├── emotion_recognition.py   # Real-time inference entry point
│   ├── model/                   # Face detection (prototxt + caffemodel)
│   ├── neuraspike/              # Config, EmotionNet, utils
│   └── output/                  # Trained emotion classification weights (.pth)
├── Model_training_*.ipynb       # Training notebooks (ResNet, VGG, CNN)
├── PROJECT_DOCUMENTATION.md     # Technical deep-dive
├── PROJECT_STRUCTURE.md         # Codebase organization
├── QUICK_START.md               # 5-minute setup
└── README.md                    # This file
```

## 🎓 Skills Demonstrated

- Deep Learning: CNNs, transfer learning, augmentation, scheduling, early stopping
- Computer Vision: face detection, preprocessing, real-time video analytics
- Software Engineering: modular design, configuration, documentation

## 🔮 Roadmap

- [ ] Multi-person tracking and temporal smoothing
- [ ] Webcam-first UX and packaged binary
- [ ] REST/gRPC API for server inference
- [ ] Mobile/edge deployment optimization

## 📚 References
- FER2013: `https://arxiv.org/abs/1608.01041`
- ResNet: `https://arxiv.org/abs/1512.03385`
- OpenCV DNN: `https://docs.opencv.org/4.x/d6/d0f/group__dnn.html`

## 👥 Team
- Yanzhang Zhu
- Nishanth Chockalingam Veerapandian
- Sai Nithish Mahadeva Rao
- Varun Vikram Sha

Developed as part of CS55100 coursework.
