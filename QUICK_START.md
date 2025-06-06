# Quick Start Guide - Emotion Recognition System

## 🚀 Get Started in 5 Minutes

This guide will help you quickly understand and run the emotion recognition system to see it in action.

## 📋 Prerequisites

- **Python 3.8+** installed
- **Git** for cloning the repository
- **Webcam** or video file for testing

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Emotion_Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Pre-trained Models
The project includes pre-trained models in the `Emotion_detection/output/` directory:
- `model_resNet_30.pth` - Best performing model (75% accuracy)
- `model_resNet_10.pth` - Lightweight model (72% accuracy)

## 🎯 Quick Demo

### Option 1: Real-time Webcam
```bash
cd Video
python emotion_recognition.py -i 0 \
    --model output/model_resNet_30.pth \
    --prototxt model/deploy.prototxt.txt \
    --caffemodel model/res10_300x300_ssd_iter_140000_fp16.caffemodel
```

### Option 2: Video File
```bash
cd Video
python emotion_recognition.py -i your_video.mp4 \
    --model output/model_resNet_30.pth \
    --prototxt model/deploy.prototxt.txt \
    --caffemodel model/res10_300x300_ssd_iter_140000_fp16.caffemodel
```

## 🎮 Controls

- **Press 'q'** to quit the application
- **Press 's'** to save the current frame
- The system will automatically detect faces and classify emotions

## 📊 What You'll See

1. **Main Window**: Video with emotion labels and bounding boxes
2. **Probability Window**: Real-time emotion probability distribution
3. **Performance**: 30+ FPS on modern hardware

## 🔬 Understanding the Output

### Emotion Classes
- 😠 **Angry**: Aggressive or hostile expressions
- 🤢 **Disgust**: Repulsed or revolted expressions  
- 😨 **Fear**: Scared or anxious expressions
- 😊 **Happy**: Joyful or pleased expressions
- 😐 **Neutral**: No particular emotion
- 😢 **Sad**: Sorrowful or dejected expressions
- 😲 **Surprised**: Astonished or shocked expressions

### Technical Features
- **Real-time Processing**: Live video analysis
- **Multi-face Detection**: Handles multiple people
- **Confidence Scoring**: Shows prediction certainty
- **GPU Acceleration**: Automatic CUDA detection

## 📚 Explore the Code

### Key Files to Review
- `README.md` - Project overview and documentation
- `PROJECT_DOCUMENTATION.md` - Technical deep-dive
- `Data.py` - Data processing pipeline
- `Emotion_detection/emotion_recognition.py` - Main inference system
- `Emotion_detection/neuraspike/emotionNet.py` - Custom CNN architecture

### Training Notebooks
- `Model_training_resNet-30_comparison.ipynb` - Comprehensive training examples
- `Model_training_CNN.ipynb` - Custom CNN training
- `Model_training_vgg.ipynb` - VGG architecture training

## 🎯 For Recruiters & Hiring Managers

### What This Project Demonstrates

#### Technical Skills
- **Deep Learning**: CNN architectures, transfer learning, optimization
- **Computer Vision**: Face detection, real-time processing, GPU acceleration
- **Software Engineering**: Clean code, documentation, modular design
- **Data Science**: Dataset management, preprocessing, augmentation

#### Professional Qualities
- **Problem Solving**: Real-world computer vision challenges
- **Performance Focus**: 30+ FPS real-time processing
- **Documentation**: Comprehensive technical writing
- **Code Quality**: Production-ready, maintainable code

### Performance Metrics
- **Accuracy**: 75% on FER2013 dataset
- **Speed**: 30+ FPS real-time processing
- **Robustness**: Handles various lighting and face angles
- **Scalability**: Modular architecture for easy extension

## 🚨 Troubleshooting

### Common Issues

#### "CUDA not available"
- The system will automatically use CPU if GPU is not available
- Performance will be slower but functionality remains

#### "Model file not found"
- Ensure you're in the correct directory
- Check that model files exist in `output/` folder

#### "Video file not found"
- Verify the video path is correct
- Supported formats: MP4, AVI, MOV

### Performance Tips
- **GPU**: Use CUDA-enabled GPU for best performance
- **Resolution**: Lower video resolution for faster processing
- **Batch Size**: Adjust based on available memory

## 🔗 Next Steps

1. **Run the demo** to see the system in action
2. **Review the code** to understand the implementation
3. **Check the notebooks** for training examples
4. **Read the documentation** for technical details

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Review the comprehensive documentation
3. Examine the code comments and docstrings

---

**This project showcases advanced machine learning skills and production-ready software engineering. The code is clean, well-documented, and demonstrates real-world problem-solving capabilities.**
