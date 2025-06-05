"""
Real-time Emotion Recognition System

This module provides a complete pipeline for real-time facial emotion recognition
from video streams. It combines state-of-the-art face detection using OpenCV DNN
with deep learning-based emotion classification.

Key Features:
- Real-time video processing with GPU acceleration
- Multi-face detection and emotion classification
- Confidence-based filtering for robust predictions
- Visual probability distribution display
- Support for both video files and webcam input

Technical Implementation:
- Face Detection: OpenCV DNN with Caffe models
- Emotion Classification: PyTorch-based ResNet models
- Image Processing: Real-time frame preprocessing and augmentation
- Performance: Optimized for 30+ FPS on modern hardware

Author: CS55100 Student
Date: 2024

Usage:
    python emotion_recognition.py -i video/input.mp4 --model output/model.pth \
        --prototxt model/deploy.prototxt.txt \
        --caffemodel model/res10_300x300_ssd_iter_140000_fp16.caffemodel
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as nnf
from torchvision import transforms
from torchvision.models import resnet18
from neuraspike import utils

# Emotion class mapping for human-readable output (7 classes)
# The order aligns with the dataset/config mapping:
# 0: angry, 1: disgust, 2: fear, 3: happy, 4: neutral, 5: sad, 6: surprise
EMOTION_DICT = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

class EmotionRecognitionSystem:
    """
    Complete emotion recognition system for real-time video processing.
    
    This class encapsulates the entire pipeline including face detection,
    emotion classification, and visualization. It's designed for both
    performance and ease of use.
    
    Attributes:
        face_detector: OpenCV DNN face detection model
        emotion_classifier: PyTorch emotion classification model
        device: Computing device (CPU/GPU)
        data_transform: Image preprocessing pipeline
        confidence_threshold: Minimum confidence for face detection
    """
    
    def __init__(self, model_path, prototxt_path, caffemodel_path, confidence=0.5):
        """
        Initialize the emotion recognition system.
        
        Args:
            model_path (str): Path to trained emotion classification model
            prototxt_path (str): Path to face detection architecture file
            caffemodel_path (str): Path to face detection weights
            confidence (float): Minimum confidence threshold for face detection
            
        Raises:
            FileNotFoundError: If model files are not found
            RuntimeError: If model loading fails
        """
        self.confidence_threshold = confidence
        
        # Initialize computing device
        self.device = self._setup_device()
        
        # Load face detection model
        self.face_detector = self._load_face_detector(prototxt_path, caffemodel_path)
        
        # Load emotion classification model
        self.emotion_classifier = self._load_emotion_classifier(model_path)
        
        # Setup image preprocessing pipeline
        self.data_transform = self._setup_preprocessing()
        
        print(f"[INFO] System initialized successfully on {self.device}")
        print(f"[INFO] Face detection confidence threshold: {confidence}")
    
    def _setup_device(self):
        """Setup and return the optimal computing device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("[INFO] CUDA GPU detected - using GPU acceleration")
        else:
            device = torch.device("cpu")
            print("[INFO] CUDA not available - using CPU")
        return device
    
    def _load_face_detector(self, prototxt_path, caffemodel_path):
        """
        Load the OpenCV DNN face detection model.
        
        Args:
            prototxt_path (str): Path to model architecture file
            caffemodel_path (str): Path to model weights
            
        Returns:
            cv2.dnn.Net: Loaded face detection network
        """
        try:
            net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            print(f"[INFO] Face detection model loaded from {prototxt_path}")
            return net
        except Exception as e:
            raise RuntimeError(f"Failed to load face detection model: {e}")
    
    def _load_emotion_classifier(self, model_path):
        """
        Load the PyTorch emotion classification model.
        
        Args:
            model_path (str): Path to trained model weights
            
        Returns:
            torch.nn.Module: Loaded emotion classification model
        """
        try:
            # Initialize ResNet18 architecture
            model = resnet18(pretrained=False)
            
            # Modify final layer for emotion classification (7 classes)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, 7)
            
            # Load trained weights
            model_weights = torch.load(model_path, map_location=self.device)
            model.load_state_dict(model_weights)
            
            # Move to device and set evaluation mode
            model.to(self.device)
            model.eval()
            
            print(f"[INFO] Emotion classification model loaded from {model_path}")
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load emotion classification model: {e}")
    
    def _setup_preprocessing(self):
        """
        Setup the image preprocessing pipeline for emotion classification.
        
        Returns:
            transforms.Compose: Preprocessing pipeline
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
            transforms.Resize((48, 48)),                  # Resize to model input size
            transforms.ToTensor()                         # Convert to PyTorch tensor
        ])
    
    def detect_faces(self, frame):
        """
        Detect faces in the input frame using OpenCV DNN.
        
        Args:
            frame (numpy.ndarray): Input frame in RGB format
            
        Returns:
            list: List of face detection results (confidence, bounding box)
        """
        # Prepare frame for DNN input
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, 
            (300, 300)
        )
        
        # Run face detection
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        # Extract valid detections above confidence threshold
        valid_detections = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                # Convert normalized coordinates to pixel coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")
                
                valid_detections.append({
                    'confidence': confidence,
                    'bbox': (start_x, start_y, end_x, end_y)
                })
        
        return valid_detections
    
    def classify_emotion(self, face_roi):
        """
        Classify emotion for a detected face region.
        
        Args:
            face_roi (numpy.ndarray): Face region of interest
            
        Returns:
            tuple: (emotion_label, confidence, probability_distribution)
        """
        # Preprocess face ROI
        face_tensor = self.data_transform(face_roi)
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        
        # Run emotion classification
        with torch.no_grad():
            predictions = self.emotion_classifier(face_tensor)
            probabilities = nnf.softmax(predictions, dim=1)
            
            # Get top prediction
            top_prob, top_class = probabilities.topk(1, dim=1)
            top_prob, top_class = top_prob.item(), top_class.item()
            
            # Get full probability distribution
            prob_distribution = [p.item() for p in probabilities[0]]
            
        return top_class, top_prob, prob_distribution
    
    def create_probability_canvas(self, probabilities):
        """
        Create a visual representation of emotion probabilities.
        
        Args:
            probabilities (list): List of probability values for each emotion
            
        Returns:
            numpy.ndarray: Canvas with probability bars
        """
        canvas = np.zeros((300, 300, 3), dtype="uint8")
        
        for i, (emotion, prob) in enumerate(zip(EMOTION_DICT.values(), probabilities)):
            prob_text = f"{emotion}: {prob * 100:.2f}%"
            width = int(prob * 300)
            
            # Draw probability bar
            cv2.rectangle(canvas, (5, (i * 50) + 5), (width, (i * 50) + 50),
                          (0, 0, 255), -1)
            
            # Add text label
            cv2.putText(canvas, prob_text, (5, (i * 50) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return canvas
    
    def process_frame(self, frame):
        """
        Process a single frame for emotion recognition.
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format
            
        Returns:
            tuple: (processed_frame, probability_canvas) for display
        """
        # Resize frame for processing
        frame = utils.resize_image(frame, width=720, height=720)
        output = frame.copy()
        
        # Convert BGR to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_detections = self.detect_faces(frame_rgb)
        
        # Process each detected face
        for detection in face_detections:
            start_x, start_y, end_x, end_y = detection['bbox']
            
            # Extract face ROI
            face_roi = frame_rgb[start_y:end_y, start_x:end_x]
            
            # Classify emotion
            emotion_class, confidence, prob_dist = self.classify_emotion(face_roi)
            
            # Draw results on output frame
            emotion_label = EMOTION_DICT[emotion_class]
            face_text = f"{emotion_label}: {confidence * 100:.2f}%"
            
            # Draw bounding box
            cv2.rectangle(output, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            
            # Draw emotion label
            y = start_y - 10 if start_y - 10 > 10 else start_y + 10
            cv2.putText(output, face_text, (start_x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 255, 0), 2)
        
        # Create probability visualization
        if face_detections:
            # Use the first detected face for probability display
            first_face = frame_rgb[face_detections[0]['bbox'][1]:face_detections[0]['bbox'][3],
                                   face_detections[0]['bbox'][0]:face_detections[0]['bbox'][2]]
            _, _, prob_dist = self.classify_emotion(first_face)
            probability_canvas = self.create_probability_canvas(prob_dist)
        else:
            probability_canvas = np.zeros((300, 300, 3), dtype="uint8")
        
        return output, probability_canvas
    
    def process_video(self, video_path):
        """
        Process video file or webcam stream for real-time emotion recognition.
        
        Args:
            video_path (str): Path to video file or '0' for webcam
        """
        # Initialize video capture
        if video_path == '0':
            cap = cv2.VideoCapture(0)  # Webcam
            print("[INFO] Using webcam input")
        else:
            cap = cv2.VideoCapture(video_path)  # Video file
            print(f"[INFO] Processing video: {video_path}")
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {video_path}")
        
        print("[INFO] Press 'q' to quit, 's' to save current frame")
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, prob_canvas = self.process_frame(frame)
                
                # Display results
                cv2.imshow("Emotion Recognition", processed_frame)
                cv2.imshow("Emotion Probabilities", prob_canvas)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    # Save current frame
                    cv2.imwrite("emotion_frame.jpg", processed_frame)
                    print("[INFO] Frame saved as 'emotion_frame.jpg'")
                
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Video processing completed")


def main():
    """Main function to run the emotion recognition system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Real-time Emotion Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video file
  python emotion_recognition.py -i video/input.mp4 --model output/model.pth \\
      --prototxt model/deploy.prototxt.txt \\
      --caffemodel model/res10_300x300_ssd_iter_140000_fp16.caffemodel
  
  # Use webcam
  python emotion_recognition.py -i 0 --model output/model.pth \\
      --prototxt model/deploy.prototxt.txt \\
      --caffemodel model/res10_300x300_ssd_iter_140000_fp16.caffemodel
        """
    )
    
    parser.add_argument("-i", "--video", type=str, required=True,
                        help="Path to video file or '0' for webcam")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Path to trained emotion classification model")
    parser.add_argument('-p', '--prototxt', type=str, required=True,
                        help='Path to face detection architecture file')
    parser.add_argument('-c', '--caffemodel', type=str, required=True,
                        help='Path to face detection weights file')
    parser.add_argument("-conf", "--confidence", type=float, default=0.5,
                        help="Minimum confidence for face detection (0.0-1.0)")
    
    args = parser.parse_args()
    
    try:
        # Initialize emotion recognition system
        print("[INFO] Initializing Emotion Recognition System...")
        emotion_system = EmotionRecognitionSystem(
            model_path=args.model,
            prototxt_path=args.prototxt,
            caffemodel_path=args.caffemodel,
            confidence=args.confidence
        )
        
        # Process video
        emotion_system.process_video(args.video)
        
    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())