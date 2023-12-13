# Emotion_Detection

1. Data.py: Deal with data loading and preprocessing
2. Cnn.py: used to create the cnn which we want to use and may make change on
3. Main.py: do training and testing

Versions used:
Python: 3.8.10
numpy: 1.24.4
keeras: 2.10.0
tensorflow: 2.10.1
opencv: 4.8.1.78


Apply the model to the video(Learned from https://neuraspike.com/blog/realtime-emotion-detection-system-pytorch-opencv/)

Command:

cd Video

python3 emotion_recognition.py -i video/Sai.mov --model output/model_resNet_10.pth  --prototxt model/deploy.prototxt.txt  --caffemodel model/res10_300x300_ssd_iter_140000_fp16.caffemodel

