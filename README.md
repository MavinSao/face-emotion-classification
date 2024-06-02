# Face Emotion Classification

## Description
This project aims to classify human emotions from facial expressions using a deep learning model. The model is trained to recognize seven different emotions: angry, disgust, fear, happy, neutral, sad, and surprise. Using a convolutional neural network (CNN) architecture, the model processes images of faces and predicts the corresponding emotion.

The project includes real-time emotion detection from webcam video feeds. A Haar Cascade classifier is used for face detection, and the detected face region is then passed to the trained model for emotion classification. The predicted emotion and confidence score are displayed on the video feed.

## Dataset
The model is trained on the FER-2013 dataset, which is available on Kaggle. The FER-2013 dataset consists of grayscale images of faces, each labeled with one of the seven emotion classes. The dataset includes 28,709 training images and 7,178 test images, each of size 48x48 pixels.

- **Dataset Link**: [FER-2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

## Features
- **Emotion Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Model Architecture**: Convolutional Neural Network (CNN) with Batch Normalization and Dropout layers for regularization
- **Real-time Emotion Detection**: Uses webcam feed to detect faces and classify emotions
- **Performance Metrics**: Displays accuracy and loss graphs after training

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-emotion-classification.git
   cd face-emotion-classification
   
2. Clone the repository:
   ```bash
   pip install -r requirements.txt


## Usage
1. Train the Model:
Run the main training script to train the model on the FER-2013 dataset.
   ```bash
   python main.py

2. Real-time Emotion Detection: 
Use the play script to start the webcam feed and classify emotions in real-time.
   ```bash
   python main.py
   
## Results
After training, the model's performance can be evaluated using accuracy and loss graphs. The real-time emotion detection displays the predicted emotion and confidence score directly on the webcam feed, providing an interactive way to test the model.

Acknowledgements
The FER-2013 dataset used in this project is publicly available on Kaggle, created by MSambare.
The Haar Cascade classifier for face detection is provided by OpenCV.
This project serves as a comprehensive example of using deep learning for image classification tasks, particularly focusing on emotion recognition from facial expressions.
