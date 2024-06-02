import cv2
import numpy as np
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('emotion-classification-model')

# Define the labels for the 7 emotion classes
labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face region
        face_crop = gray[y:y + h, x:x + w]

        # Resize the cropped face to 48x48
        face_crop = cv2.resize(face_crop, (48, 48))

        # Expand dimensions to match the input shape of the model
        face_crop = np.expand_dims(face_crop, axis=-1)  # Add channel dimension
        face_crop = np.expand_dims(face_crop, axis=0)   # Add batch dimension

        # Ensure the image is in float format
        face_crop = face_crop.astype('float32') / 255.0

        # Make a prediction
        predict = model.predict(face_crop)
        predict_number = np.argmax(predict)
        confidence_score = np.max(predict)

        # Get the predicted label
        predicted_label = labels[predict_number]

        # Print the predicted label and confidence score
        print(f"{predicted_label}: {confidence_score:.2f}")

        # Display the prediction on the frame
        text = f"{predicted_label}: {confidence_score:.2f}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame with the face detection box and prediction
    cv2.imshow('cam', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
