from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model('mask_detection_model.h5')

# Real-time emotion detection
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray_frame)

    for (x, y, w, h) in faces:
        roi = gray_frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))  # Ensure the shape is (1, 48, 48, 1)

        emotion_prediction = model.predict(roi)
        emotion_label = np.argmax(emotion_prediction)

        # Add emotion label to the video frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, str(emotion_label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()