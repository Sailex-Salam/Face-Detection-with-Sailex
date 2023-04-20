import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model('CatVsDog.model')
labels = ['cat', 'dog']
def detect_cat_dog(frame):
    frame = cv2.resize(frame, (200,200))
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)
    prediction = model.predict(frame)
    class_idx = np.argmax(prediction[0])
    label = labels[class_idx]
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def capture_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = detect_cat_dog(frame)
        cv2.imshow('Cat vs Dog Detection', result[0])
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
capture_webcam()
