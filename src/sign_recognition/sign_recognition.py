import os
import pickle

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# organization of directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))
MODEL_DIR = os.path.join(ROOT_DIR, "model", "asl_cnn_model.keras")
CLASS_INDICES_DIR = os.path.join(ROOT_DIR, "model", "class_indices.pkl")

model = load_model(MODEL_DIR)

IMG_SIZE = 64
with open(CLASS_INDICES_DIR, "rb") as f:
    CLASS_NAMES = pickle.load(f)
LABELS = list(CLASS_NAMES.keys())

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the region where hand is expected
    # TODO: readjust this and the draw box, seems the area is not correct/too big
    # TODO: can we make it possible for the model to detect the hand and draw a box around it?
    roi = frame[100:1000, 100:600]

    # Preprocess ROI like training data
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    label = LABELS[class_idx]

    # Draw box & prediction
    cv2.rectangle(frame, (100, 100), (1000, 600), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"Prediction: {label}",
        (100, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        2,
    )

    # Show
    cv2.imshow("ASL Live", frame)

    # Exit on 'q' key
    if cv2.waitKey(150) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
