import os
import pickle
from os.path import isfile, join

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- Directory Organization ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
CLASS_INDICES_DIR = os.path.join(ROOT_DIR, "model", "class_indices.pkl")

# --- Model Selection ---
# Lists all .keras model files available in the MODEL_DIR.
model_files = [
    f
    for f in os.listdir(MODEL_DIR)
    if f.endswith(".keras") and isfile(join(MODEL_DIR, f))
]

print("Choose by typing the model name:")
plural = True if len(model_files) > 1 else False
print(f"Right now, the available model{'s are' if plural else ' is'}:")
for i, file in enumerate(model_files):
    print(f"{i + 1}. {file}")

# Prompts the user to select a model by its full name.
name = input("Type the full model name (ex: model.keras): ")
print(f"Running webcam with {name}")

# Checks if the entered model name exists; exits if not found.
if name not in model_files:
    print(f"Model {name} not found in {MODEL_DIR}. Please check the model name.")
    exit(1)

CURRENT_MODEL_DIR = os.path.join(ROOT_DIR, "model", name)

model = load_model(CURRENT_MODEL_DIR)

# --- Configuration ---
IMG_SIZE = 64

# Loads the class-to-index mapping, which is essential to convert model predictions (indices) back to human-readable labels.
with open(CLASS_INDICES_DIR, "rb") as f:
    CLASS_NAMES = pickle.load(f)
LABELS = list(CLASS_NAMES.keys())

# --- Webcam Initialization ---
cap = cv2.VideoCapture(0)

# --- Main Loop for Real-time Prediction ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Defines a Region Of Interest (ROI) where the hand sign is expected.
    roi = frame[100:600, 100:600]

    # Preprocesses the ROI image to match the model's expected input format.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # Performs prediction using the loaded model.
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    label = LABELS[class_idx]

    # Draws a green rectangle around the ROI on the original frame.
    cv2.rectangle(frame, (100, 100), (599, 599), (0, 255, 0), 2)

    # Puts the predicted label text on the frame.
    cv2.putText(
        frame,
        f"Prediction: {label}",
        (100, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    cv2.imshow("ASL Sign Detection Live", frame)

    # Breaks the loop if 'q' key is pressed, allowing the user to quit.
    if cv2.waitKey(150) & 0xFF == ord("q"):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
