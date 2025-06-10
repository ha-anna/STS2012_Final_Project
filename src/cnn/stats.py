import datetime
import os
import pickle

from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

# --- Directory Organization ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))
MODEL_DIR = os.path.join(ROOT_DIR, "model", "best_model.keras")

model = load_model(MODEL_DIR)

# Load the training history (loss and accuracy values per epoch) saved as a pickle file.
with open("../../model/history.pkl", "rb") as f:
    history = pickle.load(f)

# --- Plotting Training and Validation Metrics ---
plt.plot(history["loss"], label="Training Loss")
plt.plot(history.get("val_loss", []), label="Validation Loss")
plt.plot(history.get("accuracy", []), label="Training Accuracy")
plt.plot(history.get("val_accuracy", []), label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Training and Validation Metrics")
plt.legend()

# --- Saving the Plot ---
date = datetime.datetime.now().strftime("%y%m%d_%I:%M")
filename = f"training_validation_metric_{date}.png"
full_path = os.path.join(ROOT_DIR, "model", "stats", filename)
os.makedirs(os.path.dirname(full_path), exist_ok=True)
plt.savefig(full_path)
plt.show()
