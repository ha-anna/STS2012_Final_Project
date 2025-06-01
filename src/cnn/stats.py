import os
import pickle

from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

# organization of directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))
MODEL_DIR = os.path.join(ROOT_DIR, "model", "asl_cnn_model.keras")

model = load_model(MODEL_DIR)

with open("../../model/history.pkl", "rb") as f:
    history = pickle.load(f)

# Plotting
plt.plot(history["loss"], label="Training Loss")
plt.plot(history.get("val_loss", []), label="Validation Loss")
plt.plot(history.get("accuracy", []), label="Training Accuracy")
plt.plot(history.get("val_accuracy", []), label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Training and Validation Metrics")
plt.legend()
plt.show()
