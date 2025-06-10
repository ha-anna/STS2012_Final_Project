import datetime
import os
import pickle
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
from art import *  # Used for creating ASCII art text.


def print_start_message():
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tprint("", decoration="love_music")
    print(f"Starting data preparation at {date}")
    tprint("", decoration="love_music")
    print()
    return date


def print_end_message(date_start):
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print()
    tprint("", decoration="love_music")
    print(f"Finished model training at {date}")
    # Calculates and prints the total time elapsed since the start message.
    print(
        "Time taken for data preparation and training: "
        f"{datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(date_start, '%Y-%m-%d %H:%M:%S')}"
    )
    tprint("", decoration="love_music")
    print()


def print_model_summary(model):
    tprint("", decoration="love_music")
    model.summary()  # Displays a summary of the Keras model's architecture.
    print()
    tprint("", decoration="love_music")
    print()


def create_reduced_dir(reduced_dir, train_dir, selected_classes, images_per_class):
    # If the reduced directory already exists, remove it to ensure a clean start.
    if os.path.exists(reduced_dir):
        shutil.rmtree(reduced_dir)

    os.makedirs(reduced_dir)  # Create the new (empty) reduced directory.

    # Iterate through each selected class to copy a subset of images.
    for cls in selected_classes:
        src_folder = os.path.join(train_dir, cls)
        dst_folder = os.path.join(reduced_dir, cls)
        os.makedirs(
            dst_folder, exist_ok=True
        )  # Create destination subfolder for the class.

        images = [
            f
            for f in os.listdir(src_folder)
            if os.path.isfile(os.path.join(src_folder, f))
        ]
        # Randomly select a specified number of images or all if fewer exist.
        selected_imgs = random.sample(images, min(images_per_class, len(images)))

        # Copy the selected images to the new reduced directory.
        for img in selected_imgs:
            shutil.copy(os.path.join(src_folder, img), os.path.join(dst_folder, img))

        print(f"Copied {len(selected_imgs)} images for class '{cls}'")


def save_class_indices(train_generator, ROOT_DIR):
    model_dir = os.path.join(ROOT_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)

    # Save the class-to-index mapping from the ImageDataGenerator.
    # This is crucial for correctly interpreting model predictions later.
    with open("./model/class_indices.pkl", "wb") as f:
        pickle.dump(train_generator.class_indices, f)


def save_model(model, ROOT_DIR):
    model_dir = os.path.join(ROOT_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)

    model_name = "./model/asl_cnn_model.keras"
    model.save(
        model_name
    )  # Saves the entire Keras model (architecture, weights, optimizer state).
    print(f"Model saved to {model_name}")


def plot_model_loss_accuracy(model):
    plt.figure(figsize=(12, 5))

    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(model.history["loss"], label="Training Loss")
    plt.plot(model.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(model.history["accuracy"], label="Training Accuracy")
    plt.plot(model.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def pickle_history(history, ROOT_DIR):
    model_dir = os.path.join(ROOT_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)

    with open("./model/history.pkl", "wb") as f:
        pickle.dump(history.history, f)
    print("Model training history saved to ./model/history.pkl")


def adjust_contrast(image):
    # Custom preprocessing function to adjust image contrast.
    alpha = np.random.uniform(0.8, 1.2)  # Contrast control (1.0 = original)
    image = image * alpha
    image = np.clip(image, 0, 255)
    return image
