import collections
import datetime
import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Input
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, GlobalAveragePooling2D,
                                     MaxPooling2D)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

from .utils import (adjust_contrast, create_reduced_dir, pickle_history,
                    plot_model_loss_accuracy, print_end_message,
                    print_model_summary, print_start_message,
                    save_class_indices, save_model)

# --- Path Configuration ---
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # Gets the directory of the current script.
ROOT_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "../../")
)  # Navigates up to the project root.
data_dir = os.path.join(ROOT_DIR, "data", "ASL_Alphabet_Dataset")
train_dir = os.path.join(data_dir, "asl_alphabet_train")
reduced_dir = os.path.join(data_dir, "reduced_train")

# --- Hyperparameters and Configuration ---
img_size = 64
batch_size = 32

selected_classes = ["A", "B", "C", "I", "L", "V"]
images_per_class = 5000

# --- Data Preparation ---
start_date = print_start_message()

# Creates a smaller, balanced subset of the training data for selected classes.
create_reduced_dir(reduced_dir, train_dir, selected_classes, images_per_class)

# Configure data augmentation for training images.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values to [0, 1].
    rotation_range=25,  # Slightly larger rotation
    zoom_range=0.3,  # Larger zoom
    width_shift_range=0.2,  # Larger shifts
    height_shift_range=0.2,
    shear_range=0.2,  # More shear
    brightness_range=(0.5, 1.5),  # Vary brightness → simulates webcam lighting
    channel_shift_range=30.0,  # Even for grayscale → simulates lighting change
    preprocessing_function=adjust_contrast,  # Keep your contrast function
    validation_split=0.2,
)

# Create training data generator from the reduced dataset.
train_generator = train_datagen.flow_from_directory(
    reduced_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="grayscale",
    classes=selected_classes,
    subset="training",
    shuffle=True,
)
print(collections.Counter(train_generator.classes))
save_class_indices(train_generator, ROOT_DIR)

# Validation data generator (only rescale, no augmentation).
val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
val_generator = val_datagen.flow_from_directory(
    reduced_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="grayscale",
    classes=selected_classes,
    subset="validation",
    shuffle=False,
)

# --- CNN Model Architecture ---
model = Sequential(
    [
        Input(shape=(img_size, img_size, 1)),
        # Convolutional Block 1
        Conv2D(
            32, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)
        ),  # L2 regularization helps prevent overfitting
        BatchNormalization(),  # Normalizes activations, aiding stability.
        MaxPooling2D(pool_size=(2, 2)),  # Reduces spatial dimensions.
        # Convolutional Block 2
        Conv2D(
            64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        # Convolutional Block 3
        Conv2D(
            128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        # Convolutional Block 4 (deeper feature extraction)
        Conv2D(
            256, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        GlobalAveragePooling2D(),  # Flattens feature maps by averaging, reducing parameters.
        # Fully Connected Layer
        Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.5),  # Randomly sets 50% of inputs to zero to prevent overfitting.
        # Output Layer
        Dense(
            len(selected_classes), activation="softmax"
        ),  # Softmax for multi-class probability output.
    ]
)

# --- Model Compilation ---
optimizer = AdamW(learning_rate=0.0005, weight_decay=0.0001)
model.compile(
    optimizer=optimizer,
    loss=CategoricalCrossentropy(
        label_smoothing=0.1
    ),  # Crossentropy with label smoothing for better generalization.
    metrics=["accuracy"],
)

early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)  # Stops training if validation loss doesn't improve.
checkpoint = ModelCheckpoint(
    "../../model/best_model.keras", save_best_only=True, monitor="val_loss"
)  # Saves the best model based on validation loss.
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, verbose=1
)  # Reduces learning rate when validation loss plateaus.

# --- Model Training ---
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[early_stop, checkpoint, reduce_lr],
)

# --- Post-Training Operations ---
print_end_message(start_date)
pickle_history(history, ROOT_DIR)
plot_model_loss_accuracy(history)
save_model(model, ROOT_DIR)
print_model_summary(model)

# --- Model Evaluation ---
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# --- Confusion Matrix and Classification Report ---
val_generator.reset()  # Resets the generator to ensure consistent prediction order.
preds = model.predict(val_generator)
y_pred = np.argmax(preds, axis=1)  # Converts probabilities to class labels.
y_true = val_generator.classes
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=selected_classes))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=selected_classes,
    yticklabels=selected_classes,
    cmap="Blues",
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
date = datetime.datetime.now().strftime("%y%m%d_%I:%M")
filename = f"confusion_matrix_{date}.png"
full_path = os.path.join(ROOT_DIR, "model", "stats", filename)
os.makedirs(os.path.dirname(full_path), exist_ok=True)
plt.savefig(full_path)
plt.show()
