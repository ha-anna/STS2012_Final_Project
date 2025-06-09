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
                                     Dropout, Flatten, GlobalAveragePooling2D,
                                     MaxPooling2D)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

from .utils import (adjust_contrast, create_reduced_dir, pickle_history,
                    plot_model_loss_accuracy, print_end_message,
                    print_model_summary, print_start_message,
                    save_class_indices, save_model)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/cnn
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))  # project root
data_dir = os.path.join(ROOT_DIR, "data", "ASL_Alphabet_Dataset")
train_dir = os.path.join(data_dir, "asl_alphabet_train")
reduced_dir = os.path.join(data_dir, "reduced_train")

img_size = 64
batch_size = 32

selected_classes = ["A", "B", "C", "I", "L", "V"]
images_per_class = 5000

start_date = print_start_message()
create_reduced_dir(reduced_dir, train_dir, selected_classes, images_per_class)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalization
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

# Create training data generator from reduced dataset
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

# Build CNN model architecture
model = Sequential(
    [
        Input(shape=(img_size, img_size, 1)),

        # 1st Conv Block
        Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # 2nd Conv Block
        Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # 3rd Conv Block
        Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # 4th Conv Block (NEW - deeper feature extractor)
        Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Global feature pooling
        GlobalAveragePooling2D(),

        # Fully Connected Layer with more capacity
        Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.5),

        # Output Layer - 6 classes for ASL
        Dense(6, activation="softmax"),
    ]
)

optimizer = Adam(learning_rate=0.0005)
optimizer2 = AdamW(
    learning_rate=0.0005, weight_decay=0.0001
)
model.compile(
    optimizer=optimizer2,
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

# Define callbacks: early stopping, checkpoint saving, and learning rate reduction
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    "../../model/best_model.keras", save_best_only=True, monitor="val_loss"
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[early_stop, checkpoint, reduce_lr],
)

print_end_message(start_date)

pickle_history(history, ROOT_DIR)

# Plot training & validation loss/accuracy curves
plot_model_loss_accuracy(history)

save_model(model, ROOT_DIR)

# Print model architecture summary
print_model_summary(model)


# Evaluate model on validation set
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Plot and save confusion matrix as a heatmap
val_generator.reset()
preds = model.predict(val_generator)
y_pred = np.argmax(preds, axis=1)
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
