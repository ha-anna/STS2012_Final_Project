import os

import matplotlib.pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import (create_reduced_dir, print_end_message, print_model_summary,
                   print_start_message, save_class_indices, save_model)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/cnn
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))  # project root
data_dir = os.path.join(ROOT_DIR, "data", "ASL_Alphabet_Dataset")
train_dir = os.path.join(data_dir, "asl_alphabet_train")
reduced_dir = os.path.join(data_dir, "reduced_train")

img_size = 64
batch_size = 40

selected_classes = ["A", "B", "E", "I", "L", "N", "S"]
images_per_class = 3000

start_date = print_start_message()
create_reduced_dir(reduced_dir, train_dir, selected_classes, images_per_class)

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,  # 80/20 train/val split
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

train_generator = datagen.flow_from_directory(
    reduced_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="grayscale",
    classes=selected_classes,
    subset="training",
)

save_class_indices(train_generator)

val_generator = datagen.flow_from_directory(
    reduced_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="grayscale",
    classes=selected_classes,
    subset="validation",
)

model = Sequential(
    [
        Input(shape=(img_size, img_size, 1)),
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
        ),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dense(
            7, activation="softmax"
        ),  # dense value should match number of selected_classes
    ]
)


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_generator, validation_data=val_generator, epochs=15)

# Plot training & validation loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot training & validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

save_model(model, ROOT_DIR)

print_end_message(start_date)

print_model_summary(model)

loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")
