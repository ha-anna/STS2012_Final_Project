import datetime
import json
import os
import random
import shutil

from tensorflow.keras import Input
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/cnn
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))  # project root
data_dir = os.path.join(ROOT_DIR, "data", "ASL_Alphabet_Dataset")
train_dir = os.path.join(data_dir, "asl_alphabet_train")
# test_dir = data_dir + "/asl_alphabet_test"
reduced_dir = os.path.join(data_dir, "reduced_train")

img_size = 64
batch_size = 32

selected_classes = ["A", "B", "E", "I", "L", "N", "S"]
images_per_class = 3000


if os.path.exists(reduced_dir):
    shutil.rmtree(reduced_dir)

os.makedirs(reduced_dir)

for cls in selected_classes:
    src_folder = os.path.join(train_dir, cls)
    dst_folder = os.path.join(reduced_dir, cls)
    os.makedirs(dst_folder, exist_ok=True)

    images = [
        f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))
    ]
    selected_imgs = random.sample(images, min(images_per_class, len(images)))

    for img in selected_imgs:
        shutil.copy(os.path.join(src_folder, img), os.path.join(dst_folder, img))

    print(f"Copied {len(selected_imgs)} images for class '{cls}'")

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,  # 80/20 train/val split
    # rotation_range=10,
    # zoom_range=0.1,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # horizontal_flip=True,
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

with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

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
        Conv2D(32, kernel_size=(3, 3), activation="relu",),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dense(7, activation="softmax"),  # 29 classes
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_generator, validation_data=val_generator, epochs=10)

date = datetime.datetime.now()
formatted_date = date.strftime("%Y-%m-%d_%H-%M-%S")

model_name = "./model/asl_cnn_model_(" + formatted_date + ").keras"
model.save(model_name)


loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")
