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
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy

from .utils import (create_reduced_dir, pickle_history,
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

selected_classes = ["A", "B", "C", "D"]
images_per_class = 5000

start_date = print_start_message()
create_reduced_dir(reduced_dir, train_dir, selected_classes, images_per_class)

# datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     validation_split=0.2,
#     rotation_range=20,
#     zoom_range=0.2,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
# )


# train_generator = datagen.flow_from_directory(
#     reduced_dir,
#     target_size=(img_size, img_size),
#     batch_size=batch_size,
#     class_mode="categorical",
#     color_mode="grayscale",
#     classes=selected_classes,
#     subset="training",
# )

# save_class_indices(train_generator)

# val_generator = datagen.flow_from_directory(
#     reduced_dir,
#     target_size=(img_size, img_size),
#     batch_size=batch_size,
#     class_mode="categorical",
#     color_mode="grayscale",
#     classes=selected_classes,
#     subset="validation",
# )

# model = Sequential(
#     [
#         Input(shape=(img_size, img_size, 1)),
#         Conv2D(32, (3, 3), activation="relu", padding="same"),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Conv2D(64, (3, 3), activation="relu", padding="same"),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Conv2D(128, (3, 3), activation="relu", padding="same"),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Flatten(),
#         Dropout(0.5),
#         Dense(128, activation="relu"),
#         Dense(len(selected_classes), activation="softmax"),
#     ]
# )

# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
# checkpoint = ModelCheckpoint(
#     "best_model.keras", save_best_only=True, monitor="val_loss"
# )

# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=20,
#     callbacks=[early_stop, checkpoint],
# )


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    validation_split=0.2,
)

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

val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

print(collections.Counter(train_generator.classes))

save_class_indices(train_generator, ROOT_DIR)

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

model = Sequential(
    [
        Input(shape=(img_size, img_size, 1)),
        Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(len(selected_classes), activation="softmax", kernel_regularizer=l2(0.001)),
    ]
)

model.compile(optimizer="adam", loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    "best_model.keras", save_best_only=True, monitor="val_loss"
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[early_stop, checkpoint, reduce_lr],
)

pickle_history(history, ROOT_DIR)

print(train_generator.class_indices)
print(val_generator.class_indices)

plot_model_loss_accuracy(history)

save_model(model, ROOT_DIR)

print_end_message(start_date)

print_model_summary(model)

loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# confusion matrix
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
