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
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

from .utils import (create_reduced_dir, pickle_history,
                    plot_model_loss_accuracy, print_end_message,
                    print_model_summary, print_start_message,
                    save_class_indices, save_model, adjust_contrast)

# Define project and dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/cnn
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))  # project root
data_dir = os.path.join(ROOT_DIR, "data", "ASL_Alphabet_Dataset")
train_dir = os.path.join(data_dir, "asl_alphabet_train")
reduced_dir = os.path.join(data_dir, "reduced_train")

# Set image size and training batch size
img_size = 64
batch_size = 32

# Select specific ASL classes and number of images per class for training
selected_classes = ["A", "B", "C", "I", "K", "L", "V"]
images_per_class = 5000

start_date = print_start_message()
create_reduced_dir(reduced_dir, train_dir, selected_classes, images_per_class)

# Define data augmentation for training images
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

# Print class distribution in the training set
print("Train class distribution:")
print(collections.Counter(train_generator.classes))

# Save class-to-index mapping for future use
save_class_indices(train_generator, ROOT_DIR)

# Create validation data generator (no augmentation, just rescaling)
val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
val_generator = val_datagen.flow_from_directory(
    reduced_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="grayscale",
    classes=selected_classes,
    subset="validation",
    shuffle=False)

print("Validation class distribution:")
print(collections.Counter(val_generator.classes))

# Build CNN model architecture
model = Sequential(
    [
        # Input layer: expects grayscale images of shape (64, 64, 1)
        Input(shape=(img_size, img_size, 1)),
        # 1st Convolutional Block
        # Convolution layer with 32 filters of size 3x3, ReLU activation
        # Padding='same' ensures output has the same dimensions as input
        # L2 regularization helps prevent overfitting by penalizing large weights
        Conv2D(
            32, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)
        ),
        # Batch normalization stabilizes and accelerates training by normalizing activations
        BatchNormalization(),  # TODO: this is good for our model, it helps to stabilize the learning process
        # Downsamples the feature map by a factor of 2
        MaxPooling2D((2, 2)),
        # 2nd Convolutional Block
        Conv2D(
            64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)
        ),
        BatchNormalization(),  # TODO: add dropout after batchnormalization
        MaxPooling2D((2, 2)),
        # 3rd Convolutional Block
        Conv2D(
            128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        # Global average pooling replaces flattening the entire feature map
        # It computes the average of each feature map, drastically reducing parameters
        # and improving generalization, especially in image classification tasks
        GlobalAveragePooling2D(),
        # Dropout layer randomly disables 50% of neurons during training to reduce overfitting
        Dropout(0.5),
        # Fully connected dense layer with 256 units and ReLU activation
        # L2 regularization again helps reduce overfitting
        Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
        # Another Dropout for better generalization
        Dropout(0.5),
        # Output layer: number of neurons = number of selected ASL classes
        # Softmax activation produces probability distribution over class labels
        Dense(
            len(selected_classes), activation="softmax", kernel_regularizer=l2(0.001)
        ),
    ]
)
# TODO; maybe we should delete one block to reduce overfitting


# Compile the model with Adam optimizer and label smoothing for regularization
optimizer = Adam(learning_rate=0.0005)
optimizer2 = AdamW(learning_rate=0.0005, weight_decay=0.0001)  # AdamW optimizer with weight decay
model.compile(
    optimizer=optimizer2,
    loss=CategoricalCrossentropy(),
    metrics=["accuracy"],
)
# TODO: adam learning rate is 0.001 by default, but we can change it if needed, we should reduce it (might help)
# too fast too soon problem, we can try to reduce the learning rate

# Define callbacks: early stopping, checkpoint saving, and learning rate reduction
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    "../../model/best_model.keras", save_best_only=True, monitor="val_loss"
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
# TODO: learning rate seems to be too high, that is why the valdiation loss is crazy

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stop, checkpoint, reduce_lr],
)

print_end_message(start_date)

# Save training history as a pickle file
pickle_history(history, ROOT_DIR)

# Print class indices for training and validation sets
print(train_generator.class_indices)
print(val_generator.class_indices)


# Plot training & validation loss/accuracy curves
plot_model_loss_accuracy(history)

save_model(model, ROOT_DIR)

# Print model architecture summary
print_model_summary(model)

# Evaluate model on validation set
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Generate predictions and compute confusion matrix
val_generator.reset()
preds = model.predict(val_generator)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes

# Print confusion matrix and classification report
print(confusion_matrix(y_true, y_pred))
report = classification_report(y_true, y_pred, target_names=selected_classes, output_dict=True)
per_class_acc = [report[label]['recall'] for label in selected_classes]

plt.figure(figsize=(8,5))
sns.barplot(x=selected_classes, y=per_class_acc)
plt.ylim(0,1)
plt.ylabel('Recall (Sensitivity)')
plt.title('Per-class Recall on Validation Set')
plt.show()


train_filenames = train_generator.filenames
val_filenames = val_generator.filenames

# Find intersection
overlap = set(train_filenames).intersection(val_filenames)
print(f"Number of overlapping images: {len(overlap)}")

# Plot and save confusion matrix as a heatmap
# confusion matrix wont be as good when we make it more valid for real life data
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


# Visualize a few validation images
x_val, y_val = next(val_generator)

plt.figure(figsize=(10, 5))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(x_val[i].squeeze(), cmap="gray")
    plt.title(f"Class: {selected_classes[np.argmax(y_val[i])]}")
    plt.axis("off")
plt.tight_layout()
plt.show()


# Visualize a few validation images
x_val, y_val = next(train_generator)

plt.figure(figsize=(10, 5))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(x_val[i].squeeze(), cmap="gray")
    plt.title(f"Class: {selected_classes[np.argmax(y_val[i])]}")
    plt.axis("off")
plt.tight_layout()
plt.show()


# All pixel intensities from a batch
pixels_train = x_val.ravel()
sns.histplot(pixels_train, bins=50)
plt.title("Validation pixel intensity distribution")
plt.show()

