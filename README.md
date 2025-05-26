# ASL Signs Recognition System

This project was developed as a final project for the Deep Learning based Big
Data Processing Lab course during the Spring 2025 semester at Sogang University.
It was built by a team of three students — The Overfitters.

## Project Overview

Our project focuses on recognizing American Sign Language (ASL) signs using a
Convolutional Neural Network (CNN). The goal is to create a lightweight deep
learning model that can accurately classify hand gestures representing different
ASL alphabet letters.

We chose this topic because we wanted to work on something meaningful and
socially relevant. Sign language plays a critical role in accessibility and
communication for the Deaf and hard-of-hearing communities. By exploring ASL
sign detection, we aimed to better understand how deep learning can be applied
to improve inclusivity and bridge communication gaps using computer vision.

## Getting started

### 🔧 Setup

You must first download the
[ASL(American Sign Language) Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)
and place it in `data` folder.

Create and activate a virtual environment:

```
make setup
```

This will:

- Upgrade `pip`
- Install dependencies from `requirements.txt`
- Create a virtual environment in `.venv`
- Activate the virtual environment

**Note:** On macOS/Linux, activation is done via:

```
source .venv/bin/activate
```

### 📦 Install Dependencies (if .venv already exists)

```
make install
```

This installs all required Python packages.

### 🧠 Train the CNN Model

```
make train
```

This runs the training script located at src/cnn/cnn.py. It prepares the data
and trains a CNN model on selected ASL classes.

### 🧑‍🦰 Run Face Recognition

```
make run

```

This runs the face recognition script located at
src/face_recognition/face_recognition.py.

The webcam view can be exited with `q`.

### 🧹 Code Formatting (Linting)

```
make lint
```

This formats your code using black and isort for clean, consistent style.

### 📌 Freeze Dependencies

```
make freeze

```

This updates requirements.txt with the current list of installed packages.

## Authors

- Ha Anna Maria (team leader)
- Valero Vinals Alba
- Elza Dabaeva

## Resources Used

- [ASL(American Sign Language) Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset?resource=download)
- [Python Project Structure](https://docs.python-guide.org/writing/structure/)
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
