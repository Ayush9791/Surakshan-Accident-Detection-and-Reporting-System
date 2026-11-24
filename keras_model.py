
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D,
    Flatten, Dense, Dropout
)


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu",
               padding="same", input_shape=(128, 128, 1)),
        BatchNormalization(),

        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax")
    ])

    return model


class KerasAccidentDetector:
    def __init__(self, weights_path):
        self.model = build_model()
        self.model.load_weights(str(weights_path))

    def predict(self, frame):
        # Match the original model's preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))

        image = resized.astype(np.float32) / 255.0
        image = image[np.newaxis, ..., np.newaxis]

        probabilities = self.model(image, training=False).numpy()[0]

        # Original class order: Accident=0, Not-Accident=1
        accident_probability = float(probabilities[0])

        return accident_probability
