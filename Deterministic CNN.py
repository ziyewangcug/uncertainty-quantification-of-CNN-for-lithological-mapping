import os
import numpy as np
import csv
import tifffile
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

# Hyperparameter setting
DATA_PATH = 'D:/xx'  # Define path
WINDOW_SIZE = 39
NUM_CLASSES = 8
LEARNING_RATE = 1e-5
BATCH_SIZE = 512
EPOCHS = 200

# Load data and ground truth（.tif）
def load_data():
    raw_data = tifffile.imread(f'{DATA_PATH}image_test.tif')
    labels = tifffile.imread(f'{DATA_PATH}class_test.tif')
    train_samples, val_samples = [], []
    train_labels, val_labels = [], []

# Sample preparation using sliding window
    for class_id in range(1, NUM_CLASSES + 1):
        locs = np.argwhere(labels == class_id)
        sampled = locs[np.random.choice(len(locs), int(len(locs) * 0.1), replace=False)]

        for idx, (x, y) in enumerate(sampled):

            x_start = x - WINDOW_SIZE // 2
            y_start = y - WINDOW_SIZE // 2
            patch = raw_data[x_start:x_start + WINDOW_SIZE,
                    y_start:y_start + WINDOW_SIZE, :]

            if patch.size == WINDOW_SIZE ** 2 * raw_data.shape[2]:
                target = [class_id - 1]
                if idx < int(len(sampled) * 0.8):
                    train_samples.append(patch)
                    train_labels.append(target)
                else:
                    val_samples.append(patch)
                    val_labels.append(target)
# Training data and label
    return (np.asarray(train_samples, 'float32'),
            np.asarray(val_samples, 'float32'),
            np.asarray(train_labels),
            np.asarray(val_labels),
            raw_data.shape[2])

# CNN network architecture
def build_model(input_shape):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Input testing data
def evaluate_model(model):
    raw_data = tifffile.imread(f'{DATA_PATH}image_test.tif')
    predictions = []

# Prediction
    for i in range(raw_data.shape[0] - WINDOW_SIZE + 1):
        row_patches = [
            raw_data[i:i + WINDOW_SIZE, j:j + WINDOW_SIZE, :]
            for j in range(raw_data.shape[1] - WINDOW_SIZE + 1)
        ]
        batch_pred = model.predict(np.array(row_patches), verbose=0)
        predictions.extend(np.argmax(batch_pred, axis=1))
    return np.array(predictions).reshape(
        (raw_data.shape[0] - WINDOW_SIZE + 1,
         raw_data.shape[1] - WINDOW_SIZE + 1))

if __name__ == '__main__':
    train_data, val_data, train_labels, val_labels, num_channels = load_data()
    input_shape = (WINDOW_SIZE, WINDOW_SIZE, num_channels)

# Model training
    model = build_model(input_shape)
    history = model.fit(train_data, train_labels,
                        validation_data=(val_data, val_labels),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE)
# Save training loss curves
    with open('training_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
        for epoch in range(len(history.history['loss'])):
            row = [
                epoch + 1,
                history.history['loss'][epoch],
                history.history['accuracy'][epoch],
                history.history['val_loss'][epoch],
                history.history['val_accuracy'][epoch]
            ]
            writer.writerow(row)

# Prediction visualization
    pred_map = evaluate_model(model)
    tifffile.imwrite(f'{DATA_PATH}pred_map.tif', pred_map.astype(np.uint8))
    plt.imshow(pred_map, cmap='jet')
    plt.show()

# Model evaluation
    labels = tifffile.imread(f'{DATA_PATH}class_test.tif')
    labels = labels[WINDOW_SIZE // 2: -(WINDOW_SIZE // 2),
             WINDOW_SIZE // 2: -(WINDOW_SIZE // 2)]
    accuracy = np.mean(pred_map.flatten() == labels.flatten() - 1)
    print(f'Test Accuracy: {accuracy:.4f}')