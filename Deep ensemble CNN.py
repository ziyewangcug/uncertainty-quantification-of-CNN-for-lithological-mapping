import os
import csv
import numpy as np
import scipy.io as scio
import tifffile
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf


# Hyperparameter setting
DATA_PATH = 'D:/xx'  # Define path
WINDOW_SIZE = 39
NUM_CLASSES = 8
LEARNING_RATE = 1e-5
BATCH_SIZE = 512
EPOCHS = 200
N_MODELS = 100  # Ensemble numbers


# Load data and ground truth（.tif）
def load_data():
    raw_data = tifffile.imread(f'{DATA_PATH}image_test.tif')
    labels = tifffile.imread(f'{DATA_PATH}class_test.tif')
    train_samples, val_samples = [], []
    train_labels, val_labels = [], []

# Sample preparation using sliding window
    for class_id in range(1, NUM_CLASSES + 1):
        locs = np.argwhere(labels == class_id)
        if len(locs) == 0:
            continue
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
    return (np.array(train_samples, 'float32'),
            np.array(val_samples, 'float32'),
            np.array(train_labels),
            np.array(val_labels),
            raw_data.shape[2])

# Deep ensemble CNN network architecture
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
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Input testing data
def ensemble_predict(models):
    raw_data = tifffile.imread(f'{DATA_PATH}image_test.tif')
    all_preds = []

    for model in models:
        model_pred = []
        for i in range(raw_data.shape[0] - WINDOW_SIZE + 1):
            row_patches = [
                raw_data[i:i + WINDOW_SIZE, j:j + WINDOW_SIZE, :]
                for j in range(raw_data.shape[1] - WINDOW_SIZE + 1)
            ]
            batch_pred = model.predict(np.array(row_patches), verbose=0)
            model_pred.append(np.argmax(batch_pred, axis=1))

        model_pred = np.concatenate(model_pred).reshape(
            (raw_data.shape[0] - WINDOW_SIZE + 1,
             raw_data.shape[1] - WINDOW_SIZE + 1))
        all_preds.append(model_pred)
    result_cube = np.stack(all_preds, axis=-1)
    return result_cube

# Save multiple predictions as npy and mat
def save_results(result_cube):
    np.save(f'{DATA_PATH}prediction_cube.npy', result_cube)
    scio.savemat(f'{DATA_PATH}prediction_cube.mat',
                 {'prediction_cube': result_cube})
    print(f"Results saved to {DATA_PATH}")

# Save training loss curves
def save_training_history(history, model_id):
    with open(HISTORY_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        for epoch in range(len(history.history['loss'])):
            row = [
                model_id,
                epoch + 1,
                history.history['loss'][epoch],
                history.history['accuracy'][epoch],
                history.history['val_loss'][epoch],
                history.history['val_accuracy'][epoch]
            ]
            writer.writerow(row)

if __name__ == '__main__':
    with open(HISTORY_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_id', 'epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])

# Model training
    models = []
    for i in range(N_MODELS):
        train_data, val_data, train_labels, val_labels, num_channels = load_data(seed=i)
        input_shape = (WINDOW_SIZE, WINDOW_SIZE, num_channels)

        model = build_model(input_shape)
        history = model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        models.append(model)
        save_training_history(history, model_id=i+1)
        print(f'Model {i + 1}/{N_MODELS} trained')

    prediction_cube = ensemble_predict(models)
    save_results(prediction_cube)
    print(f"Training history saved to {HISTORY_CSV}")

# Model evaluation
    labels = tifffile.imread(f'{DATA_PATH}class_test.tif')
    labels = labels[WINDOW_SIZE // 2: -(WINDOW_SIZE // 2),
             WINDOW_SIZE // 2: -(WINDOW_SIZE // 2)]
    accuracies = [np.mean(prediction_cube[..., m].flatten() == labels.flatten() - 1)
                  for m in range(N_MODELS)]
    print(f"Individual Accuracies: {accuracies}")
