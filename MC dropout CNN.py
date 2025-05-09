import numpy as np
import tifffile
import scipy.io as scio
import csv
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Hyperparameter setting
DATA_PATH = 'D:/xx'  # Define path
WINDOW_SIZE = 39
NUM_CLASSES = 8
LEARNING_RATE = 1e-5
BATCH_SIZE = 512
EPOCHS = 200
MC_SAMPLES = 100  # Monte Carlo sampling time


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
    return (np.array(train_samples, 'float32'),
            np.array(val_samples, 'float32'),
            np.array(train_labels),
            np.array(val_labels),
            raw_data.shape[2])

# MC Dropout CNN network architecture
def build_model(input_shape):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

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

# Input testing data
def evaluate_model(model):
    raw_data = tifffile.imread(f'{DATA_PATH}image_test.tif')
    H = raw_data.shape[0] - WINDOW_SIZE + 1
    W = raw_data.shape[1] - WINDOW_SIZE + 1

    mc_probs = np.zeros((H, W, MC_SAMPLES, NUM_CLASSES))
    mc_pred_classes = np.zeros((H, W, MC_SAMPLES), dtype=int)

    for i in range(H):
        row_patches = [
            raw_data[i:i + WINDOW_SIZE, j:j + WINDOW_SIZE, :]
            for j in range(W)
        ]
        batch_patches = np.array(row_patches, dtype='float32')

        # MC sampling
        for mc_iter in range(MC_SAMPLES):
            pred = model(batch_patches, training=True)  # Keep Dropout activate
            mc_probs[i, :, mc_iter, :] = pred.numpy()
            mc_pred_classes[i, :, mc_iter] = np.argmax(pred, axis=1)

# Save multiple predictions as npy and mat
    np.save(f'{DATA_PATH}mc_predictions.npy', mc_pred_classes)
    scio.savemat(f'{DATA_PATH}mc_predictions.mat', {
        'mc_pred': mc_pred_classes
    })

# Entropy and variance
    uncertainties = np.zeros((H, W, 2))
    for i in range(H):
        for j in range(W):
            pixel_probs = mc_probs[i, j, :, :]
            mean_probs = np.mean(pixel_probs, axis=0)
            std_dev = np.std(mean_probs)
            entropy_val = entropy(mean_probs)
            uncertainties[i, j, 0] = std_dev
            uncertainties[i, j, 1] = entropy_val
    return mc_probs, uncertainties

# Save training loss curves
if __name__ == '__main__':
    with open(HISTORY_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_id', 'epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
    with open(MC_HISTORY_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mc_sample_id', 'epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])

    train_data, val_data, train_labels, val_labels, num_channels = load_data()
    input_shape = (WINDOW_SIZE, WINDOW_SIZE, num_channels)

# Model training
    model = build_model(input_shape)
    history = model.fit(train_data, train_labels,
                        validation_data=(val_data, val_labels),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE)
    save_training_history(history, model_id=1)

# Uncertainty evaluation
    mc_probs, uncertainties = evaluate_model(model)

# Model evaluation
    labels = tifffile.imread(f'{DATA_PATH}class_test.tif')
    labels = labels[WINDOW_SIZE // 2: -(WINDOW_SIZE // 2),
             WINDOW_SIZE // 2: -(WINDOW_SIZE // 2)]
    final_pred = np.argmax(np.mean(mc_probs, axis=2), axis=-1)
    accuracy = np.mean(final_pred.flatten() == labels.flatten() - 1)

