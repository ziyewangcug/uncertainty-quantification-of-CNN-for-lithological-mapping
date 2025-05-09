import os
import numpy as np
import csv
import scipy.io as scio
import tifffile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer, BatchNormalization, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

# Hyperparameter setting
DATA_PATH = 'D:/xx'  # Define path
WINDOW_SIZE = 39
NUM_CLASSES = 8
LEARNING_RATE = 1e-5
BATCH_SIZE = 512
EPOCHS = 30

# KL Divergence Strength
kl_weight = 1e-8

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
            patch = raw_data[x_start:x_start + WINDOW_SIZE, y_start:y_start + WINDOW_SIZE, :]
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

# Bayes by backprop CNN network architecture
class BayesianConv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='same',
                 activation=None, prior_mu=0.0, prior_sigma=0.1, **kwargs):
        super(BayesianConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.activation = tf.keras.activations.get(activation)
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

    def build(self, input_shape):
        kernel_shape = self.kernel_size + (int(input_shape[-1]), self.filters)
        self.kernel_mu = self.add_weight(name='kernel_mu', shape=kernel_shape,
                                         initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1), trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho', shape=kernel_shape,
                                          initializer=tf.keras.initializers.RandomNormal(mean=-5, stddev=0.1), trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu', shape=(self.filters,),
                                       initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1), trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho', shape=(self.filters,),
                                        initializer=tf.keras.initializers.RandomNormal(mean=-5, stddev=0.1), trainable=True)

    def call(self, inputs):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        bias_sigma = tf.math.softplus(self.bias_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(tf.shape(self.kernel_mu))
        bias = self.bias_mu + bias_sigma * tf.random.normal(tf.shape(self.bias_mu))

        outputs = tf.nn.conv2d(inputs, kernel, strides=(1,) + self.strides + (1,), padding=self.padding)
        outputs = tf.nn.bias_add(outputs, bias)

        if self.activation:
            outputs = self.activation(outputs)

        kl = self.kl_divergence(self.kernel_mu, kernel_sigma, self.prior_mu, self.prior_sigma) + \
             self.kl_divergence(self.bias_mu, bias_sigma, self.prior_mu, self.prior_sigma)
        self.add_loss(kl_weight * kl)
        return outputs

    def kl_divergence(self, mu, sigma, prior_mu, prior_sigma):
        return tf.reduce_sum(
            tf.math.log(prior_sigma / sigma) +
            (sigma ** 2 + (mu - prior_mu) ** 2) / (2.0 * prior_sigma ** 2) - 0.5
        )

class BayesianDense(Layer):
    def __init__(self, units, activation=None, prior_mu=0.0, prior_sigma=0.1):
        super(BayesianDense, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

    def build(self, input_shape):
        self.w_mu = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1), trainable=True)
        self.w_rho = self.add_weight(shape=(input_shape[-1], self.units),
                                     initializer=tf.keras.initializers.RandomNormal(mean=-5, stddev=0.1), trainable=True)
        self.b_mu = self.add_weight(shape=(self.units,),
                                    initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1), trainable=True)
        self.b_rho = self.add_weight(shape=(self.units,),
                                     initializer=tf.keras.initializers.RandomNormal(mean=-5, stddev=0.1), trainable=True)

    def call(self, inputs):
        w_sigma = tf.math.softplus(self.w_rho)
        b_sigma = tf.math.softplus(self.b_rho)
        w = self.w_mu + w_sigma * tf.random.normal(tf.shape(self.w_mu))
        b = self.b_mu + b_sigma * tf.random.normal(tf.shape(self.b_mu))

        outputs = tf.matmul(inputs, w) + b
        if self.activation:
            outputs = self.activation(outputs)

        kl = self.kl_divergence(self.w_mu, w_sigma, self.prior_mu, self.prior_sigma) + \
             self.kl_divergence(self.b_mu, b_sigma, self.prior_mu, self.prior_sigma)
        self.add_loss(kl_weight * kl)
        return outputs

    def kl_divergence(self, mu, sigma, prior_mu, prior_sigma):
        return tf.reduce_sum(
            tf.math.log(prior_sigma / sigma) +
            (sigma ** 2 + (mu - prior_mu) ** 2) / (2.0 * prior_sigma ** 2) - 0.5
        )

# Build Bayesian Model
def build_bayesian_model(input_shape):
    inputs = Input(shape=input_shape)
    x = BayesianConv2D(64, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = BayesianConv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = BayesianDense(256, activation='relu')(x)
    outputs = BayesianDense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

# Model training
def evaluate_model(model):
    raw_data = tifffile.imread(f'{DATA_PATH}image_test.tif')
    predictions = []

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

# --- Main ---
if __name__ == '__main__':
    train_data, val_data, train_labels, val_labels, num_channels = load_data()
    input_shape = (WINDOW_SIZE, WINDOW_SIZE, num_channels)

    model = build_bayesian_model(input_shape)
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

# Save multiple predictions as npy and mat
    pred_map = evaluate_model(model)
    scio.savemat(f'{DATA_PATH}pred_map.mat', {'pred_map': pred_map})
    np.save(f'{DATA_PATH}pred_map.npy', pred_map)
    plt.imshow(pred_map, cmap='jet')
    plt.show()

# Model evaluation
    labels = tifffile.imread(f'{DATA_PATH}class_test.tif')
    labels = labels[WINDOW_SIZE // 2: -(WINDOW_SIZE // 2),
                    WINDOW_SIZE // 2: -(WINDOW_SIZE // 2)]
    accuracy = np.mean(pred_map.flatten() == labels.flatten() - 1)
    print(f'Test Accuracy: {accuracy:.4f}')
