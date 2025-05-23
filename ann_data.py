# Imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from keras import models, layers

from keras.optimizers import Adam

from keras.utils import to_categorical

from keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.utils import shuffle

from data_gen import data_gen  # Ensure this module is correctly implemented

# Set seed for reproducibility

SEED = 8

np.random.seed(SEED)

# Experimental specification

flowrate_ul = 40  # µL/min

flowrate_m3 = flowrate_ul * 1e-9

u = [120, 0, 0, 0]

x_ba = 1.5

x_oh = x_ba * 9

x_in = [x_ba, x_oh, 0, 0]

# Reactor specifications

reactor_length = 825e-6  # meters

diameter_internal = 1e-3  # meters

voidage = 0.54

reactor_volume = np.pi * (diameter_internal / 2) ** 2 * reactor_length

effective_volume = reactor_volume * voidage

res_time = effective_volume / flowrate_m3

# Generate data

noise_level = 0.0

data = data_gen(1000, x_in, u, res_time, noise_level=noise_level, method=0)

raw_data = data.values

X = raw_data[:, :len(x_in)]

Y = to_categorical(raw_data[:, len(x_in)])

# Shuffle and split data

X, Y = shuffle(X, Y, random_state=SEED)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED)

# Normalize features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Model configuration

num_nodes = 32

dropout_rate = 0.1

learning_rate = 0.01

batch_size = 32

epochs = 10

input_dim = len(x_in)

output_dim = Y.shape[1]

# Build model

model = models.Sequential([

    Dense(num_nodes, activation='relu', input_shape=(input_dim,)),

    Dense(num_nodes, activation='relu'),

    Dropout(dropout_rate),

    Dense(output_dim, activation='softmax')

])

model.compile(

    optimizer=Adam(learning_rate=learning_rate),

    loss='categorical_crossentropy',

    metrics=['accuracy']

)

# Train model

val_split = 1 / len(x_in)

history = model.fit(

    X_train, Y_train,

    epochs=epochs,

    batch_size=batch_size,

    validation_split=val_split,

    verbose=1

)


# Plot training history

def plot_history(hist):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(hist.history['loss'], label='Train Loss')

    plt.plot(hist.history['val_loss'], label='Val Loss')

    plt.title('Loss')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(hist.history['accuracy'], label='Train Acc')

    plt.plot(hist.history['val_accuracy'], label='Val Acc')

    plt.title('Accuracy')

    plt.legend()

    plt.tight_layout()

    plt.show()


plot_history(history)

# Evaluate model

test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)

print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# Confusion Matrix

y_true = np.argmax(Y_test, axis=1)

y_pred = np.argmax(model.predict(X_test), axis=1)

conf_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")

plt.show()

# Save and validate model

model.save("my_model.keras")

reconstructed = models.load_model("my_model.keras")

# Test for identical prediction

np.testing.assert_allclose(

    model.predict(X_test[:1]),

    reconstructed.predict(X_test[:1])

)

print("Original vs Reconstructed Prediction:", model.predict(X_test[:1]), reconstructed.predict(X_test[:1]))

