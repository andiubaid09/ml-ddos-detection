import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import layers

num_classes = 2
model = keras.Sequential([
    #Layer 1
    layers.Dense(units=112, activation='relu', input_shape=(x_train_scaled.shape[1],)),
    layers.Dropout(0.0), #No Dropout

    #Layer 2
    layers.Dense(units=128, activation='relu'),
    layers.Dropout(0.1),

    #Layer 3
    layers.Dense(units=64, activation='tanh'),
    layers.Dropout(0.0),

    #Output Layers
    layers.Dense(units=num_classes, activation='softmax')
])

learning_rate = 0.0009739768003064985

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
#Training Model MultiLayer Perceptron
model.fit(
    x_train_scaled,
    y_train,
    epochs=50,
    validation_data=(x_test_scaled, y_test)
)

import numpy as np

y_pred_prob = model.predict(x_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)
