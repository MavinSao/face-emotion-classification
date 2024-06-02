"""
Author : Mavin Sao
Date : 2024.06.02.
"""
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import data_reader
import numpy as np

# Decide how many epochs to train for.
EPOCHS = 50  # Increased to allow early stopping to be more effective

# Read the data.
dr = data_reader.DataReader()

# Expand dimensions of the data to match the expected input shape of the model
dr.train_X = np.expand_dims(dr.train_X, axis=-1)
dr.test_X = np.expand_dims(dr.test_X, axis=-1)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen.fit(dr.train_X)

# Create the neural network using the provided architecture.
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Dense(7, activation='softmax'))

# Compile the neural network.
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
              metrics=['accuracy'],
              loss='sparse_categorical_crossentropy')

# Train the neural network.
print("\n\n************ TRAINING START ************ ")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(datagen.flow(dr.train_X, dr.train_Y, batch_size=64),
                    validation_data=(dr.test_X, dr.test_Y),
                    epochs=EPOCHS,
                    callbacks=[early_stop])

# Plot the training results.
data_reader.draw_graph(history)
model.save('emotion-classification-model')
