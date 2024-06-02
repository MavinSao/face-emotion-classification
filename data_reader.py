"""
Author : Mavin Sao
Date : 2024.06.02.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

# Class to read and preprocess data.
class DataReader():
    def __init__(self):
        self.label = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []

        self.read_images()

    def read_images(self):
        print("Reading Data...")
        self.train_X, self.train_Y = self.load_data("data/train")
        self.test_X, self.test_Y = self.load_data("data/test")

        self.train_X = np.asarray(self.train_X) / 255.0
        self.train_Y = np.asarray(self.train_Y)
        self.test_X = np.asarray(self.test_X) / 255.0
        self.test_Y = np.asarray(self.test_Y)

        # Data reading is complete.
        # Print the information of the read data.
        print("\n\nData Read Done!")
        print("Training X Size : " + str(self.train_X.shape))
        print("Training Y Size : " + str(self.train_Y.shape))
        print("Test X Size : " + str(self.test_X.shape))
        print("Test Y Size : " + str(self.test_Y.shape) + '\n\n')

    def load_data(self, directory):
        data_X = []
        data_Y = []
        classes = os.listdir(directory)
        for i, cls in enumerate(classes):
            print("Opening " + cls + "/")
            for el in os.listdir(os.path.join(directory, cls)):
                img_path = os.path.join(directory, cls, el)
                img = Image.open(img_path) #.convert('L') ->  # Convert to grayscale
                data_X.append(np.asarray(img))
                data_Y.append(self.label.index(cls))
                img.close()
        return data_X, data_Y

def draw_graph(history):
    train_history = history.history["loss"]
    validation_history = history.history["val_loss"]
    fig = plt.figure(figsize=(8, 8))
    plt.title("Loss History")
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS Function")
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig.savefig("train_history.png")

    train_history = history.history["accuracy"]
    validation_history = history.history["val_accuracy"]
    fig = plt.figure(figsize=(8, 8))
    plt.title("Accuracy History")
    plt.xlabel("EPOCH")
    plt.ylabel("Accuracy")
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig.savefig("accuracy_history.png")
