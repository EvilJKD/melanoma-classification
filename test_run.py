import pandas as pd
import numpy as np
import os
import cv2
import warnings
import time

import tensorflow as tf
import pickle as pck
from tensorflow import keras
from keras import layers, metrics
from keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam


# Image Preprocessing
def custom_preprocessing(image):
    image_arr = keras.utils.img_to_array(image, data_format=None, dtype=None)
    image = image_arr.astype('float32')
    normalized = image * 1. / 255.0  # Normalización en rango 0 - 1

    return normalized

# Constants
image_size = (128, 128)
batch_size = 32
image_dir = "../../../Proyecto Integrador/scripts/dataset3/test"
selected_model_path = "./Checkpoints/saved-model-150.hdf5"
test_datagen = ImageDataGenerator(preprocessing_function=custom_preprocessing)


# Load selected model
def load_model(selected_model_path):
    selected_model = keras.models.load_model(selected_model_path)

    return selected_model

# Make Predictions to measure metrics
# Loads the test set and gets the metrics
def make_predictions(test_dataframe):
    test_data_generator = test_datagen.flow_from_dataframe(dataframe=test_dataframe, directory=image_dir,
                                                            x_col="filename", y_col="label",
                                                            class_mode="binary", shuffle=False,
                                                            subset='training', target_size=image_size,
                                                            batch_size=batch_size)
    selected_model = load_model(selected_model_path)

    filenames = test_data_generator.filenames
    nb_samples = len(filenames)

    predicted_values = selected_model.predict(test_data_generator, nb_samples).ravel()


    true_names = test_data_generator.filenames
    true_values = test_data_generator.classes



    current_results = pd.DataFrame({
        'filename': true_names,
        'true_value': true_values,
        'predicted_-value': predicted_values
    })

    # Metrics used are metrics=['AUC', 'accuracy', 'Precision', 'Recall'], in that order
    results = selected_model.evaluate(test_data_generator)
    # So, the results array go as follows:
    # 0: loss   1: AUC  2: accuracy     3: precision    4: recall

    print("---------- Results ----------")
    print(f"AUC:\t{results[1]}")
    print(f"Accuracy:\t{results[2]}")
    print(f"Precision:\t{results[3]}")
    print(f"Recall:\t{results[4]}")

    return current_results



if __name__ == "__main__":
    # Gets the names and labels of the test set and loads them into a DataFrame for further use
    test_data = pd.read_csv(f"{image_dir}/labels.csv")

    # Initialize method
    make_predictions(test_data)