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



from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import auc


from utils import load_data_and_plot

# Constants 
image_size = (128, 128)
batch_size = 32
epochs = 500
image_dir = "../../../Proyecto Integrador/scripts/dataset3/train"

# Means

fpr_mean = np.linspace(0, 1, 100)
tprs = []
aucs = []

pres = []
recs = []


# Image Preprocessing
def custom_preprocessing(image):
    image_arr = keras.utils.img_to_array(image, data_format=None, dtype=None)
    image = image_arr.astype('float32')
    normalized = image * 1./255.0 # Normalización en rango 0 - 1
    
    return normalized

# preprocessing_function=custom_preprocessing
# Generadores de datos, alimentan al entrenamiento y la validacion
train_datagen = ImageDataGenerator(preprocessing_function=custom_preprocessing)

test_datagen = ImageDataGenerator(preprocessing_function=custom_preprocessing)

train_data = pd.read_csv("../../../Proyecto Integrador/scripts/dataset3/train/labels.csv")

Y = train_data[['label']]




# Definicion del modelo CNN
def define_model_N1():
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape = (128, 128, 3), kernel_initializer = "he_uniform", padding='same', data_format='channels_last'), #Kernel initializer -> he_uniform

        keras.layers.Conv2D(32, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Dropout(0.2),

        keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Dropout(0.2),

        keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.2),

        keras.layers.Conv2D(512, (5, 5), padding='same', activation='relu'),
        keras.layers.Conv2D(512, (5, 5), activation='relu'),
        keras.layers.MaxPooling2D((4, 4)),
        keras.layers.Dropout(0.2),

        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model


def run_training(training_data, validation_data, model, modelName, idx):
    
    print("ModelName", modelName)
    
    # Generadores de los datos para el entrenamiento y la validacion
    train_data_generator = train_datagen.flow_from_dataframe(dataframe=training_data, directory=image_dir,
                                                x_col="filename", y_col="label",
                                                class_mode="binary", shuffle=True, 
                                                subset='training', target_size=image_size, batch_size=batch_size)
    valid_data_generator = test_datagen.flow_from_dataframe(dataframe=validation_data, directory=image_dir,
                                                x_col="filename", y_col="label",
                                                class_mode="binary", shuffle=True, 
                                                subset='training', target_size=image_size, batch_size=batch_size)

    # Optimizador ADAM
    opt = Adam(learning_rate=1e-4)

    filepath = "./Checkpoints/saved-model-{epoch:02d}.hdf5"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        period=50)

    # Compilacion del modelo
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['AUC', 'accuracy', 'Precision', 'Recall']) 
    
    # Entrenamiento
    history = model.fit(train_data_generator, epochs=epochs, validation_data=valid_data_generator, verbose=1, callbacks=[cp_callback])
    
    
    # Métricas
    filenames = valid_data_generator.filenames
    nb_samples = len(filenames)
    
    predicted = model.predict(valid_data_generator, nb_samples).ravel()
    
    true_values = valid_data_generator.classes

    np.save(f'./Histories/{modelName.split("_")[0]}/History_{modelName}_{idx}.npy', history.history)
    
    
def run_model_n1():
    
    print('Thread for Model N2!')
    
    model_name = "Model2"
    

    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=None) # Repeated Stratified 10 Cross Fold
    fold = 1
    
    for train_idx, val_idx in skf.split(np.zeros(len(Y)), Y): # No es necesario saber los X, sino solo los Y, puesto que es estratificado
        
        print(f"Currently training on ProjectModel, fold {fold}")

        # Separacion en Train y Validation
        training_data = train_data.iloc[train_idx]
        validation_data = train_data.iloc[val_idx]

        model_n1 = define_model_N1()

        # Model.fit() / entrenamiento y resultados
        run_training(training_data, validation_data, model_n1, f"{model_name}_{fold}", fold)
        
        fold += 1

    history_path = "./Histories"
    predictions_path = "./Predictions"

    load_data_and_plot(history_path, predictions_path, epochs, model_name)

if __name__ == "__main__":
    run_model_n1() # Para correr el entrenamiento del modelo, recuperar las metricas y los predicted

    
    

