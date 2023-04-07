import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import create_ds_cnn_model
from utils import calculate_mfcc, model_info, read_audio
from load_dataset import load_dataset
import math
import os
import numpy as np
from tensorflow.keras.models import load_model

def process_raw(train_data, test_data, train_label, test_label):
    print("Loading train data ... ")

    if os.path.exists('dataset_label/train_data.npy'):
        print("Train data found!")
        train_data = np.load('dataset_label/train_data.npy')
    else:
        length = len(train_data)
        train_data_audio= []
        train_data_mfcc = []
        for idx in range(length):
            train_data_audio.append(read_audio(train_data[idx]))
            train_data_mfcc.append(calculate_mfcc(train_data_audio[idx]))
            print('conversion status : ', idx, '/',length, end='\r')
        train_data = np.array(train_data_mfcc)
        np.save('dataset_label/train_data.npy', train_data)

    print("Train data loaded")
    print("Loading test data ... ")
    if os.path.exists('dataset_label/test_data.npy'):
        print("Test data found!")
        test_data = np.load('dataset_label/test_data.npy')
    else:
        length = len(test_data)
        test_data_audio= []
        test_data_mfcc = []
        for idx in range(length):
            test_data_audio.append(read_audio(test_data[idx]))
            test_data_mfcc.append(calculate_mfcc(test_data_audio[idx]))
            print('conversion status : ', idx, '/',length, end='\r')
        test_data = np.array(test_data_mfcc)
        np.save('dataset_label/test_data.npy', test_data)

    print("Test data loaded")
    if os.path.exists('dataset_label/test_label.npy'):
        print("test_label found!")
        test_label = np.load('dataset_label/test_label.npy')
    else:
        test_label = np.array(test_label)
        np.save('dataset_label/test_label.npy', test_label)

    if os.path.exists('dataset_label/train_label.npy'):
        print("train_label found!")
        train_label = np.load('dataset_label/train_label.npy')
    else:
        train_label = np.array(train_label)
        np.save('dataset_label/train_label.npy', train_label)

    return train_data, test_data, train_label, test_label


def train(model_existing = False):

     # Split the data into training and validation sets
    train_data, train_label, test_data, test_label = load_dataset()

    train_data, test_data, train_label, test_label = process_raw(train_data, test_data, train_label, test_label)

    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[2]))
    test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[2]))

    if(model_existing == False):
        # Define the model
        checkpoint_path = "/Users/soonha/Desktop/SeniorThesis/model_ckpts/dscnn"
        model_settings, model_size_info = model_info()
        model = create_ds_cnn_model(model_settings, model_size_info)

        # Define the loss function and optimizer
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()

        # Compile the model
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

        # Define the checkpoints
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )

        # Train the model
        batch_size = 100
        epochs = 30
        steps_per_epoch = math.ceil(len(train_data) / batch_size)
        val_steps = math.ceil(len(test_data) / batch_size)

        history = model.fit(
            x=train_data,
            y=train_label,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(test_data, test_label),
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=[checkpoint_callback],
            shuffle = True
        )
    else:
        model = load_model("/Users/soonha/Desktop/SeniorThesis/saved_model/ds_cnn_pure.h5")
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data, test_label)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # Save the final model
    model.save('saved_model/ds_cnn_pure.h5')


