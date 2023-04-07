import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import create_ds_cnn_model
from utils import calculate_mfcc, mix_snr, model_info, read_audio
from load_dataset import load_dataset
import math
import os
import numpy as np
from tensorflow.keras.models import load_model
from fgsm import FGSM

def process_fgsm(train_data, test_data, train_label, test_label, snr):
    print("Loading train data ... ")
    if os.path.exists(f'dataset_label/train_data_advtrain_snr{snr}.npy'):
        print("Dataset found_train")
        train_data = np.load(f'dataset_label/train_data_advtrain_snr{snr}.npy')
        train_label = np.load(f'dataset_label/train_label_advtrain_snr{snr}.npy')
    else:
        length = len(train_data)
        train_data_audio= []
        train_data_mfcc = []
        train_label_ = []
        for idx in range(length):
            readaudio = read_audio(train_data[idx])
            if(readaudio is not None):
                train_data_audio.append(readaudio)
            else:
                continue
            original_audio = readaudio
            perturb_audio = FGSM(train_data[idx], train_label[idx])
            adversarial_sample = mix_snr(original_audio, perturb_audio, snr)
            train_data_mfcc.append(calculate_mfcc(adversarial_sample))
            train_label_.append(train_label[idx])
            print('conversion status : ', idx, '/',length, end='\r')
        train_data = np.array(train_data_mfcc)
        train_label = np.array(train_label_)
        np.save(f'dataset_label/train_data_advtrain_snr{snr}.npy', train_data)
        np.save(f'dataset_label/train_label_advtrain_snr{snr}.npy', train_label)

    if os.path.exists(f'dataset_label/test_data_advtrain_snr{snr}.npy'):
        print("dataset found")
        test_data = np.load(f'dataset_label/test_data_advtrain_snr{snr}.npy')
        test_label = np.load(f'dataset_label/test_label_advtrain_snr{snr}.npy')
    else:
        length = len(test_data)
        test_data_audio= []
        test_data_mfcc = []
        test_label_ = []
        for idx in range(length):
            readaudio = read_audio(test_data[idx])
            if(readaudio is None):
                continue
            
            calculatemfcc = calculate_mfcc(readaudio)
            test_data_mfcc.append(calculatemfcc)
            test_label_.append(test_label[idx])
            print('conversion status : ', idx, '/',length, end='\r')

        test_data = np.array(test_data_mfcc)
        test_label = np.array(test_label_)
        np.save(f'dataset_label/test_data_advtrain_snr{snr}.npy', test_data)
        np.save(f'dataset_label/test_label_advtrain_snr{snr}.npy', test_label)

    return train_data, test_data, train_label, test_label


def train(model_existing = False, snr=35):

     # Split the data into training and validation sets
    train_data, train_label, test_data, test_label = load_dataset()

    train_data, test_data, train_label, test_label = process_fgsm(train_data, test_data, train_label, test_label, snr)


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
        model = load_model("saved_model/dscnn_advtrain_fgsm.h5")
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data, test_label)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # Save the final model
    model.save(f'saved_model/dscnn_advtrain_fgsm_snr{snr}.h5')

for snr in [25, 30, 40]:
    train(snr=snr)