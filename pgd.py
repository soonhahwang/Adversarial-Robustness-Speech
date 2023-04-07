import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops
from peturb import create_peturbation_from_mfcc
from utils import model_info, read_audio, mix_snr, validate_single_audio, get_noise_from_sound
from load_dataset import load_dataset, decoder
import numpy as np
from scipy.io import wavfile
import os
from validate import validate_single_audio
import random


def PGD(audio, SNR, iterations, label, alpha = 0.01):
    try:
        wav_file = tf.io.read_file(audio)
        audio = audio_ops.decode_wav(wav_file, desired_channels=1, desired_samples=16000)  
        audio = audio.audio  
        original = audio

        audio_peturbed = create_peturbation_from_mfcc(audio, label)
        original = np.array(original)
        audio_peturbed  = np.array(audio_peturbed)
        _, epsilon = get_noise_from_sound(original, audio_peturbed, SNR)
        alpha = alpha*epsilon
        

        max = np.max(audio)
        min = np.min(audio)

        below = audio - epsilon
        above = audio + epsilon

        for i in range(iterations):
            audio_peturbed = create_peturbation_from_mfcc(audio, label)
            audio = audio + alpha*audio_peturbed
            # print(t)
            audio = tf.clip_by_value(audio, below, above)
            audio = tf.clip_by_value(audio, min, max)

        return audio

    except Exception as e:
        print(f"Error occurred in PGD() function for audio {audio}: {str(e)}")
        return None


def audio_process(test_data, test_label):

    print("Loading test data ... ")
    if os.path.exists('dataset_label/audio_process_data.npy'):
        print("Test data found!")
        test_data = np.load('dataset_label/audio_process_data.npy')
    else:
        test_data = [read_audio(test_data[idx]) for idx in range(len(test_data))]
        test_data = np.array(test_data)
        np.save('dataset_label/audio_process_data.npy', test_data)

    if os.path.exists('dataset_label/audio_process_label.npy'):
        print("test_label found!")
        test_label = np.load('dataset_label/audio_process_label.npy')
    else:
        test_label = np.array(test_label)
        np.save('dataset_label/audio_process_label.npy', test_label)

    return test_data, test_label


def attack(SNRs=[5, 10, 15, 20, 25, 30, 35, 40, 45], iteration=5, alpha = 0.01):
    _, __, test_data_raw, test_label = load_dataset()
    # print(test_data_raw)
    # test_data, test_label = audio_process(test_data_raw, test_label)
    
    for SNR in SNRs:
        pgd_noise = []
        original = []
        length = 500
        data_and_labels = list(zip(test_data_raw, test_label))
        random.shuffle(data_and_labels)
        test_data_raw, test_label = zip(*data_and_labels)
        original = np.array(test_data_raw)

        directory = os.path.join('audio/pgd/', f'snr{SNR}/iter{iteration}')
        total_files = 0

        for _, _, files in os.walk(directory):
            total_files += len(files)

        count = length-total_files
        cur_idx = 0
        for idx in range(length):

            if count < 0:
                break

            ################
            # Simple trick #
            ################
            filename = original[cur_idx]
            # print(filename)
            path_parts = filename.split(os.path.sep)
            category = path_parts[-3]  # should be 'off'
            file_name = path_parts[-1]  # should be '92521ccc_nohash_0.wav'
            new_file_path = os.path.join(category, file_name)

            ###############
            ###############
            filepath = os.path.join('audio/pgd/', f'snr{SNR}/iter{iteration}', new_file_path)
            if os.path.exists(filepath):
                continue

            directory = os.path.join('audio/pgd/', f'snr{SNR}/iter{iteration}', category)
            if not os.path.exists(directory):
                os.makedirs(directory)

            pgd_audio = PGD(original[cur_idx], SNR, iteration, test_label[cur_idx])
            if pgd_audio is None:
                continue
            pgd_noise.append(pgd_audio)
            
            print('conversion status : ', cur_idx, '/', length-total_files, "for SNR = ", SNR, "for Iteration", iteration,"PATH = ", original[cur_idx], end='\r')

            # print(filepath)
            audio_attacked=np.int16(pgd_noise[cur_idx]*32767)
            wavfile.write(filepath, 16000, audio_attacked)
            count -= 1
            cur_idx += 1

# for k in range(10):            
#     for iteration in [5, 20]:
#         attack([5, 10, 15, 20, 25, 30, 35, 40, 45], iteration=iteration)