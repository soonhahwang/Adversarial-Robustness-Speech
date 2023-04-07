import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops
from peturb import create_peturbation_from_mfcc
from utils import model_info, read_audio, mix_snr, validate_single_audio
from load_dataset import load_dataset
import numpy as np
from scipy.io import wavfile
import os

def FGSM(audio, label):
    wav_file = tf.io.read_file(audio)
    audio = audio_ops.decode_wav(wav_file, desired_channels=1, desired_samples=16000) 
    audio = audio.audio   
    audio_peturbed = create_peturbation_from_mfcc(audio, label)

    return audio_peturbed

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


def attack(SNRs = [5, 10, 15, 20, 25, 30, 35, 40, 45]):
    _, __, test_data_raw, test_label = load_dataset()
    # test_data, test_label = audio_process(test_data_raw, test_label)
    fgsm_noise=[]
    original=[]
    length = len(test_data_raw)
    for idx in range(length):
        fgsm_noise.append(FGSM(test_data_raw[idx], test_label[idx]))
        original.append(read_audio(test_data_raw[idx]))
        print('conversion status : ', idx, '/',length, end='\r')
    fgsm_noise = np.array(fgsm_noise)
    original = np.array(original)
    np.save('fgsm_noise.npy', fgsm_noise)
    np.save('original.npy', original)

    fgsm_noise = np.load('fgsm_noise.npy')
    original = np.load('original.npy')

    PATH = "audio/fgsm/"
    for SNR in SNRs:
        for idx in range(length):
            audio_attacked = mix_snr(original[idx], fgsm_noise[idx], SNR)


            ################
            # Simple trick #
            ################
            filename = test_data_raw[idx]
            path_parts = filename.split(os.path.sep)
            category = path_parts[-3]  # should be 'off'
            file_name = path_parts[-1]  # should be '92521ccc_nohash_0.wav'
            new_file_path = os.path.join(category, file_name)
            
            ###############
            ###############

            filepath = os.path.join('audio/fgsm/', f'snr{SNR}', new_file_path)
            print(filepath)
            audio_attacked=np.int16(audio_attacked*32767)
            wavfile.write(filepath, 16000, audio_attacked)