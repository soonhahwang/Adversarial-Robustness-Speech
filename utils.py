import numpy as np
import tensorflow as tf
import os
from tensorflow.python.ops import gen_audio_ops as audio_ops
from model import create_ds_cnn_model
import math
from tensorflow.keras.models import load_model
from load_dataset import decoder

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length

    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': sample_rate,
    }
    
def model_info():
    label_count = 10
    sample_rate = 16000
    clip_duration_ms = 1000
    window_size_ms = 40
    window_stride_ms = 20
    dct_coefficient_count = 10
    model_settings = prepare_model_settings(label_count, sample_rate, clip_duration_ms, window_size_ms, window_stride_ms, dct_coefficient_count)
    model_size_info = [5, 64, 10, 4, 2, 2, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1]

    return model_settings, model_size_info

def read_audio(dir):
    try:
        wav_file = tf.io.read_file(dir)
        model_settings, model_size_info = model_info()
        audio = audio_ops.decode_wav(wav_file, desired_channels=1, desired_samples=model_settings["sample_rate"])  
        return audio.audio
    except Exception as e:
        print(f"Error occurred in PGD() function for audio {dir}: {str(e)}")
        return None

def calculate_mfcc(waveform, dir = True):
    try:
        waveform = tf.transpose(waveform)
        # equal_length = tf.concat([waveform, zero_padding], 0)

        spectrogram = tf.signal.stft(waveform, frame_length=640, frame_step=320,
                            fft_length=640)
        spectrogram = tf.abs(spectrogram)
        # spectrogram = spectrogram[..., tf.newaxis]
        num_spectrogram_bins = spectrogram.shape[-1]

        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, 16000, lower_edge_hertz,
        upper_edge_hertz)
        mel_spectrograms = tf.tensordot(
        spectrogram, linear_to_mel_weight_matrix, 1)

        mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

        # print(log_mel_spectrograms)

        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., :10]
        mfccs = tf.reshape(mfccs, (1,490))

        return mfccs

    except Exception as e:
        print(f"Error occurred in PGD() function for audio: {str(e)}")
        return None

def validate_single_audio(audio_path, model_path='saved_model/ds_cnn_pure.h5'):
    
    model = load_model(model_path)
    audio = read_audio(audio_path)
    mfcc_audio = calculate_mfcc(audio)
    mfcc_audio_np = np.array(mfcc_audio)
    prediction = model.predict(mfcc_audio_np)

    print("Predicted : ", decoder(prediction), " with probability of ", np.max(prediction))
    
def mix_snr(audio, noise, snr):
  noise, _ = get_noise_from_sound(audio, noise, snr)
  max = np.max(audio)
  min = np.min(audio)
  audio = audio + noise
  audio = tf.clip_by_value(audio, min, max)

  return audio
  
def get_noise_from_sound(signal,noise,SNR):
    RMS_s=math.sqrt(np.mean(signal**2))
    #required RMS of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    
    #current RMS of noise
    RMS_n_current=math.sqrt(np.mean(noise**2))
    noise=noise*(RMS_n/RMS_n_current)
    # print(RMS_n_current)
    return noise, RMS_n/RMS_n_current