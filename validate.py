import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from scipy.io import wavfile
from sklearn.metrics import accuracy_score
from load_dataset import load_dataset, decoder, load_dataset_attacked, encoder
from utils import read_audio, calculate_mfcc, validate_single_audio

def validate(type = 'raw', snr=5, makesample = False, iteration=5, model='saved_model\ds_cnn_pure.h5', alpha_calc=False, alpha=0.1):

    # Load the trained model
    
    model = load_model(model)

    test_data = [] 
    test_label = []
    # Load the validation data
    print("TYPE = ", type)
    if alpha_calc == False:
        if(type == 'raw'):
            _, __, test_data_raw, test_label = load_dataset()
            test_data, test_label = process_raw(test_data_raw, test_label)

        elif (type == 'fgsm'):
            data_dir = f'audio/fgsm/snr{snr}'
            test_data_raw, test_label = load_dataset_attacked(data_dir)
            test_data, test_label = process_fgsm(test_data_raw, test_label, snr)
            # print(test_data[0])
            # print(test_data.shape)
            
        elif (type == 'pgd'):
            data_dir = f'audio/pgd/snr{snr}/iter{iteration}'
            test_data_raw, test_label = load_dataset_attacked(data_dir)
            test_data, test_label = process_fgsm(test_data_raw, test_label, snr, pgd=True, iteration=iteration)
    else:
        data_dir = f'audio/testalpha/alpha{alpha}/snr{snr}/iter{iteration}'
        test_data_raw, test_label = load_dataset_attacked(data_dir)
        test_data, test_label = process_fgsm(test_data_raw, test_label, snr, pgd=True, iteration=iteration)
    # print(test_data)
    print(test_data.shape, snr)
    test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[2]))

    # Make predictions on the validation data
    test_pred = model.predict(test_data)

    # Convert the one-hot encoded labels to categorical labels
    test_label_real = decoder(test_label)

    # Convert the predicted probabilities to categorical predictions
    test_label_pred = decoder(test_pred)
    accuracy = accuracy_score(test_label_real, test_label_pred)
    print("Accuracy : ", accuracy)
    # Select 10 random audio samples from the validation set
    if(makesample == True):
        num_samples = 3
        sample_indices = np.random.choice(len(test_data), num_samples, replace=False)

        # Loop over the selected samples
        for i, index in enumerate(sample_indices):
            # Get the audio file and label
            audio_file = test_data_raw[index]
            label = test_label_real[index]
            
            # Get the predicted label
            predicted_label = test_label_pred[index]
            
            # Load the audio file
            sample_rate, audio = wavfile.read(audio_file)
            
            # Save the audio file with its label and predicted label
            # filename = f'sample_{i}_label_{label}_predicted_{predicted_label}.wav'
            # filepath = os.path.join('audio/samples', filename)
            # wavfile.write(filepath, sample_rate, audio)

    return accuracy


def process_raw(test_data, test_label):

    print("Loading test data ... ")
    if os.path.exists('dataset_label/test_data.npy'):
        print("Test data found!")
        test_data = np.load('dataset_label/test_data.npy')
        test_label_ = np.load('dataset_label/test_label.npy')
    else:
        length = len(test_data)
        test_data_mfcc = []
        test_label_ = []
        for idx in range(length):
            readaudio = read_audio(test_data[idx])
            if(readaudio is not None):
                test_data_mfcc.append(calculate_mfcc(readaudio))
                test_label_.append(test_label[idx])
            else:
                continue
            print('conversion status : ', idx, '/',length, end='\r')
        test_data = np.array(test_data_mfcc)
        test_label_ = np.array(test_label_)
        np.save('dataset_label/test_data.npy', test_data)
        np.save('dataset_label/test_label.npy', test_label_)
        

    return test_data, test_label_

def process_fgsm(test_data, test_label, snr, pgd=False, iteration=5):

    print("Loading test data ... ")
    if(pgd == False):
        if os.path.exists(f'dataset_label/fgsm/test_data_fgsm_snr{snr}.npy'):
            print("Test data found!")
            test_data = np.load(f'dataset_label/fgsm/test_data_fgsm_snr{snr}.npy')
            test_label = np.load(f'dataset_label/fgsm/test_label_fgsm_snr{snr}.npy')
        else:
            length = len(test_data)
            test_data_mfcc = []
            test_label_ = []
            for idx in range(length):
                readaudio = read_audio(test_data[idx])
                if(readaudio is not None):
                   test_data_mfcc.append(calculate_mfcc(readaudio))
                   test_label_.append(test_label[idx])
                else:
                    continue
                print('conversion status : ', idx, '/',length, f' for SNR = {snr}',end='\r')
            test_data = np.array(test_data_mfcc)
            test_label = np.array(test_label_)
            np.save(f'dataset_label/fgsm/test_data_fgsm_snr{snr}.npy', test_data)
            np.save(f'dataset_label/fgsm/test_label_fgsm_snr{snr}.npy', test_label)
    if(pgd == True):
        if os.path.exists(f'dataset_label/pgd{iteration}/test_data_pgd_snr{snr}_iter{iteration}.npy'):
            print("Test data found!")
            test_data = np.load(f'dataset_label/pgd{iteration}/test_data_pgd_snr{snr}_iter{iteration}.npy')
        else:
            length = len(test_data)
            test_data_audio= []
            test_data_mfcc = []
            for idx in range(length):
                test_data_audio.append(read_audio(test_data[idx]))
                test_data_mfcc.append(calculate_mfcc(test_data_audio[idx]))
                print('conversion status : ', idx, '/',length, f' for SNR = {snr}',end='\r')
            test_data = np.array(test_data_mfcc)
            np.save(f'dataset_label/pgd{iteration}/test_data_pgd_snr{snr}_iter{iteration}.npy', test_data)
        
        if os.path.exists(f'dataset_label/pgd{iteration}/test_label_pgd_snr{snr}_iter{iteration}.npy'):
            print("test_label found!")
            test_label = np.load(f'dataset_label/pgd{iteration}/test_label_pgd_snr{snr}_iter{iteration}.npy')
        else:
            test_label = np.array(test_label)
            np.save(f'dataset_label/pgd{iteration}/test_label_pgd_snr{snr}_iter{iteration}.npy', test_label)

    return test_data, test_label

# accuracy = []
# for snr in [5, 10, 15, 20, 25, 30]:
#     accuracy.append((snr, validate('fgsm', snr=snr)))
# accuracy.append(('vanilla', validate()))

# print(accuracy)

# validate_single_audio("audio/speech_commands/speech_commands_v0.02/yes/test/0a9f9af7_nohash_2.wav")