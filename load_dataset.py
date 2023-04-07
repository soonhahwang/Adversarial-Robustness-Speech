import os
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def encoder(label):
    if label == "yes":
        one_hot = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif label == "no":
        one_hot = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif label == "up":
        one_hot = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif label == "down":
        one_hot = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif label == "left":
        one_hot = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif label == "right":
        one_hot = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif label == "on":
        one_hot = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif label == "off":
        one_hot = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif label == "stop":
        one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif label == "go":
        one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    return one_hot

def decoder(one_hot):
    # print(one_hot)
    label_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    index = np.argmax(one_hot, axis=1)
    decoded = [label_list[index[idx]] for idx in range(len(one_hot))]
    return decoded

def load_dataset(data_dir="audio\speech_commands\speech_commands_v0.02"):
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isdir(file_path):
                    for j in os.listdir(file_path):
                        if file == 'train':
                            train_data.append(os.path.join(file_path, j))
                            train_label.append(encoder(folder))
                        else:
                            test_data.append(os.path.join(file_path, j))
                            test_label.append(encoder(folder))
        else:
            continue

    return train_data, train_label, test_data, test_label


def load_dataset_attacked(data_dir="audio/speech_commands/speech_commands_v0.02"):
    test_data = []
    test_label = []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    test_data.append(file_path)
                    test_label.append(encoder(folder))
        else:
            continue

    return test_data, test_label
