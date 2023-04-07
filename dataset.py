import os
import random
import shutil

# Set the path to the directory containing the audio folders
path = '/Users/soonha/Desktop/SeniorThesis/audio/speech_commands/speech_commands_v0.02'

# Set the percentage of files to use for testing
test_percentage = 20

# Loop over the audio folders in the directory
for foldername in os.listdir(path):
    folderpath = os.path.join(path, foldername)
    if os.path.isdir(folderpath):
        # Create a list of all audio file paths in the folder
        audio_files = [os.path.join(folderpath, f) for f in os.listdir(folderpath) if f.endswith('.wav')]

        # Shuffle the list of audio file paths randomly
        random.shuffle(audio_files)

        # Calculate the number of files to use for testing based on the test percentage
        num_test_files = int(len(audio_files) * (test_percentage / 100))

        # Split the audio file paths into train and test sets
        train_files = audio_files[num_test_files:]
        test_files = audio_files[:num_test_files]

        # Create the train and test directories for the current folder
        train_dir = os.path.join(folderpath, 'train')
        test_dir = os.path.join(folderpath, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Move the train files to the train directory
        for file in train_files:
            shutil.move(file, os.path.join(train_dir, os.path.basename(file)))

        # Move the test files to the test directory
        for file in test_files:
            shutil.move(file, os.path.join(test_dir, os.path.basename(file)))
