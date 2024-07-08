# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import gc
import librosa
import soundfile as sf
import librosa.display
import wave
import os
import random
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve,StratifiedShuffleSplit
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
#from torchview import draw_graph
from pathlib import Path
import torchaudio
print("Libraries imported - ready to use PyTorch", torch.__version__)

# import tqdm to show a smart progress meter
from tqdm import trange, tqdm

# import warnings to hide the unnessairy warniings
import warnings
warnings.filterwarnings('ignore')

import h5py
import pandas as pd
#Data Augmenatation and Save to H5 database

#data_root = "/home/zj/Data/Calltypes/FSLTrainNew" #/home/zj/Data/Calltypes/FSLTrainNew /home/zj/Data/Calltypes/FSLTestNew /home/zj/Data/Calltypes/FSLTestNoAug 
data_root = "/home/zj/Data/Calltypes/FSLTestNoAugImpr"
categories = os.listdir(data_root)

all_files = []
for category in categories:
    category_path = os.path.join(data_root, category)
    for path, subdirs, files in os.walk(category_path):
        for name in files:
            all_files.append(os.path.join(path, name))
print('files number is')
print(len(all_files))
print('class number is')
print(len(categories))

#Time consuming work
import os
import pandas as pd

import os
import pandas as pd

# Data directory
data_dir = '/home/zj/Data/Calltypes/FSLTestNoAugImpr'

# Create an empty DataFrame to store wave file information
data_df = pd.DataFrame(columns=['filename', 'fold', 'target', 'files_path'])

# Initialize the fold counter
fold_counter = 1  

# Get the list of subdirectories under the data directory
subdirectories = os.listdir(data_dir)

#subdirectories_to_include = ['S1', 'S2', 'S4']

#subdirectories = [subdir for subdir in os.listdir(data_dir) if subdir in subdirectories_to_include]

# Traverse the subdirectories under the data directory (each subdirectory represents a call type)
for target, subdir in enumerate(subdirectories):
    subdir_path = os.path.join(data_dir, subdir)
    print(subdir)
    # Traverse files in subdirectories
    for file in os.listdir(subdir_path):
        # Extract file name, fold, target
        filename = file
        fold = fold_counter  # Fold# loop from 1 to 5'
        files_path = os.path.join(subdir_path, file)
        category = subdir
        # Get the duration of an audio file
        audio_duration = librosa.get_duration(filename=files_path)

        # If the audio file duration is less than or equal to 5.5 seconds, add it to the DataFrame
        if audio_duration <= 5.5:
            # Create a temporary DataFrame and append to the main DataFrame
            temp_df = pd.DataFrame({'filename': [filename], 'fold': [fold], 'target': [target], 'files_path': [files_path], 'category': [category]})
            data_df = pd.concat([data_df, temp_df], ignore_index=True)

        # Update fold counter
        fold_counter = (fold_counter % 5) + 1

# Save DataFrame to CSV file, Meta data only, no wave vector data, for checking the training data
data_df.to_csv('melFSLTestNoAugImpr.csv', index=False)
print('len(data_df)')
print(len(data_df))

#Start to convert wave files to vector data
# Default FFT window size
n_fft = 2048  # FFT window size
hop_length = 512  # number audio of frames between STFT columns

#2 hours running!
print('Start to convert wave files to vector data!')
def compute_log_mel_spect(audio,sample_rate):  # Size of the Fast Fourier Transform (FFT), which will also be used as the window length
    n_fft=1024
    hop_length=512
    window_type ='hann'
    mel_bins = 60 # Number of mel bands
    normalized_y = librosa.util.normalize(audio)
    Mel_spectrogram = librosa.feature.melspectrogram(y=normalized_y, sr=22050,hop_length=hop_length, win_length=n_fft, n_mels = mel_bins)
    mel_spectrogram_db = librosa.power_to_db(Mel_spectrogram)
    return  mel_spectrogram_db

#Create the list to contain a list of feature data
data=[]

hop_length=512

import pandas as pd

# Create a new DataFrame to store feature data and meta-information
dataset = pd.DataFrame(columns=['filename', 'fold', 'target', 'files_path', 'data'])
# Copy meta information from data_df to dataset
dataset['filename'] = data_df['filename']
dataset['fold'] = data_df['fold']
dataset['target'] = data_df['target']
dataset['files_path'] = data_df['files_path']
dataset['category'] = data_df['category']

import numpy as np
desired_seconds = 3

# Loop through each audio file
for i in tqdm(range(len(data_df))):
    wave_file = dataset.iat[i, 3]
    audio, sample_rate = librosa.load(wave_file)
    mel_spectrogram_db = compute_log_mel_spect(audio, sample_rate)

    # Count the number of frames in the current audio file
    current_frames = mel_spectrogram_db.shape[1]
    # Define the target duration (in frames, calculated based on sample rate and required seconds)
    target_frames = int(desired_seconds * sample_rate / hop_length)

    # Calculate the number of frames required to fill
    padding_frames = max(0, target_frames - current_frames)

    # Using NumPy for padding
    padded_data = np.pad(mel_spectrogram_db, ((0, 0), (0, padding_frames)), mode='constant')
    if current_frames > target_frames:
        mel_spectrogram_db = mel_spectrogram_db[:, :target_frames]

    delta = librosa.feature.delta(padded_data, width=3, mode='nearest')
    data.append(np.dstack((padded_data, delta)))

# Suppose data is a list containing multiple arrays of shape (60, 130, 2)
for i in range(len(data)):
    padding_frames = 173 - data[i].shape[1]
    # If the current frame number is less than the target frame number, fill it
    if padding_frames > 0:
        data[i] = np.pad(data[i], ((0, 0), (0, padding_frames), (0, 0)), mode='edge')

dataset['data']=data
print('len(dataset)')
print(len(dataset))

#save the data to h5 database!
dataset['data']=data
import h5py
# Assume `dataset` is your DataFrame and `data` is a list containing your features
print('Save DataFrame to h5 database')
with h5py.File('../Dataset/melFSLTestNoAugImpr.h5', 'w') as hf:
    # Iterate through your DataFrame and save each item
    for i, row in dataset.iterrows():
        #print(f"Shape of data[{i}]:", np.shape(data[i]))
        grp = hf.create_group(name=str(i))  # Use the index as the name for the group
        grp.create_dataset('filename', data=row['filename'])
        grp.create_dataset('fold', data=row['fold'])
        grp.create_dataset('target', data=row['target'])
        grp.create_dataset('files_path', data=row['files_path'])
        grp.create_dataset('category', data=row['category'])
        grp.create_dataset('data', data=data[i])  # Save the features from your data list

print('Save to h5 database melFSLTestNoAugImpr.h5 successfully!')
