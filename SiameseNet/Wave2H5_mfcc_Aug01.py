#Create training Dataset, required at least 32GB RAM to run, 97.4K pairs
import os
import glob
import librosa
import numpy as np
from tqdm import tqdm

import librosa
import numpy as np
from tqdm import tqdm

def compute_mfcc(audio, sample_rate, n_mfcc=20, n_fft=2048, hop_length=512):
    """
    Compute MFCC features from an audio signal.

    Parameters:
    - audio: Audio signal array.
    - sample_rate: Sampling rate of the audio signal.
    - n_mfcc: Number of MFCC features to compute.
    - n_fft: Length of the FFT window.
    - hop_length: Number of samples between successive frames.

    Returns:
    - mfcc_features: Numpy array of MFCC features.
    """
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_features_db = librosa.power_to_db(mfcc_features, ref=np.max)
    return mfcc_features_db

def audio2mfcc(file_path, n_mfcc=20, max_pad_len=432):
    """
    Load an audio file, compute its MFCC features, and pad the result to a fixed size.

    Parameters:
    - file_path: Path to the audio file.
    - n_mfcc: Number of MFCC features to compute.
    - max_pad_len: Maximum length to pad the MFCC features.

    Returns:
    - padded_mfcc_features: Padded MFCC feature matrix.
    """
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None, mono=True)
    # Compute MFCC features
    mfcc_features = compute_mfcc(y, sr, n_mfcc=n_mfcc)
    # Pad or truncate the features
    pad_width = max_pad_len - mfcc_features.shape[1]
    if pad_width < 0:
        mfcc_features = mfcc_features[:, :max_pad_len]
    else:
        mfcc_features = np.pad(mfcc_features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    
    return mfcc_features
    
mfcc1 = audio2mfcc(file_a)    



def get_training_data(base_path):
    class_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    train_pairs = []
    train_labels = []
    train_filenames = []

    progress_updates = 0
    for i, folder_a in enumerate(class_folders):
        files_a = glob.glob(os.path.join(base_path, folder_a, '*.wav'))
        for j, folder_b in enumerate(class_folders):
            files_b = glob.glob(os.path.join(base_path, folder_b, '*.wav'))
            for file_a in files_a:
                for file_b in files_b:
                    if i == j and files_a.index(file_a) < files_b.index(file_b) or i < j:
                        # Generate MFCC features
                        mfcc1 = audio2mfcc(file_a)
                        mfcc2 = audio2mfcc(file_b)
                        train_pairs.append([mfcc1, mfcc2])
                        train_labels.append(0 if i == j else 1)
                        train_filenames.append([file_a, file_b])
                        progress_updates += 1
                        if progress_updates % 100 == 0:
                            print(f"Progress: {progress_updates} pairs processed")

    return np.array(train_pairs), np.array(train_labels), np.array(train_filenames)

# Example usage
#base_path = '/home/zj/Data/Calltypes/calltype_wave/Train10class/' #/home/zj/Data/Calltypes/FSLTestNew /home/zj/Data/Calltypes/FSLTestNew /home/zj/Data/Calltypes/FSLTrainNoAug 
base_path = '/home/zj/Data/Calltypes/FSLOffClassTestImprv2' #/home/zj/Data/Calltypes/FSLTrain /home/jack/Data/calltypes/FSLOffClassTest FSLOffClassTestNoAug /home/zj/Data/Calltypes/FSLTrainNew
#FSLTrain  is too huge and can't be generated /home/zj/Data/Calltypes/FSLTestNoAugImpr FSLTestNoAugImpr /home/zj/Data/Calltypes/FSLTestNoAugImpr  /home/zj/Data/Calltypes/FSLTrainNewV1
#/home/zj/Data/Calltypes/FSLOffClassTestImprv3 /home/zj/Data/Calltypes/FSLOffClassTestImpr /home/zj/Data/Calltypes/FSLOffClassTestImprv2 /home/zj/Data/Calltypes/FSLOffClassTestImprv3
X_train, Y_train, all_filenames_train = get_training_data(base_path)

import h5py
# Save train data to an H5 file or proceed with further processing

def save_to_h5(X_train, Y_train, all_filenames_train, h5_file_path):
    with h5py.File(h5_file_path, 'w') as hf:
        # Saving feature vectors
        hf.create_dataset('X_train', data=X_train)
        
        # Saving labels
        hf.create_dataset('Y_train', data=Y_train)
        
        # Saving filenames. Since filenames are strings, they need special handling
        filenames = np.array(all_filenames_train, dtype=h5py.string_dtype(encoding='utf-8'))
        hf.create_dataset('all_filenames_train', data=filenames)

# Specify the path for your H5 file
h5_file_path = '../Dataset/Mfccpairs_FSLOffClassTestImprv2.h5' #/home/zj/Data/Calltypes/FSLOffClassTestNoAug

# Save to H5
print('Saving training data to H5 file...')
save_to_h5(X_train, Y_train, all_filenames_train, h5_file_path)
print('Done!')
#93K records

#Because the data is already stored into H5 database, the working dataset could be freed from memory, they are quite huge
#Occupied 25GB RAM
del X_train, Y_train, all_filenames_train
import gc
gc.collect()
print('Memoary released!')
