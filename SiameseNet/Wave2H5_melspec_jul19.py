import os
import glob
import librosa
import numpy as np
import h5py
from tqdm import tqdm

def compute_log_mel_spect(audio, sample_rate, n_fft=1024, hop_length=512, mel_bins=60):
    """
    Compute log-mel spectrogram from an audio signal.

    Parameters:
    - audio: Audio signal array.
    - sample_rate: Sampling rate of the audio signal.
    - n_fft: Length of the FFT window.
    - hop_length: Number of samples between successive frames.
    - mel_bins: Number of mel bands.

    Returns:
    - mel_spectrogram_db: Numpy array of log-mel spectrogram.
    """
    normalized_y = librosa.util.normalize(audio)
    mel_spectrogram = librosa.feature.melspectrogram(y=normalized_y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=mel_bins)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

def audio2mel(file_path, n_fft=1024, hop_length=512, mel_bins=60, max_pad_len=432):
    """
    Load an audio file, compute its log-mel spectrogram, and pad the result to a fixed size.

    Parameters:
    - file_path: Path to the audio file.
    - n_fft: Length of the FFT window.
    - hop_length: Number of samples between successive frames.
    - mel_bins: Number of mel bands.
    - max_pad_len: Maximum length to pad the mel spectrogram.

    Returns:
    - padded_mel_spectrogram: Padded mel spectrogram matrix.
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)
    mel_spectrogram = compute_log_mel_spect(y, sr, n_fft=n_fft, hop_length=hop_length, mel_bins=mel_bins)
    pad_width = max_pad_len - mel_spectrogram.shape[1]
    if pad_width < 0:
        mel_spectrogram = mel_spectrogram[:, :max_pad_len]
    else:
        mel_spectrogram = np.pad(mel_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mel_spectrogram

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
                        mel1 = audio2mel(file_a)
                        mel2 = audio2mel(file_b)
                        train_pairs.append([mel1, mel2])
                        train_labels.append(0 if i == j else 1)
                        train_filenames.append([file_a, file_b])
                        progress_updates += 1
                        if progress_updates % 100 == 0:
                            print(f"Progress: {progress_updates} pairs processed")

    return np.array(train_pairs), np.array(train_labels), np.array(train_filenames)

def save_to_h5(X_train, Y_train, all_filenames_train, h5_file_path):
    with h5py.File(h5_file_path, 'w') as hf:
        hf.create_dataset('X_train', data=X_train)
        hf.create_dataset('Y_train', data=Y_train)
        filenames = np.array(all_filenames_train, dtype=h5py.string_dtype(encoding='utf-8'))
        hf.create_dataset('all_filenames_train', data=filenames)

#base_path = '/home/zj/Data/Calltypes/FSLTestNoAugImpr'
base_path = '/home/zj/Data/Calltypes/FSLTrainNewV1' 
X_train, Y_train, all_filenames_train = get_training_data(base_path)

h5_file_path = '/media/zj/hdd/data/Melpairs_FSLTrainNewV1.h5'
print('Saving training data to H5 file...')
save_to_h5(X_train, Y_train, all_filenames_train, h5_file_path)
print('Done!')

del X_train, Y_train, all_filenames_train
import gc
gc.collect()
print('Memory released!')
