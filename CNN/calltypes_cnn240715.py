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
import torchvision.transforms as transforms
from torchvision.transforms import Resize

# import tqdm to show a smart progress meter
from tqdm import trange, tqdm

# import warnings to hide the unnessairy warniings
import warnings
warnings.filterwarnings('ignore')

import h5py
import pandas as pd

print('Read DataFrame from h5 Train database')
#Read from H5 database
data_list = []

with h5py.File('../Dataset/mfccFSLTrainNew.h5', 'r') as hf: # melFSLTrainNoAugCNNInput melFSLTrainNoAugCNN0606.h5 melFSLTrainNoAug
    for i in hf.keys(): 
        grp = hf[str(i)]
        data_dict = {
            'filename': grp['filename'][()],
            'fold': grp['fold'][()],
            'target': grp['target'][()],
            'files_path': grp['files_path'][()],
            'data': grp['data'][:]
        }
        data_list.append(data_dict)

train_df = pd.DataFrame(data_list)

data_list = []

with h5py.File('../Dataset/mfccFSLTestNoAugImpr.h5', 'r') as hf: # calltypespec20240224.h5 melFSLTestNoAugImpr.h5 melFSLTestNew.h5
    for i in hf.keys():
        grp = hf[str(i)]
        data_dict = {
            'filename': grp['filename'][()],
            'fold': grp['fold'][()],
            'target': grp['target'][()],
            'files_path': grp['files_path'][()],
            'data': grp['data'][:]
        }
        data_list.append(data_dict)

test_df = pd.DataFrame(data_list)


# Select the relevant columns
train_df = train_df[['data', 'target', 'filename', 'fold', 'files_path']]
test_df = test_df[['data', 'target', 'filename', 'fold', 'files_path']]
print(len(train_df))#45330
print(len(test_df))#11403

# ### 3. Data Loader
class CallDataset(Dataset):
    def __init__(self, dataset, transformation, device):
        self.dataset = dataset
        self.device = device
        self.transformation = transformation
        self.length = len(self.dataset)
        self.filenames = self.dataset['filename']
        self.folds = self.dataset['fold']
        self.files_paths = self.dataset['files_path']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Assert that index is within the range of dataset length
        assert index < self.__len__(), f"Index {index} out of bounds!"

        if index >= len(self):
            raise IndexError(f"Index out of bound. Total length is {len(self)} but trying to access index {index}")
        spectrogram = torch.tensor(self._get_audio_spectrogram(index))

        label = self._get_audio_sample_label(index)
        filename = self.filenames.iloc[index]
        fold = self.folds.iloc[index]  #
        files_path = self.files_paths.iloc[index]  #
        spectrogram = spectrogram.permute(2, 0, 1)
        resize_transform = Resize((60, 173)) #hardcode!?? Need change to data shape
        spectrogram = resize_transform(spectrogram)
        return spectrogram, label, filename, fold, label, files_path  # label is used for both label and target

    def _get_additional_info(self, index):
        filename = self.dataset.iloc[index, 0]  # Adjust the index accordingly
        fold = self.dataset.iloc[index, 1]  # Adjust the index accordingly
        target = self.dataset.iloc[index, 2]  # Adjust the index accordingly
        files_path = self.dataset.iloc[index, 3]  # Adjust the index accordingly
        return filename, fold, target, files_path

    def _get_audio_spectrogram(self, index):
        y = self.dataset.iloc[index, 0].astype(np.float32)
        return y

    def _get_audio_sample_label(self, index):
        return self.dataset.iloc[index, 1]

    def _get_filename(self, index):
        # Print the current index and filenames Series length
        # Assert that index is within the range of filenames Series
        assert index < len(self.filenames), f"Index {index} out of bounds for filenames!"
        return self.dataset.iloc[index]['filename']

    def _get_fold(self, index):
        return self.dataset.iloc[index, 3]  # Adjust the index accordingly

    def _get_target(self, index):
        return self.dataset.iloc[index, 4]  #Adjust the index accordingly

    def _get_files_path(self, index):
        return self.dataset.iloc[index, 5]  # Adjust the index accordingly

class MyReshape(object):
    """Reshape the image array."""
    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))

        self.output_size = output_size
    def __call__(self, image):
        return image.reshape(self.output_size)


# build transformation pipelines for data augmentation

train_transforms = transforms.Compose([ MyReshape(output_size=(2,60, 173) ) ])
test_transforms = transforms.Compose([MyReshape(output_size=(2,60, 173))])

batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = CallDataset(train_df,train_transforms,device)
test_data = CallDataset(test_df,test_transforms,device)
import torch.optim as optim
from torchvision import transforms


# Update training loop to use data augmentation
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

import torch
import torch.nn as nn
# V. Model Building

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 5 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=24, kernel_size=(6, 6), stride=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.2),

        )
            
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(6, 6), stride=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(5, 5), stride=(2, 2)),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(5, 5), stride=(2, 2)),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(4, 4), stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self._to_linear = None
        self._initialize_to_linear(torch.zeros(1, 2, 60, 173))

        self.connected_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, 200),  # Use the calculated _to_linear value
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 50),
            nn.Softmax(dim=1),
        )

    def _initialize_to_linear(self, input_data):
        # Calculate the size for the first linear layer
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        self._to_linear = x.nelement() // x.size(0)
        print("Calculated _to_linear:", self._to_linear)

    def forward(self, input_data):
        # Pass data through conv layers
        x = self.conv1(input_data)
        #print(f'Output after conv1: {x.shape}')
        x = self.conv2(x)
        #print(f'Output after conv2: {x.shape}')
        x = self.conv3(x)
        #print(f'Output after conv3: {x.shape}')
        x = self.conv4(x)
        #print(f'Output after conv4: {x.shape}')
        x = self.conv5(x)
        #print(f'Output after conv5: {x.shape}')
        x = self.conv6(x)
        #print(f'Output after conv6: {x.shape}')
        # Flatten and pass data through connected layers
        x = self.connected_layer(x)
        return x

# Instantiate and use the CNN
cnn = CNN()
print(cnn._to_linear)  # Should output 5376 if the input data during training is correct

# Create random input data
input_data = torch.randn(64, 2, 60, 173)  # Assuming this is the correct input size

# Forward pass through the network
output = cnn(input_data)

# ## A. Training Loop
import pandas as pd
def train(model, device, train_loader, optimizer, loss_criteria, epoch):
    model.train()
    train_loss = 0
    results = []  # Result list
    print("------------------------------- Epoch:", epoch, "-------------------------------")
    for batch_idx, (data, target, filename, fold, target_val, files_path) in enumerate(train_loader):

        data, target = data.to(device), target.to(device).long()  # Convert to long integer

        optimizer.zero_grad()
        output = model(data)

        loss = loss_criteria(output, target)  # target is long integer now

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        # Get the predicted value from model
        pred = output.argmax(dim=1, keepdim=True)

        # Collect the result
        for i in range(len(pred)):
            results.append([filename[i], fold[i], target_val[i], files_path[i], pred[i].item()])
            
        
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx + 1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))

    # Convert the results to a DataFrame and save as a CSV file
    results_df = pd.DataFrame(results, columns=['filename', 'fold', 'target', 'files_path', 'prediction'])
    results_df.to_csv(f'mfccFSLTrainNew.csv', index=False)
    
    cleaned_results = []
    for row in results:
        # Decode the byte strings to utf-8 and convert tensors to integers
        filename = row[0].decode('utf-8') if isinstance(row[0], bytes) else row[0]
        fold = row[1].item() if isinstance(row[1], torch.Tensor) else row[1]
        target = row[2].item() if isinstance(row[2], torch.Tensor) else row[2]
        files_path = row[3].decode('utf-8') if isinstance(row[3], bytes) else row[3]
        prediction = row[4]  # Assuming prediction is already an integer
        cleaned_results.append([filename, fold, target, files_path, prediction])

    # Now create the DataFrame with the cleaned data
    clean_results_df = pd.DataFrame(cleaned_results, columns=['filename', 'fold', 'target', 'files_path', 'prediction'])
    clean_results_df.to_csv(f'mfccFSLTrainNew_clean.csv', index=False)

    return avg_loss


# ## B. Testing loop


def test(model, device, test_loader, loss_criteria):
    model.eval()
    test_loss = 0
    correct = 0
    results = []  # Initialize an empty list to collect results
    with torch.no_grad():
        batch_count = 0
        for data, target, filename, fold, target_val, files_path in test_loader:  # Assume the unpacking matches your DataLoader structure
            batch_count += 1
            data, target = data.to(device), target.to(device).long()
            output = model(data)
            test_loss += loss_criteria(output, target).item()

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

            # Collect the result for each prediction
            for i in range(len(predicted)):
                results.append([filename[i], fold[i], target_val[i], files_path[i], predicted[i].item()])
                
    avg_loss = test_loss / batch_count
    accuracy = correct / len(test_loader.dataset)
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset), 100. * accuracy))

    # Convert the results to a DataFrame and save as a CSV file
    results_df = pd.DataFrame(results, columns=['filename', 'fold', 'target', 'files_path', 'prediction'])
    results_df.to_csv(f'mfccFSLTrainNew_mfccFSLTestNoAugImpr0715.csv', index=False)
    
    cleaned_results = []
    for row in results:
        # Decode the byte strings to utf-8 and convert tensors to integers
        filename = row[0].decode('utf-8') if isinstance(row[0], bytes) else row[0]
        fold = row[1].item() if isinstance(row[1], torch.Tensor) else row[1]
        target = row[2].item() if isinstance(row[2], torch.Tensor) else row[2]
        files_path = row[3].decode('utf-8') if isinstance(row[3], bytes) else row[3]
        prediction = row[4]  # Assuming prediction is already an integer
        cleaned_results.append([filename, fold, target, files_path, prediction])

    # Now create the DataFrame with the cleaned data
    clean_results_df = pd.DataFrame(cleaned_results, columns=['filename', 'fold', 'target', 'files_path', 'prediction'])
    clean_results_df.to_csv(f'mfccFSLTrainNew_mfccFSLTestNoAugImpr0715_clean.csv', index=False)

    return avg_loss, accuracy        
        
import pandas as pd

# ## C. Training & Evaluating the model


def training(model, train_loader, test_loader, device):
    # Use an "Adam" optimizer to adjust weights
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    

    # Specify the loss criteria
    loss_criteria = nn.CrossEntropyLoss()

    # Initialize best_valid_loss to a high value
    best_valid_loss = float('inf')  # This line is important

    # Track metrics in these arrays
    epoch_nums = []
    training_loss = []
    validation_loss = []
    
    import csv
    validation_csv_filename = "validation_result0612.csv"
    with open(validation_csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Training Loss", "Validation Loss", "Accuracy"])

    epochs = 500
    print(f'Training on {device}')

    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = train(model, device, train_loader, optimizer, loss_criteria, epoch)
        test_loss, accuracy = test(model, device, test_loader, loss_criteria)

        # Saving model if test_loss is lower
        if test_loss < best_valid_loss:
            best_valid_loss = test_loss
            torch.save(model.state_dict(), 'mfccFSLTrainNewModel0715.pt')

        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
        
        # 
        print('write validation_csv_filename')
        with open(validation_csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, test_loss, accuracy])

    # Now the function will run for the full number of epochs without early stopping

# Assuming you have initialized train_loader and test_loader
model = CNN()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to the appropriate device


training(model, train_loader, test_loader, device)

# Define optimizer with weight decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_criteria = nn.CrossEntropyLoss()  # Define the loss criteria

# Assuming the CNN model has been trained and is named `model`
# Assuming `device` is already set to either 'cuda' or 'cpu'
# Assuming `test_loader` is your DataLoader instance for test data
# Assuming `loss_criteria` is defined, e.g., nn.CrossEntropyLoss()
# Evaluate the model on the test set
test_loss = test(model, device, test_loader, loss_criteria)

# The test function prints the average loss and accuracy, so you don't need to print them again here.

