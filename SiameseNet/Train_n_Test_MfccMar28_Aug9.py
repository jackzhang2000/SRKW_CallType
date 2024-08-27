import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import glob
import librosa
import numpy as np
import tqdm

#Read training data from H5 database
import h5py
import numpy as np

def read_from_h5(h5_file_path):
    with h5py.File(h5_file_path, 'r') as hf:
        X_train = hf['X_train'][:]
        Y_train = hf['Y_train'][:]
        all_filenames_train = hf['all_filenames_train'][:]

        # Check if filenames are stored as bytes and decode them
        if isinstance(all_filenames_train[0], bytes):
            all_filenames_train = [filename.decode('utf-8', errors='ignore') for filename in all_filenames_train]
        elif isinstance(all_filenames_train[0], np.bytes_):
            all_filenames_train = [filename.tobytes().decode('utf-8', errors='ignore') for filename in all_filenames_train]

    return X_train, Y_train, all_filenames_train
    


# Specify the path for your H5 file
#h5_file_path = 'S2T7_trn_dataFeb29.h5'
#h5_file_path = 'Tst_calltypemar1.h5' #Six classes paired mfcc data
h5_file_path = '../DataProcess/Mfcc_FSLTrain_Mar28.h5' #10 classes paired mfcc data
#h5_file_path = 'Mfcc_10Class_Mar11.h5' #Six classes paired mel data
#h5_file_path = './DataProcess/Mfcc_10Cla_ZNorml1samp.h5' #Try 10 classes Normalized mfcc data

# Read from H5
print('Reading training data from H5 file...')
X_train1, Y_train1, all_filenames_train1 = read_from_h5(h5_file_path)

#split the training set to training and validation
from sklearn.model_selection import train_test_split

# Split data into training and validation sets
X_train, X_val, Y_train, Y_val, filenames_train, filenames_val = train_test_split(X_train1, Y_train1, all_filenames_train1, test_size=0.2, random_state=42)

# Print the shapes of the datasets
print("Training Data Shapes:")
print("X_train1:", X_train.shape)
print("Y_train1:", Y_train.shape)
print("filenames_train:", len(filenames_train))

# Print the shapes of the datasets
print("Validation Data Shapes:")
print("X_val:", X_val.shape)
print("Y_val:", Y_val.shape)
print("filenames_val:", len(filenames_val))

import numpy as np

# Assuming Y_train1 is already loaded and contains your labels

# Calculate the counts of each label
labels, counts = np.unique(Y_train, return_counts=True)

# Display the counts
for label, count in zip(labels, counts):
    print(f"Label {label}: {count} counts")


# Assuming you have a GPU available, use it; otherwise, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import glob
import librosa
import numpy as np
import tqdm


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=(6, 6), stride=1, padding=1)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=(4, 4), stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)  # p is the dropout probability
        # Placeholder for the dynamically calculated size
        self._to_linear = None
        self.fc1 = None  # Placeholder, will be defined after _to_linear is calculated
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)  # Apply dropout after activation functions or pooling layers
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
            self.fc1 = nn.Linear(self._to_linear, 512).to(x.device)  # Dynamically create the fc1 layer
        x = x.view(-1, self._to_linear)  # Flatten
        x = F.relu(self.fc1(x))
        
        return x
        
# Later, when defining your CNNEncoder in your network
# Just make sure to pass a dummy input through the encoder once before starting the training to set _to_linear


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn_encoder = CNNEncoder()
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x1, x2):
        output1 = self.cnn_encoder(x1)
        output2 = self.cnn_encoder(x2)
        output1 = self.fc2(output1)
        output2 = self.fc2(output2)
        return output1, output2    

# Assuming you have your DataLoader setup
# Instantiate the model and move it to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
siamese_net = SiameseNetwork().to(device)

# Example forward pass (assuming you have DataLoader setup correctly)
# for input1, input2, labels in data_loader:
#     input1, input2 = input1.to(device), input2.to(device)
#     output1, output2 = siamese_network(input1.unsqueeze(1), input2.unsqueeze(1))  # Add channel dimension

from torchviz import make_dot


# Now, when you pass the input to the model, it should no longer raise the device mismatch error:
dummy_input = torch.zeros(1, 1, 20, 432)  # Sample input size
# Then, before passing your input to the model, ensure it's on the same device as the model:
dummy_input = dummy_input.to(device)  # Move dummy_input to the appropriate device

# Run model to get output
output1, output2 = siamese_net(dummy_input, dummy_input)


#graph = make_dot(output1, params=dict(list(siamese_net.named_parameters())))
#graph.render("SiameseNetwork", format="png")

import torch.onnx

# Export to ONNX
torch.onnx.export(siamese_net, (dummy_input, dummy_input), "siamese_networkJul5.onnx", verbose=True)


# Assuming you have a Dataset class for your data
class SiameseDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        pair = self.pairs[index]
        label = self.labels[index]
        return pair[0], pair[1], torch.tensor(label, dtype=torch.float32)  # Convert label to tensor

# Assuming you have a DataLoader for your dataset


# Create datasets
train_dataset = SiameseDataset(X_train, Y_train)
val_dataset = SiameseDataset(X_val, Y_val)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

#train_dataset = SiameseDataset(X_train1, Y_train1)
#train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)



# Assuming you have your DataLoader setup

# Print the structure of the CNNEncoder part of the Siamese network
print("CNN Encoder structure:")
print(siamese_net.cnn_encoder)

# Optionally, if you want to print the entire Siamese network structure, you can do:
print("Complete Siamese Network structure:")
print(siamese_net)

import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) +
                                      (target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
        
# Use ContrastiveLoss in your training loop
criterion = ContrastiveLoss()

# Use the Adam optimizer with a lower learning rate
#optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)
optimizer = optim.Adam(siamese_net.parameters(), lr=0.001, weight_decay=0.0001)
#optimizer = optim.Adam(siamese_net.parameters(), lr=0.00001)  # Adjust the learning rate as needed


from torch.cuda.amp import GradScaler, autocast
# Instantiate GradScaler for mixed-precision training
scaler = GradScaler()

# Training loop
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Early stopping and learning rate scheduler
early_stopping_patience = 10
no_improvement_epochs = 0
best_val_loss = float('inf')

# Scheduler to reduce learning rate on plateau
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)


from torch.optim.lr_scheduler import ReduceLROnPlateau

# Initialize the optimizer
optimizer = optim.Adam(siamese_net.parameters(), lr=0.001, weight_decay=0.0001)

# Initialize the learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# Training loop with early stopping and LR scheduling
epochs = 50
early_stopping_patience = 10
no_improvement_epochs = 0
best_val_loss = float('inf')

for epoch in range(epochs):
    siamese_net.train()
    total_train_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        input1, input2, target = data
        input1, input2, target = input1.to(device), input2.to(device), target.to(device)

        input1 = input1.unsqueeze(1)
        input2 = input2.unsqueeze(1)

        optimizer.zero_grad()

        with autocast():
            output1, output2 = siamese_net(input1, input2)
            loss = criterion(output1, output2, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()

    average_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch}, Training Loss: {average_train_loss}")

    # Validation step
    siamese_net.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            input1, input2, target = data
            input1, input2, target = input1.to(device), input2.to(device), target.to(device)

            input1 = input1.unsqueeze(1)
            input2 = input2.unsqueeze(1)

            output1, output2 = siamese_net(input1, input2)
            loss = criterion(output1, output2, target)

            total_val_loss += loss.item()

    average_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch}, Validation Loss: {average_val_loss}")

    # Check for early stopping
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        no_improvement_epochs = 0
        # Save the best model
        torch.save(siamese_net.state_dict(), '../Model/MfccFSLTrain_dropout_Mar28Aug9.pth')
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= early_stopping_patience:
            print("Early stopping triggered")
            break

    # Adjust learning rate based on validation loss
    scheduler.step(average_val_loss)
    
# After training, delete the train dataset from memory
del X_train1, Y_train1, all_filenames_train1
# After training is done, release memory by deleting train_loader and invoking garbage collection
del train_loader, train_dataset
import gc
gc.collect()
print('Memory occupied by Dataset and Dataloader is released!')