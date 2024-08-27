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

# Specify the path for your H5 file
h5_file_path = '/media/zj/hdd/data/Melpairs_FSLTrainNewV1.h5'
print('Reading training data from H5 file...')
X_train1, Y_train1, all_filenames_train1 = read_from_h5(h5_file_path)



# Print the shapes of the datasets
print("Training Data Shapes:")
print("X_train1:", X_train1.shape)
print("Y_train1:", Y_train1.shape)
print("all_filenames_train1:", len(all_filenames_train1))



import numpy as np

# Assuming Y_train1 is already loaded and contains your labels

# Calculate the counts of each label
labels, counts = np.unique(Y_train1, return_counts=True)

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
# CNN Encoder definition
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=(6, 6), stride=1, padding=1)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=(4, 4), stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)
        self._to_linear = None
        self.fc1 = None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
            self.fc1 = nn.Linear(self._to_linear, 512).to(x.device)
        x = x.view(-1, self._to_linear)
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

# Example forward pass to initialize all layers dynamically
dummy_input = torch.zeros(1, 1, 60, 432, device=device)
_ = siamese_net.cnn_encoder(dummy_input)
# Now, when you pass the input to the model, it should no longer raise the device mismatch error:
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
train_dataset = SiameseDataset(X_train1, Y_train1)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)


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
epochs = 70
for epoch in range(epochs):
    total_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        input1, input2, target = data
        input1, input2, target = input1.to(device), input2.to(device), target.to(device)

        # Correcting the input data shape before passing it to the network
        input1 = input1.unsqueeze(1)  # Adds a channel dimension if it's not already there
        input2 = input2.unsqueeze(1)  # Same for the second input

        # Verify the shape
        #print(input1.shape)  # Should print something like torch.Size([batch_size, 1, 20, 400])

        optimizer.zero_grad()

        with autocast():  # Enable mixed-precision training
            output1, output2 = siamese_net(input1, input2)
            loss = criterion(output1, output2, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    # Print average loss per epoch
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}, Average Loss: {average_loss}")


# After training, save the model
torch.save(siamese_net.state_dict(), '../Model/MelFSLTrainNewV1_modelEpoch70.pth')

# After training, delete the train dataset from memory
del X_train1, Y_train1, all_filenames_train1
# After training is done, release memory by deleting train_loader and invoking garbage collection
del train_loader, train_dataset
import gc
gc.collect()
print('Memory occupied by Dataset and Dataloader is released!')


#Read 6Call testing data from H5 database
import h5py
import numpy as np



# Read testing data from H5 file
h5_file_path = '/media/zj/hdd/data/Melpairs_FSLTestNoAugImpr.h5'
print('Reading testing data from H5 file...')
X_test, Y_test, all_filenames_test = read_from_h5(h5_file_path)

# Print the shapes of the datasets
print("Testing Data Shapes:")
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)
print("all_filenames_test:", len(all_filenames_test))

# Calculate the counts of each label

# Specify the path for your H5 file
#h5_file_path = 'Mel_10Class_Mar11.h5' #10 class mel data from training dataset
#h5_file_path = 'Tst_calltypemar1.h5' #Six classes paired mfcc data
#h5_file_path = 'Mfcc_6Class_Mar12.h5' #Six classes paired mfcc data created on Mar14
#h5_file_path = '../DataProcess/Mfcc_6Cla_ZNorml1samp.h5' #Six classes paired mfcc data created on Mar14


import numpy as np

# Calculate the counts of each label
labels, counts = np.unique(Y_test, return_counts=True)

# Display the counts
for label, count in zip(labels, counts):
    print(f"Label {label}: {count} counts")

# Assuming you have a DataLoader for your dataset
test_dataset = SiameseDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=512) #Change to 256 to reduce VRAM usage on GPU
    
#test The trained 10Class model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import glob
import librosa
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F  # Import the functional module
import csv


# Assuming you have a GPU available, use it; otherwise, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
siamese_net = SiameseNetwork().to(device)

# Load the trained model
siamese_net.load_state_dict(torch.load('../Model/MelFSLTrainNewV1_modelEpoch70.pth'))
siamese_net.eval()

# Run a dummy forward pass to initialize all layers dynamically
# The dimensions here should match the input dimensions you expect for your actual data
# Corrected dummy input initialization with the specific shape (60, 432), mfcc is (20, 432)
dummy_input = torch.zeros(1, 1, 60, 432, device=device)  # Mel spec vector shape 
# Now, run a dummy forward pass through the cnn_encoder part of your Siamese network
_ = siamese_net.cnn_encoder(dummy_input)  # This initializes the fc1 layer dynamically

# After this, you can safely load your saved model
#saved_model_path = '10call_mel_siam_Mar11.pth'
#saved_model_path = '10call_mfcc_siam_Mar12v1.pth'
#siamese_net.load_state_dict(torch.load(saved_model_path, map_location=device), strict=False)
siamese_net.eval()  # Set the model to evaluation mode
siamese_net = siamese_net.cuda()

correct = 0
total = 0
threshold_distance = 0.5  # Define a threshold distance for deciding if pairs are similar

# Convert bytes to strings and extract class info
def decode_and_classify(filename_bytes):
    filename_str = filename_bytes.decode('utf-8')
    class_info = filename_str.split('/')[-2]  # Assuming class info is the second last element in the path
    return filename_str, class_info


# Evaluation function definition
# Function to evaluate the test dataset and write results to CSV
def evaluate_and_write_to_csv(test_loader, all_filenames, output_csv_path):
    
    with torch.no_grad(), open(output_csv_path, mode='w', newline='') as file:  # Disable gradient computation
        writer = csv.writer(file)
        writer.writerow(['Filename1', 'Filename2', 'Class_a', 'Class_b', 'Actual Label', 'Euclidean Distance'])
        #for data in test_loader:
        for i, data in enumerate(tqdm(test_loader, desc="Evaluating")):
        
            input1, input2, labels = data
            input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)
            #x1, x2, labels = data
            #x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            input1 = input1.unsqueeze(1)
            input2 = input2.unsqueeze(1)  
        
            # Forward pass
            output1, output2 = siamese_net(input1, input2) # Get the features with pretrained siamese model
            
            # Calculate the Euclidean distance between the outputs
            #euclidean_distance = F.pairwise_distance(output1, output2)
            euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
           
            for j in range(input1.size(0)):  # Iterate through each item in the batch
                # Write to CSV for each item in the batch
                filename1, class_a = decode_and_classify(all_filenames_test[i*test_loader.batch_size+j][0])
                filename2, class_b = decode_and_classify(all_filenames_test[i*test_loader.batch_size+j][1])
                writer.writerow([filename1, filename2, class_a, class_b, labels[j].item(), euclidean_distance[j].item()])

    

# Assuming all_filenames_test1 is structured correctly to match test_loader indices


#output_csv_path = 'Melsiam_Mar9Evalwith10Class.csv'
output_csv_path = '../Result/FSLTest_drpoutMar29Jul23.csv'
evaluate_and_write_to_csv(test_loader, all_filenames_test, output_csv_path)


# After testing, delete the test dataset from memory
del X_test, Y_test, all_filenames_test
# After training is done, release memory by deleting train_loader and invoking garbage collection
del test_loader, test_dataset
import gc
gc.collect()
print('Memory occupied by Dataset and Dataloader is released!')
