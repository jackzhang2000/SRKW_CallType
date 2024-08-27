# Similarity Is What You Need: Identifying Southern Resident Killer Whale (SRKW) Call Types Under Sparse Sampling Using Siamese Networks
# Data, Model training and Test code and result 

This repository is Experment Code for CNN call type classification, Siamese Network call type similarity model

## Abstract
Southern Resident Killer Whales (SRKW) are highly intelligent marine mammals facing extinction in the North Pacific. These whales emit three types of sounds: clicks, whistles, and pulsed calls, with over 40 distinct pulsed call types known as their "dialects." However, due to limited and poor-quality data, only nine call types have sufficient annotated recordings for analysis. To address this challenge, this paper proposes a progressive approach to improve SRKW call type identification. Initially, data augmentation techniques were employed to enhance training data volume, leading to a traditional CNN model achieving 97.8% accuracy on 17 call types. Subsequently, a Siamese Network model was developed to infer similarities between call types, achieving state-of-the-art performance with 98.5% accuracy. Finally, the Siamese Network's generalization ability was evaluated on nine additional call types, maintaining high accuracy and recall but with lower precision, which can be improved through manual review and retraining. This study demonstrates that data augmentation and Siamese Networks are effective strategies for overcoming few-shot learning challenges in SRKW call type identification, achieving robust performance even with limited annotated data.

Keywords: Marine Mammal Conservation, Southern Resident Killer Whales (SRKW); Acoustic Classification; Few-Shot Learning (FSL); Convolutional Neural Network (CNN), Data Augmentation, Siamese Network; Similarity measurement; Contrastive Learning; Transfer Learning; Meta Learning.

## Requirements

CUDA Version: 11.4

Python : 3.11.5

To install dependencies:

```setup
sudo pip3 install -r requirements.txt
```
## Dataset
Original Wave files:

![image](https://github.com/user-attachments/assets/d7343888-5926-4da3-8bd2-e6bbfb13ea36)


Experment obtained the following performance on 17 In-training call types for CNN, Siamese Network, and 14 out-of-training call types for Siamese Network
Model Metrics	CNN	Siamese network on in-training call type	Siamese network on out-of-training call type
Accuracy	97.80%	98.50%	79.70%
Recall (Sensitivity)	81.40%	89.30%	79.50%
Specificity	98.80%	99.20%	79.70%
F1	81.40%	88.80%	44.40%
Precision	81.40%	88.20%	30.80%
![image](https://github.com/user-attachments/assets/6facfff5-1792-4ec3-be7b-dc1b3a673147)

