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
For your convenience, you can download the datasets directly from links on the left, or you can make them from scratch following the original splits on the right. 

Call Type	Non-Augmented Wave Files	Augmented Wave Files	Testing Wave Files	Remark
ArgBermHBW	12	60	4	"SX: Southern Resident Killer Whale
ArgBermHBW: Bermuda Humpback Whales
DelawSpwale: Delaware Sperm Whale
Sea wave: Ocean background noise
TX: Bigg‘s (Transient) Killer Whales in Northern Pacific

"
DelawSpwale	10	50	10	
S10	19	48	10	
S18	16	50	6	
S1d	28	102	16	
S2	38	190	12	
S31	14	46	7	
S35	4	13	5	
S36	104	104	10	
S40	2	5	4	
S44b	18	90	7	
S4d	48	239	16	
S5	3	10	4	
S6	8	37	11	
S9	2	10	10	
Seawave	4	4	4	
T7	4	20	4	
Grand Total	334	1078	140	
![image](https://github.com/user-attachments/assets/d7343888-5926-4da3-8bd2-e6bbfb13ea36)


Experment obtained the following performance on mini-ImageNet, Omniglot, CUB-200-2011 and CIFAR-FS.

**miniImageNet**:

|     Method    |   Backbone   |   5way-1shot   |   5way-5shot   |
| :-----------: |:------------:|----------------|:--------------:|
|  MatchingNet  |    ConvNet   |   43.56±0.84   |   55.31± 0.73  |
|    ProtoNet   |    ConvNet   |   49.42±0.78   |   68.20±0.66   |
|      MAML     |    ConvNet   |   48.70±1.84   |   55.31±0.73   |
|      DPGN     |    ConvNet   | 66.01±0.36 | 82.83±0.41 |
