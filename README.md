# Similarity Is What You Need: Identifying Southern Resident Killer Whale (SRKW) Call Types Under Sparse Sampling Using Siamese Networks
# Data, Model training and Test code and result 

This repository is Experment Code for CNN call type classification, Siamese Network call type similarity model

## Abstract
As an important branch of deep learning, few-shot Learning does not require a large amount of data but chooses a softer approach to solve problems, where it can be perfectly integrated with techniques such as meta learning and data augmentation. In this repository, I test four algorithms:Matching Network, ProtoNet, MAML and Distribution Propagation Graph NN on four FSL datasets: miniImageNet, Omniglot, CUB-200-2011 and CIFAR-FS.

## Requirements

CUDA Version: 10.4

Python : 3.6.9

To install dependencies:

```setup
sudo pip3 install -r requirements.txt
```
## Dataset
For your convenience, you can download the datasets directly from links on the left, or you can make them from scratch following the original splits on the right. 

|    Dataset    | Algorithms Paper |
| :-----------: |:----------------:|
|  [Mini-ImageNet](https://drive.google.com/open?id=15WuREBvhEbSWo4fTr1r-vMY0C_6QWv4w)  |  [Matching Networks](https://arxiv.org/pdf/1606.04080.pdf)  | 
|    [Omniglot](https://drive.google.com/file/d/1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG)   |   [ProtoNet](https://arxiv.org/abs/1703.05175)   |
|  [CIFAR-FS](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS)  |   [MAML](https://arxiv.org/abs/1703.03400)   |
|      [CUB-200-2011](https://github.com/wyharveychen/CloserLookFewShot/tree/master/filelists/CUB)     |   [DP GNN](https://arxiv.org/abs/2003.14247)   |



Experment obtained the following performance on mini-ImageNet, Omniglot, CUB-200-2011 and CIFAR-FS.

**miniImageNet**:

|     Method    |   Backbone   |   5way-1shot   |   5way-5shot   |
| :-----------: |:------------:|----------------|:--------------:|
|  MatchingNet  |    ConvNet   |   43.56±0.84   |   55.31± 0.73  |
|    ProtoNet   |    ConvNet   |   49.42±0.78   |   68.20±0.66   |
|      MAML     |    ConvNet   |   48.70±1.84   |   55.31±0.73   |
|      DPGN     |    ConvNet   | 66.01±0.36 | 82.83±0.41 |
