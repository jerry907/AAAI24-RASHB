# AAAI24: Near-Optimal Resilient Aggregation Rules for Distributed Learning Using 1-Center and 1-Mean Clustering with Outliers

This codebase is based on a fork of the [Leaf](leaf.cmu.edu) benchmark suite, [tRFA](https://github.com/krishnap25/tRFA), and MEBwO(https://github.com/tomholmes19/Minimum-Enclosing-Balls-with-Outliers), and provides scripts to reproduce the experimental results in the paper [Near-Optimal Resilient Aggregation Rules for Distributed 
Learning Using 1-Center and 1-Mean Clustering with Outliers].

If you use this code, please cite the paper using the bibtex reference below

```
place_holder
```

Abstract
-----------------
Byzantine machine learning has garnered considerable attention in light of the unpredictable faults that can occur in large-scale distributed learning systems. The key to secure resilience against Byzantine machines in distributed learning is resilient aggregation mechanisms. Although abundant resilient aggregation rules have been proposed, they are designed in ad-hoc manners, imposing extra barriers on comparing, analyzing, and improving the rules across performance criteria. This paper studies near-optimal aggregation rules using clustering in the presence of outliers. Our outlier-robust clustering approach utilizes geometric properties of the update vectors provided by workers. 

Our analysis show that constant approximations to the 1-center and 1-mean clustering problems with outliers provide near-optimal resilient aggregators for metric-based criteria, which have been proven to be crucial in the homogeneous and heterogeneous cases respectively. In addition, we discuss two contradicting types of attacks under which no single aggregation rule is guaranteed to improve upon the naive average. Based on the discussion, we propose a two-phase resilient aggregation framework. 

We run experiments for image classification using a non-convex loss function. The proposed algorithms outperform previously known aggregation rules by a large margin with both homogeneous and heterogeneous data distributions among non-faulty workers.

The [accompanying paper](place_holder).


Installation                                                                                                                   
-----------------
This code is written in Python 3.8
and has been tested on PyTorch 1.4+.
A conda environment file is provided in `rfa.yml` with all dependencies except PyTorch. 
It can be installed by using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
as follows

```
conda env create -f rfa.yml 
```

**Installing PyTorch:** Instructions to install a PyTorch compatible with the CUDA on your GPUs (or without GPUs) can be found [here](https://pytorch.org/get-started/locally/).


Dataset
-----------
Due to the large size of dataset, we ddi not upload dataset here, and the complete dataset is available at 
https://drive.google.com/file/d/115Yf3yDkapxYZpzh43OCr4D4KX0S3GIG/view?usp=sharing.
1. ### FEMNIST 

  * **Overview:** Character Recognition Dataset
  * **Original dataset:** 62 classes (10 digits, 26 lowercase, 26 uppercase), 3500 total users.(leaf.cmu.edu)
  * **Preprocess:** We sample 5% of the images in the original dataset to construct our datasets. For the homogeneous setting, each client sample images from a uniform distribution over 62 classes.   We generate heterogeneous datasets for clients using categorical distributions qmdrawn from a Dirichlet distribution q ∼ Dir(αp), where p is a prior class distribution over 62 classes (Hsu, Qi, and Brown 2019). Each client sample from a categorical distribution characterized by an independent q . In our experiment for the heterogeneous setting, we let α = 0.1, which is described as the extreme heterogeneity setting in (Allouah et al. 2023a).
  * **Task:** Image Classification
  * **Directory:** ```data/femnist``` 

2. ### CIFAR10

  * **Overview:** Tiny Images Dataset
  * **Original dataset:** 60000 32x32 colour images in 10 classes, with 6000 images per class.(https://www.cs.toronto.edu/~kriz/cifar.html)
  * **Preprocess:** We use a small dataset of 35 clients uniformly sampled from the CIFAR-10 dataset, and each client contains 300 train samples and 60 test samples.
  * **Task:** Image Classification
  * **Directory:** ```data/cifar10``` 


Reproducing Experiments in the Paper
-------------------------------------

As the data has been set up, the scripts provided in the folder ```models/scripts/``` can be used 
to reproduce the experiments in the paper.

Change directory to ```models``` and run the scripts as 
```
./scripts/femnist_cnn/run.sh  
```
