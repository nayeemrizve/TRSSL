# Towards Realistic Semi-Supervised Learning

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-realistic-semi-supervised-learning/open-world-semi-supervised-learning-on-cifar-1)](https://paperswithcode.com/sota/open-world-semi-supervised-learning-on-cifar-1?p=towards-realistic-semi-supervised-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-realistic-semi-supervised-learning/open-world-semi-supervised-learning-on-1)](https://paperswithcode.com/sota/open-world-semi-supervised-learning-on-1?p=towards-realistic-semi-supervised-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-realistic-semi-supervised-learning/open-world-semi-supervised-learning-on-cifar)](https://paperswithcode.com/sota/open-world-semi-supervised-learning-on-cifar?p=towards-realistic-semi-supervised-learning)

Implementation of [Towards Realistic Semi-Supervised Learning](https://arxiv.org/abs/2207.02269).

Deep learning is pushing the state-of-the-art in many computer vision applications. However, it relies on large annotated data repositories, and capturing the unconstrained nature of the real-world data is yet to be solved. Semi-supervised learning (SSL) complements the annotated training data with a large corpus of unlabeled data to reduce annotation cost. The standard SSL approach assumes unlabeled data are from the same distribution as annotated data. Recently, a more realistic SSL problem, called open-world SSL, is introduced, where the unannotated data might contain samples from unknown classes. In this paper, we propose a novel pseudo-label based approach to tackle SSL in open-world setting. At the core of our method, we utilize sample uncertainty and incorporate prior knowledge about class distribution to generate reliable class-distribution-aware pseudo-labels for unlabeled data belonging to both known and unknown classes. Our extensive experimentation showcases the effectiveness of our approach on several benchmark datasets, where it substantially outperforms the existing state-of-the-art on seven diverse datasets including CIFAR-100 (∼17%), ImageNet-100 (∼5%), and Tiny ImageNet (∼9%). We also highlight the flexibility of our approach in solving novel class discovery task, demonstrate its stability in dealing with imbalanced data, and complement our approach with a technique to estimate the number of novel classes.




## Training
```shell
# For CIFAR10 10% Labels and 50% Novel Classes 
python3 train.py --dataset cifar10 --lbl-percent 10 --novel-percent 50 --arch resnet18

# For CIFAR100 10% Labels and 50% Novel Classes 
python3 train.py --dataset cifar100 --lbl-percent 10 --novel-percent 50 --arch resnet18

# For CIFAR100 10% Labels and 50% Novel Classes and Imbalance Factor 10
python3 train.py --dataset cifar10 --lbl-percent 10 --novel-percent 50 --arch resnet18 --imb-factor 10

For training on the other datasets, please download the dataset and put under the "name_of_the_dataset" folder and put the train and validation/test images under "train" and "test" folder. After that, please set the value of data_root argument as "name_of_the_dataset".

# For Tiny ImageNet 10% Labels and 50% Novel Classes
python3 train.py --dataset tinyimagenet --lbl-percent 10 --novel-percent 50 --arch resnet18

# For ImageNet-100 10% Labels and 50% Novel Classes
python3 train.py --dataset imagenet100 --lbl-percent 10 --novel-percent 50 --arch resnet50

# For Oxford-IIIT Pet 50% Labels and 50% Novel Classes
python3 train.py --dataset oxfordpets --lbl-percent 50 --novel-percent 50 --arch resnet18

# For FGVC-Aircraft 50% Labels and 50% Novel Classes
python3 train.py --dataset aircraft --lbl-percent 50 --novel-percent 50 --arch resnet18

# For Stanford-Cars 50% Labels and 50% Novel Classes
python3 train.py --dataset stanfordcars --lbl-percent 50 --novel-percent 50 --arch resnet18
```

## Citation
```
@inproceedings{rizve2022towards,
  title={Towards realistic semi-supervised learning},
  author={Rizve, Mamshad Nayeem and Kardan, Navid and Shah, Mubarak},
  booktitle={European Conference on Computer Vision},
  pages={437--455},
  year={2022},
  organization={Springer}
}

```
