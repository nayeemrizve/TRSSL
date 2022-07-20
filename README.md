# Towards Realistic Semi-Supervised Learning

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
```

## Citation
```
@inproceedings{rizve2022towards,
Title={Towards Realistic Semi-Supervised Learning},
Author={Mamshad Nayeem Rizve and Navid Kardan and Mubarak Shah},
booktitle={European Conference on Computer Vision},
Year={2022}

```
