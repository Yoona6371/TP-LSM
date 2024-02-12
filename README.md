# TP-LSM
**TP-LSM: Temporal Pyramid Long Short-Term Time Modeling Network for Multi-label Action Detection**

![model](https://github.com/Yoona6371/TP-LSM/blob/main/model2.jpg)

In this repository, we provide an implementation of "TP-LSM: Temporal Pyramid Long Short-Term Time Modeling Network for Multi-label Action Detection" on Charades dataset (Localization setting, i.e., Charades_v1_localize). If you want to train and evaluate TP-LSM, you can follow the following steps.

## Prepare the I3D feature
Like the previous works (e.g. TGM, PDAN), MS-TCT is built on top of the pre-trained I3D features. Thus, feature extraction is needed before training the network.

1. Please download the Charades dataset (24 fps version) from this [link](https://prior.allenai.org/projects/charades).
2. Follow this [repository](https://github.com/piergiaj/pytorch-i3d) to extract the snippet-level I3D feature.

## Dependencies
Please satisfy the following dependencies to train TP-LSM correctly:

+ pytorch 1.10.0
+ python 3.8
+ timm 0.4.12
+ pickle5
+ scikit-learn
+ numpy

## Quick Start
1. Change the rgb_root to the extracted feature path in the train.py.
2. Use `./run_TPLSM_Charades.sh` for training on Charades-RGB. The best logits will be saved automatically in `./save_logit` (you can change in tran.py).

