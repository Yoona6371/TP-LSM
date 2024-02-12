#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH
# python train.py \
# -lr 0.0002 \  mu
# -num_clips 128 \ mu
# -batch_size 3  mu
# mu f=0.1 b=3 pati=25


# -lr 0.0001 \  cha
# -num_clips 256 \  cha
# -batch_size 8or10 cha
# charades f=0.1 b=3 pati=7

# TSU



python train.py \
-dataset charades \
-mode rgb \
-model TP_LSM \
-train True \
-num_clips 128 \
-lr 0.0002 \
-comp_info False \
-epoch 150 \
-unisize True \
-batch_size 3