#!/bin/bash

COUNT=0
GPU=4
for IDX in {0..14}
do
    if [[ COUNT -eq 4 ]]
    then
        (( COUNT=0 ))
        (( GPU += 1 ))
    fi
    CUDA_VISIBLE_DEVICES=$GPU python train.py --cfg experiments/singleview/mpiiface/mpiiface_train.yaml --test_idx $IDX &
    # echo $COUNT
    # echo $IDX
    # echo $GPU
    (( COUNT += 1 ))
done