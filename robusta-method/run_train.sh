#!/bin/bash
# Train the RobustA model on clean (uncorrupted) RGB + Audio features.
# Best checkpoint is saved to ckpt/{comet_run_name}/robusta_best.pkl

CUDA_VISIBLE_DEVICES=1 python main.py \
    --single-branch \
    --modality MIX2 \
    --audio-projection lp \
    --max-epoch 50 \
    --batch-size 640 \
    --lr 0.0001 \
    --seed 2333
