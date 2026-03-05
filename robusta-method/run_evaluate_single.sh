#!/bin/bash
# Quick single-experiment runner. Uncomment ONE section and run.
# Override Python: PYTHON=/path/to/python bash run_evaluate_single.sh

PYTHON=${PYTHON:-python}
COMMON="--single-branch --modality MIX2"

# --- Audio babble - Average Weighting ---
# $PYTHON infer.py $COMMON \
#     --corrupt-modality AUDIO --corruption-percentage 1.0 \
#     --test_corrupted_audio_list ./list/babble_3.list \
#     --alpha 0.5 --beta 0.5

# --- Audio babble - GMM Dynamic Weighting ---
# $PYTHON infer_gmm.py $COMMON \
#     --corrupt-modality AUDIO --corruption-percentage 1.0 \
#     --test_corrupted_audio_list ./list/babble_3.list \
#     --c 0.008 --x0 -500

# --- RGB brightness - Average Weighting ---
# $PYTHON infer_rgb.py $COMMON \
#     --corrupt-modality RGB --corruption-percentage 1.0 \
#     --test-corrupted-list ./list/brightness_3_10.list \
#     --alpha 0.5 --beta 0.5

# --- RGB brightness - GMM Dynamic Weighting ---
# $PYTHON infer_rgb_gmm.py $COMMON \
#     --corrupt-modality RGB --corruption-percentage 1.0 \
#     --test-corrupted-list ./list/brightness_3_10.list \
#     --c 0.008 --x0 -500

# --- Both - Average Weighting ---
# $PYTHON inferBOTH.py $COMMON \
#     --corrupt-modality RGB AUDIO --corruption-percentage 1.0 \
#     --test-corrupted-list ./list/brightness_3_10.list \
#     --test_corrupted_audio_list ./list/babble_3.list \
#     --alpha 0.5 --beta 0.5

# --- Both - GMM Dynamic Weighting ---
# $PYTHON inferBOTH_gmm.py $COMMON \
#     --corrupt-modality RGB AUDIO --corruption-percentage 1.0 \
#     --test-corrupted-list ./list/brightness_3_10.list \
#     --test_corrupted_audio_list ./list/babble_3.list \
#     --c 0.008 --x0 -500
