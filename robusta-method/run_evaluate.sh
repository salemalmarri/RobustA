#!/bin/bash
# =========================================================================
# Unified evaluation runner for RobustA paper experiments.
#
# Dispatches to the correct infer script based on corruption type and mode:
#   Static:  infer.py / infer_rgb.py / inferBOTH.py
#   GMM:     infer_gmm.py / infer_rgb_gmm.py / inferBOTH_gmm.py
#
# Override Python: PYTHON=/path/to/python bash run_evaluate.sh
# =========================================================================

PYTHON=${PYTHON:-python}
COMMON="--single-branch --modality MIX2"
PERCENTAGES=(0.0 0.25 0.5 0.75 1.0)

ALPHA=0.5
BETA=0.5
C=0.008
X0=-500

RGB_LISTS=(
    brightness_3_10.list
    contrast_3_10.list
    fog_3_10.list
    rain_3_10.list
    motion_blur_3_10.list
    saturate_3_10.list
    bit_error_3_10.list
    shot_noise_3_10.list
)

AUDIO_LISTS=(
    babble_3.list
    bitrate_3.list
    hfchannel_3.list
    pink_3.list
    pitch_shift_3.list
    random_dropout_3.list
    reverb_3.list
    white_3.list
)

echo "=== RGB Corruptions - Average Weighting ==="
for list_file in "${RGB_LISTS[@]}"; do
  for perc in "${PERCENTAGES[@]}"; do
    echo "RGB avg: $list_file perc=$perc"
    $PYTHON infer_rgb.py $COMMON \
      --corrupt-modality RGB --corruption-percentage "$perc" \
      --test-corrupted-list "./list/$list_file" \
      --alpha "$ALPHA" --beta "$BETA"
  done
done

echo "=== RGB Corruptions - GMM Dynamic Weighting ==="
for list_file in "${RGB_LISTS[@]}"; do
  for perc in "${PERCENTAGES[@]}"; do
    echo "RGB gmm: $list_file perc=$perc"
    $PYTHON infer_rgb_gmm.py $COMMON \
      --corrupt-modality RGB --corruption-percentage "$perc" \
      --test-corrupted-list "./list/$list_file" \
      --alpha "$ALPHA" --beta "$BETA" --c "$C" --x0 "$X0"
  done
done

echo "=== Audio Corruptions - Average Weighting ==="
for list_file in "${AUDIO_LISTS[@]}"; do
  for perc in "${PERCENTAGES[@]}"; do
    echo "Audio avg: $list_file perc=$perc"
    $PYTHON infer.py $COMMON \
      --corrupt-modality AUDIO --corruption-percentage "$perc" \
      --test_corrupted_audio_list "./list/$list_file" \
      --alpha "$ALPHA" --beta "$BETA"
  done
done

echo "=== Audio Corruptions - GMM Dynamic Weighting ==="
for list_file in "${AUDIO_LISTS[@]}"; do
  for perc in "${PERCENTAGES[@]}"; do
    echo "Audio gmm: $list_file perc=$perc"
    $PYTHON infer_gmm.py $COMMON \
      --corrupt-modality AUDIO --corruption-percentage "$perc" \
      --test_corrupted_audio_list "./list/$list_file" \
      --alpha "$ALPHA" --beta "$BETA" --c "$C" --x0 "$X0"
  done
done

echo "=== Both Corruptions - Average Weighting ==="
for rgb_list in "${RGB_LISTS[@]}"; do
  for audio_list in "${AUDIO_LISTS[@]}"; do
    for perc in "${PERCENTAGES[@]}"; do
      echo "Both avg: RGB=$rgb_list Audio=$audio_list perc=$perc"
      $PYTHON inferBOTH.py $COMMON \
        --corrupt-modality RGB AUDIO --corruption-percentage "$perc" \
        --test-corrupted-list "./list/$rgb_list" \
        --test_corrupted_audio_list "./list/$audio_list" \
        --alpha "$ALPHA" --beta "$BETA"
    done
  done
done

echo "=== Both Corruptions - GMM Dynamic Weighting ==="
for rgb_list in "${RGB_LISTS[@]}"; do
  for audio_list in "${AUDIO_LISTS[@]}"; do
    for perc in "${PERCENTAGES[@]}"; do
      echo "Both gmm: RGB=$rgb_list Audio=$audio_list perc=$perc"
      $PYTHON inferBOTH_gmm.py $COMMON \
        --corrupt-modality RGB AUDIO --corruption-percentage "$perc" \
        --test-corrupted-list "./list/$rgb_list" \
        --test_corrupted_audio_list "./list/$audio_list" \
        --alpha "$ALPHA" --beta "$BETA" --c "$C" --x0 "$X0"
    done
  done
done

echo "Evaluation sweep complete."
