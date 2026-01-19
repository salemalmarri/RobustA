#!/usr/bin/env bash

##############################################################################
# Overlay Corruptions (white, pink, hfchannel, babble) at severities 3 & 5
##############################################################################

# WHITE noise (severity 3, then severity 5)
CUDA_LAUNCH_BLOCKING=1 python main.py feature_type=vggish device="cpu" file_with_video_paths="XD_Violence_testing.txt" output_path="./outputNEWtemp/white_3" on_extraction="save_numpy" overlay_audio_type="white" audio_corruption_type="overlay" audio_severity=3
# CUDA_LAUNCH_BLOCKING=1 python main.py feature_type=vggish device="cpu" file_with_video_paths="XD_Violence_testing.txt" output_path="./outputNEWtemp/white_5" on_extraction="save_numpy" overlay_audio_type="white" audio_corruption_type="overlay" audio_severity=5

# PINK noise (severity 3, then severity 5)
CUDA_LAUNCH_BLOCKING=1 python main.py feature_type=vggish device="cpu" file_with_video_paths="XD_Violence_testing.txt" output_path="./outputNEWtemp/pink_3" on_extraction="save_numpy" overlay_audio_type="pink" audio_corruption_type="overlay" audio_severity=3
# CUDA_LAUNCH_BLOCKING=1 python main.py feature_type=vggish device="cpu" file_with_video_paths="XD_Violence_testing.txt" output_path="./outputNEWtemp/pink_5" on_extraction="save_numpy" overlay_audio_type="pink" audio_corruption_type="overlay" audio_severity=5

# HFCHANNEL noise (severity 3, then severity 5)
CUDA_LAUNCH_BLOCKING=1 python main.py feature_type=vggish device="cpu" file_with_video_paths="XD_Violence_testing.txt" output_path="./outputNEWtemp/hfchannel_3" on_extraction="save_numpy" overlay_audio_type="hfchannel" audio_corruption_type="overlay" audio_severity=3
# CUDA_LAUNCH_BLOCKING=1 python main.py feature_type=vggish device="cpu" file_with_video_paths="XD_Violence_testing.txt" output_path="./outputNEWtemp/hfchannel_5" on_extraction="save_numpy" overlay_audio_type="hfchannel" audio_corruption_type="overlay" audio_severity=5

# BABBLE noise (severity 3, then severity 5)
CUDA_LAUNCH_BLOCKING=1 python main.py feature_type=vggish device="cpu" file_with_video_paths="XD_Violence_testing.txt" output_path="./outputNEWtemp/babble_3" on_extraction="save_numpy" overlay_audio_type="babble" audio_corruption_type="overlay" audio_severity=3
# CUDA_LAUNCH_BLOCKING=1 python main.py feature_type=vggish device="cpu" file_with_video_paths="XD_Violence_testing.txt" output_path="./outputNEWtemp/babble_5" on_extraction="save_numpy" overlay_audio_type="babble" audio_corruption_type="overlay" audio_severity=5


##############################################################################
# Other Corruptions (bitrate, pitch_shift, reverb, random_dropout) at 3 & 5
##############################################################################

# # BITRATE (severity 3, then severity 5)
CUDA_LAUNCH_BLOCKING=1 python main.py feature_type=vggish device="cpu" file_with_video_paths="XD_Violence_testing.txt" output_path="./outputNEWtemp/bitrate_3" on_extraction="save_numpy" audio_corruption_type="bitrate" audio_severity=3
# CUDA_LAUNCH_BLOCKING=1 python main.py feature_type=vggish device="cpu" file_with_video_paths="XD_Violence_testing.txt" output_path="./outputNEWtemp/bitrate_5" on_extraction="save_numpy" audio_corruption_type="bitrate" audio_severity=5

# # PITCH SHIFT (severity 3, then severity 5)
CUDA_LAUNCH_BLOCKING=1 python main.py feature_type=vggish device="cpu" file_with_video_paths="XD_Violence_testing.txt" output_path="./outputNEWtemp/pitch_shift_3" on_extraction="save_numpy" audio_corruption_type="pitch_shift" audio_severity=3
# CUDA_LAUNCH_BLOCKING=1 python main.py feature_type=vggish device="cpu" file_with_video_paths="XD_Violence_testing.txt" output_path="./outputNEWtemp/pitch_shift_5" on_extraction="save_numpy" audio_corruption_type="pitch_shift" audio_severity=5

# # REVERB (severity 3, then severity 5)
CUDA_LAUNCH_BLOCKING=1 python main.py feature_type=vggish device="cpu" file_with_video_paths="XD_Violence_testing.txt" output_path="./outputNEWtemp/reverb_3" on_extraction="save_numpy" audio_corruption_type="reverb" audio_severity=3
# CUDA_LAUNCH_BLOCKING=1 python main.py feature_type=vggish device="cpu" file_with_video_paths="XD_Violence_testing.txt" output_path="./outputNEWtemp/reverb_5" on_extraction="save_numpy" audio_corruption_type="reverb" audio_severity=5

# # RANDOM DROPOUT (severity 3, then severity 5)
CUDA_LAUNCH_BLOCKING=1 python main.py feature_type=vggish device="cpu" file_with_video_paths="XD_Violence_testing.txt" output_path="./outputNEWtemp/random_dropout_3" on_extraction="save_numpy" audio_corruption_type="random_dropout" audio_severity=3
# CUDA_LAUNCH_BLOCKING=1 python main.py feature_type=vggish device="cpu" file_with_video_paths="XD_Violence_testing.txt" output_path="./outputNEWtemp/random_dropout_5" on_extraction="save_numpy" audio_corruption_type="random_dropout" audio_severity=5


##############################################################################
# DONE
##############################################################################
echo "All runs completed with overlay noise + other corruptions at severities 3 and 5!"
