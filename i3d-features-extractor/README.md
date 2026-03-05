# I3D Feature Extraction (RobustA)

This module extracts ResNet-I3D visual features from MP4 videos for the RobustA pipeline.

For each input video, it saves one `.npy` file (video-level temporal feature sequence).

## Quick Start

```bash
cd i3d-features-extractor

# 1) Download I3D pretrained weights (Caffe2 format)
wget https://dl.fbaipublicfiles.com/video-nonlocal/i3d_baseline_32x2_IN_pretrain_400k.pkl -P pretrained/

# 2) Convert to PyTorch checkpoint used by this code
python -m utils.convert_weights \
  pretrained/i3d_baseline_32x2_IN_pretrain_400k.pkl \
  pretrained/i3d_r50_kinetics.pth

# 3) Extract features
python main.py \
  --datasetpath ../XD-Violence-Corrupted-3 \
  --outputpath ../I3D-Corrupted-Features-Fixed-3 \
  --pretrainedpath pretrained/i3d_r50_kinetics.pth
```

Run the same command with `../XD-Violence-Corrupted-5` to extract severity-5 features.

## Arguments

- `--datasetpath`: input root folder (recursive search for `.mp4`)
- `--outputpath`: output folder for extracted `.npy` features
- `--pretrainedpath`: path to converted I3D checkpoint
- `--frequency`: frame stride between snippets (default: `16`)
- `--batch_size`: snippet batch size (default: `20`)
- `--sample_mode`: snippet sampling mode (default: `oversample`)

## Notes

- `ffmpeg` must be available in your environment (used to decode video frames).
- Existing output files are skipped automatically.

## Credits

Base components are adapted from:

- [pytorch-resnet3d](https://github.com/Tushar-N/pytorch-resnet3d)
- [pytorch-i3d-feature-extraction](https://github.com/Finspire13/pytorch-i3d-feature-extraction)
