# Visual Corruption Generation (RobustA)

This folder contains the visual corruption pipeline used in:

- RobustA: Robust Anomaly Detection in Multimodal Data ([arXiv:2511.07276](https://arxiv.org/abs/2511.07276))

The benchmark framework from the NeurIPS 2021 corruption benchmark is included for reference.  
For RobustA, the main scripts are:

- `corruptAllV2-3.py` (severity 3)
- `corruptAllV2-5.py` (severity 5)

## Quick Start

```bash
cd video-corruption
pip install -r requirements.txt
```

Edit paths in each script:

- `input_folder`: clean XD-Violence MP4 folder
- `output_folder`: where corrupted MP4 videos will be written

Run:

```bash
python corruptAllV2-3.py
python corruptAllV2-5.py
```

## Corruption Names (Paper vs Code)

Visual corruptions in the paper:

- Bit Error
- Brightness
- Contrast
- Fog
- Rain
- Motion Blur
- Saturation
- Shot Noise

Code/list names used in this repository:

- `bit_error`
- `brightness`
- `contrast`
- `fog`
- `rain`
- `motion_blur`
- `saturate` (corresponds to paper term **Saturation**)
- `shot_noise`

## Notes

- Scripts process MP4 videos and save corrupted MP4 outputs.
- `ffmpeg` must be installed and available in PATH.
- Generated corrupted videos are later used by `i3d-features-extractor` to extract visual features.
