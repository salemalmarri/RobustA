# RobustA: Robust Anomaly Detection in Multimodal Data

Official code for the paper: [arXiv:2511.07276](https://arxiv.org/abs/2511.07276)

RobustA proposes a multimodal anomaly detection method that learns a shared representation
space for different modalities and employs a dynamic weighting scheme during inference
based on the estimated level of corruption. Trained on clean RGB (I3D) + Audio (VGGish)
features from the XD-Violence dataset and evaluated under modality corruption.

## Installation

```bash
# Python 3.7+ required
pip install -r requirements.txt
```

## Data Preparation

### 1. Download Features

Download the XD-Violence I3D and VGGish features from the
[XD-Violence project page](https://roc-ng.github.io/XD-Violence/).

The corrupted modality features (RobustA dataset) will be made publicly available.

### 2. Organize Data

Place or symlink your data so the parent directory of this repo contains:

```
your-workspace/
  XD-Violence-V0.2/                   # Clean RGB (I3D) test features
  ownExtractedV0.0.2/                 # Clean Audio (VGGish) features
  video_features/                      # Corrupted audio features
  I3D-Corrupted-Features-Fixed-3/      # Corrupted RGB features (severity 3)
  I3D-Corrupted-Features-Fixed-5/      # Corrupted RGB features (severity 5)
  I3D-VAL/                             # Validation I3D features (for GMM)
  VGGISH-VAL/                          # Validation VGGish features (for GMM)
  robusta-method/                        # <-- this repo
```

### 3. Quick Setup (symlinks)

If your data lives elsewhere, use the provided helper:

```bash
bash setup_data.sh /path/to/your/data/root
```

This creates symlinks in the parent directory so the list files resolve correctly.

## Pre-trained Checkpoint

The pre-trained model checkpoint is at `ckpt/robusta_best.pkl`.

## Usage

### Training

```bash
bash run_train.sh
```

To enable Comet ML logging, set your API key:
```bash
export COMET_API_KEY="your-key-here"
```

### Evaluation (single experiment)

```bash
# Audio babble corruption, 100%, static weighting
python infer.py --single-branch --modality MIX2 \
    --corrupt-modality AUDIO --corruption-percentage 1.0 \
    --test_corrupted_audio_list ./list/babble_3.list \
    --alpha 0.5 --beta 0.5

# Same with GMM dynamic weighting
python infer_gmm.py --single-branch --modality MIX2 \
    --corrupt-modality AUDIO --corruption-percentage 1.0 \
    --test_corrupted_audio_list ./list/babble_3.list \
    --c 0.008 --x0 -500
```

### Full evaluation sweep

```bash
bash run_evaluate.sh
```

To use a specific Python environment:
```bash
PYTHON=/path/to/python bash run_evaluate.sh
```

## File Structure

### Core Model

| File | Purpose |
|---|---|
| `model.py` | RobustA model architecture |
| `layers.py` | GCN, attention layers |
| `main.py` | Training entry point |
| `train.py` | Training loop |
| `test.py` | Validation during training |
| `option.py` | All CLI arguments |
| `dataset2.py` | Training dataset loader |
| `utils.py` | Utility functions |

### Evaluation - Static Weighting

| Infer Script | Corruption | Dataset | Test Function |
|---|---|---|---|
| `infer.py` | Audio | `datasetCM.py` | `test2_vggish__.py` |
| `infer_rgb.py` | RGB | `datasetCM.py` | `test2__.py` |
| `inferBOTH.py` | Both | `datasetCMBOTH.py` | `test2_combined__.py` |

### Evaluation - GMM Dynamic Weighting

| Infer Script | Corruption | Dataset | Test Function |
|---|---|---|---|
| `infer_gmm.py` | Audio | `datasetCM.py` | `test2_vggish_.py` |
| `infer_rgb_gmm.py` | RGB | `datasetCM.py` | `test2_.py` |
| `inferBOTH_gmm.py` | Both | `datasetCMBOTH.py` | `test2_combined.py` |

### Scripts

| File | Purpose |
|---|---|
| `run_train.sh` | Train the model |
| `run_evaluate.sh` | Full sweep (all corruptions, percentages, static + GMM) |
| `run_evaluate_single.sh` | Quick single-experiment runner |
| `setup_data.sh` | Create data symlinks |
| `create_gmm.py` | Recreate GMM models from training data |

## GMM Models

Two pre-fitted GMM models are included for dynamic weighting:

**RGB** (`val_gmm_model_with_pca_128.joblib`):
- Source: I3D validation features (120 files)
- PCA: 2048-d to 128-d, then GMM with 5 components (full covariance)

**Audio** (`val_gmm_model_VGGISH-4-48-6.joblib`):
- Source: VGGish training features (48 randomly sampled files)
- GMM: 4 components (full covariance) on raw 128-d features

To recreate: `python create_gmm.py`

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--alpha` | 0.5 | Static weight for RGB modality |
| `--beta` | 0.5 | Static weight for Audio modality |
| `--c` | 0.008 | Sigmoid steepness for GMM dynamic weighting |
| `--x0` | -500 | Sigmoid midpoint offset |
| `--corruption-percentage` | 0.0 | Fraction of test samples to corrupt (0.0-1.0) |
| `--corrupt-modality` | AUDIO | Which modality to corrupt (AUDIO, RGB, or both) |

## Corruption Types

**Visual (RGB):** brightness, contrast, fog, rain, motion_blur, saturate, bit_error, shot_noise

**Audio:** babble, bitrate, hfchannel, pink, pitch_shift, random_dropout, reverb, white

## Citation

```bibtex
@article{almarri2025robusta,
  title={RobustA: Robust Anomaly Detection in Multimodal Data},
  author={AlMarri, Salem and Liaqat, Muhammad Irzam and Zaheer, Muhammad Zaigham and Nawaz, Shah and Nandakumar, Karthik and Schedl, Markus},
  journal={arXiv preprint arXiv:2511.07276},
  year={2025}
}
```
