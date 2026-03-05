## RobustA Quick Start (VGGish + Audio Corruptions)

This folder supports both:

- clean VGGish extraction
- corrupted VGGish extraction (RobustA corruption protocol)

```bash
cd video_features
```

One-time setup:

1. Download NoiseX-92 and place it at `./NoiseX-92`
2. Download RobustA models folder and place it at `./models`

### A) Clean VGGish extraction

Use this when you want clean (non-corrupted) VGGish features:

```bash
python main.py \
  feature_type=vggish \
  device="cpu" \
  file_with_video_paths="XD_Violence_testing.txt" \
  output_path="./output_clean/vggish" \
  on_extraction="save_numpy"
```

### B) Corrupted VGGish extraction (single corruption command template)

Use this template for one corruption at a time:

```bash
# Non-overlay corruption template:
python main.py \
  feature_type=vggish \
  device="cpu" \
  file_with_video_paths="XD_Violence_testing.txt" \
  output_path="./output_corrupted/<CORRUPTION_NAME>_<SEVERITY>" \
  on_extraction="save_numpy" \
  audio_corruption_type="<CORRUPTION_NAME>" \
  audio_severity=<SEVERITY>
```

Example:

```bash
python main.py \
  feature_type=vggish \
  device="cpu" \
  file_with_video_paths="XD_Violence_testing.txt" \
  output_path="./output_corrupted/reverb_3" \
  on_extraction="save_numpy" \
  audio_corruption_type="reverb" \
  audio_severity=3
```

Overlay template (`white`, `pink`, `hfchannel`, `babble`):

```bash
python main.py \
  feature_type=vggish \
  device="cpu" \
  file_with_video_paths="XD_Violence_testing.txt" \
  output_path="./output_corrupted/<OVERLAY_NAME>_<SEVERITY>" \
  on_extraction="save_numpy" \
  audio_corruption_type="overlay" \
  overlay_audio_type="<OVERLAY_NAME>" \
  audio_severity=<SEVERITY>
```

### C) Batch corrupted extraction with `.sh`

Use `run_ALL.sh` to run all configured corruption commands in batch:

```bash
bash run_ALL.sh
```

The default script covers:

- Overlay family: `white`, `pink`, `hfchannel`, `babble`
- Other corruptions: `bitrate`, `pitch_shift`, `reverb`, `random_dropout`

### Paper-to-Code Naming (Audio)

- **Overlay** (paper umbrella term) -> `white`, `pink`, `hfchannel`, `babble`
- **Bit Error / compression-style corruption** -> `bitrate`
- **Pitch Shift** -> `pitch_shift`
- **Random Dropout** -> `random_dropout`
- **Reverb** -> `reverb`

## Required Downloads

Before using this repository, please download the following resources and place them in the appropriate locations.

### 1. NoiseX-92 Dataset

Download **NoiseX-92** from the link below:

- NoiseX-92:  
  https://www.dropbox.com/scl/fo/tn8prgq4zlw830u8tpqdu/AIrWOY326HNpFDKwXPt_sFM?rlkey=cy1ds9u3hp5k0z4bsjdym083k&dl=0

After downloading, extract the contents and place the folder at `video_features/NoiseX-92`.

### 2. Pretrained Models Folder

Download the **models** folder from the link below:

- Models folder:  
  https://www.dropbox.com/scl/fo/28igwa06aif0fnsbyqhe4/APSO91eCkvbieg1zciZKPgw?rlkey=6mqz1kddm5af4ooecjgp2v2ck&dl=0

After downloading, extract and place the folder at `video_features/models`.

---

<div align="center">

# Video Features

<img  src="https://github.com/v-iashin/v-iashin.github.io/raw/master/images/video_features/vid_feats.gif" width="300" />

`video_features` allows you to extract features from video clips.
It supports a variety of extractors and modalities,
i.e. visual appearance, optical flow, and audio.
See more details in [Documentation](https://v-iashin.github.io/video_features/).
</div>

## Supported Models

Action Recognition

- [S3D (Kinetics 400)](https://v-iashin.github.io/video_features/models/s3d)
- [R(2+1)d RGB (IG-65M, Kinetics 400)](https://v-iashin.github.io/video_features/models/r21d)
- [I3D-Net RGB + Flow (Kinetics 400)](https://v-iashin.github.io/video_features/models/i3d)

Sound Recognition

- [VGGish (AudioSet)](https://v-iashin.github.io/video_features/models/vggish)

Optical Flow

- [RAFT (FlyingChairs, FlyingThings3D, Sintel, KITTI)](https://v-iashin.github.io/video_features/models/raft)

Frame-wise Features

- [All models from TIMM e.g. ViT, ConvNeXt, EVA, Swin, DINO (ImageNet, LAION, etc)](https://v-iashin.github.io/video_features/models/timm)
- [CLIP](https://v-iashin.github.io/video_features/models/clip)
- [ResNet-18,34,50,101,152 (ImageNet)](https://v-iashin.github.io/video_features/models/resnet)


## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1csJgkVQ3E2qOyVlcOM-ACHGgPBBKwE2Y?usp=sharing)

or run with `conda` locally:
```bash
# clone the repo and change the working directory
git clone https://github.com/v-iashin/video_features.git
cd video_features

# install environment
conda env create -f conda_env.yml

# load the environment
conda activate video_features

# extract r(2+1)d features for the sample videos
python main.py \
    feature_type=r21d \
    device="cuda:0" \
    video_paths="[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"

# if you have many GPUs, just run this command from another terminal with another device
# device can also be "cpu"
```

If you are more comfortable with Docker, there is a
Docker image with a pre-installed environment that supports all models.
Check out the
[Docker support](https://v-iashin.github.io/video_features/meta/docker).
documentation page.


## Multi-GPU and Multi-Node Setups

With `video_features`, it is easy to parallelize feature extraction among many GPUs.
It is enough to start the script in another terminal with another GPU (or even the same one)
pointing to the same output folder and input video paths.
The script will check if the features already exist and skip them.
It will also try to load the feature file to check if it is corrupted (i.e. not openable).
This approach allows you to continue feature extraction if the previous script failed for some reason.

If you have an access to a GPU cluster with shared disk space you may scale
extraction with as many GPUs as you can by creating several single-GPU jobs with the same command.

Since each time the script is run the list of input files is shuffled, you don't need to worry that
workers will be processing the same video.
On a rare occasion when the collision happens, the script will rewrite previously extracted features.


## Input
The inputs are paths to video files.
Paths can be passed as a list of paths to videos or as a text file formatted with a single path per line.

## Output
Output is defined by the `on_extraction` argument; by default it prints the features to the command line.
Possible values of output are `['print', 'save_numpy', 'save_pickle']`. `save_*` options save the features in
the `output_path` folder with the same name as the input video file but with the `.npy` or `.pkl` extension.

## Used in

* [SpecVQGAN](https://arxiv.org/abs/2110.08791) branch `specvqgan`
* [BMT](https://arxiv.org/abs/2005.08271) branch `bmt`
* [MDVC](https://arxiv.org/abs/2003.07758) branch `mdvc`

Please, let me know if you found this repo useful for your projects or papers.


## Acknowledgements

- [@Kamino666](https://github.com/Kamino666): added CLIP model as well as Windows and CPU support (and many other useful things).
- [@ohjho](https://github.com/ohjho): added support of 37-layer R(2+1)d favors.
- [@borijang](https://github.com/borijang): for solving bugs with file names, I3D checkpoint loading enhancement and code style improvements.
- [@bjuncek](https://github.com/bjuncek): for helping with timm models and offline discussion.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{videofeatures2020,
  title = {Video Features},
  author = {Vladimir Iashin and other contributors},
  year = {2020},
  howpublished = {\url{https://github.com/v-iashin/video_features}},
}
```
