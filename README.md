# RobustA: Robust Anomaly Detection in Multimodal Data

Official repository for **RobustA: Robust Anomaly Detection in Multimodal Data**, a comprehensive benchmark for evaluating anomaly detection robustness across multimodal data with various corruption types. This repository includes corrupted video datasets, pre-extracted features, and tools for feature extraction and corruption generation. For more details, refer to the paper: https://arxiv.org/abs/2511.07276

<div align="center">
<img src="doc/medras-noborder.png" width="400" alt="RobustA Framework">
<p><em>Examples of few visual and audio corruptions demonstrating the challenging scenarios presented in RobustA.</em></p>
</div>

## Table of Contents
- [RobustA Benchmark](#robusta-benchmark)
  - [RobustA MP4 Files](#robusta-mp4-files)
  - [RobustA I3D Features](#robusta-i3d-features)
  - [RobustA VGGISH Features](#robusta-vggish-features)
- [Video Corruptions](#video-corruptions)
- [Feature Extraction](#feature-extraction)
  - [I3D Feature Extraction](#i3d-feature-extraction)
  - [Video Features](#video-features)
- [Citation](#citation)

---
## RobustA Benchmark

This benchmark provides datasets for evaluating anomaly detection robustness across various corruption types in multimodal data. The original datasets and related resources are available at the [XD-Violence](https://roc-ng.github.io/XD-Violence/) repository. Below are the download links for each corruption type and model variant.

---
### RobustA MP4 Files

- **Bit Error:** [Download](https://www.dropbox.com/scl/fo/1fbktl0zltmh7d97tbqfc/AJpeJK-oTkcLFikHHeQ3Wrc?rlkey=835kovmcok5cwae5cdmbvi8va&st=nzf12lh7&dl=0)
- **Brightness:** [Download](https://www.dropbox.com/scl/fo/npgksz1yp5banfixlbm8u/AOCt6Ly0tunprJDkTF7YjlU?rlkey=6kw3osq1vm5h4ae4fd4xxywjk&st=rznwndwu&dl=0)
- **Contrast:** [Download](https://www.dropbox.com/scl/fo/582ze4eyvynndyiz6776n/AJ_S52cxluW6jEWIQy0_oxA?rlkey=qr91sp1r8ptp36fxv3vtlskfi&st=i5a19sb1&dl=0)
- **Fog:** [Download](https://www.dropbox.com/scl/fo/djmvu8cnhv9136dbjz3b7/AN3dAfPcGgmBwSTMaKkxl1Q?rlkey=sxj6haqykztkuvsj23nh31e08&st=s1ohh947&dl=0)
- **Motion Blur:** [Download](https://www.dropbox.com/scl/fo/p13jzhy6oua1bt7ro822n/ANT1iVrj1nvToohoP7sBdIE?rlkey=ijk3b4ayqmu119ibcbgwqpkkl&st=yeosywen&dl=0)
- **Rain:** [Download](https://www.dropbox.com/scl/fo/pzsju2adhxsgxxx4lml3k/AOIXlQrpFkjjfN9oxgp2kO4?rlkey=868znff3m1ztuad2cquuqjv4o&st=iddtfr0e&dl=0)
- **Saturate:** [Download](https://www.dropbox.com/scl/fo/pzsju2adhxsgxxx4lml3k/AOIXlQrpFkjjfN9oxgp2kO4?rlkey=868znff3m1ztuad2cquuqjv4o&st=iddtfr0e&dl=0)
- **Shot Noise:** [Download](https://www.dropbox.com/scl/fo/f2ir2w72qdb5qgeamvmax/AOiXJWjuuZ81uN75oE4zBvw?rlkey=gwjg4h2q5j8zf5luj1p30qz7h&st=ve31o2sz&dl=0)

---

### RobustA I3D Features

- **Bit Error:** [Download](https://www.dropbox.com/scl/fo/jy2l2r97nffd2ak36j8u1/AO0sbGpweJ0BXNL975BOz8s?rlkey=23sy9fnygmb9ez3t1mxsgnxwa&st=y2pywiqo&dl=0)
- **Brightness:** [Download](https://www.dropbox.com/scl/fo/692xjhhkzfhs207bfvoiw/AEnXedUEKv_RsGmeZcymzR4?rlkey=i25atw1xecfoygrc926ygq4zf&st=p7nuk0nm&dl=0)
- **Contrast:** [Download](https://www.dropbox.com/scl/fo/6f99or7wuy7ib41ujlh96/ABXhLugKYlEbzPrq2dqJEUo?rlkey=gyhet5t63vpbc67eiy4j2kz44&st=uiu5ea5b&dl=0)
- **Fog:** [Download](https://www.dropbox.com/scl/fo/x6ufc49d2hvm1qqezjine/AN5CVRi8LUjELMsbZOl5vTE?rlkey=un55ev45drtczmfi7k2ezcszw&st=j1dj35wf&dl=0)
- **Motion Blur:** [Download](https://www.dropbox.com/scl/fo/5s1bj85jl0d0byr4rvjch/AOUUx7jH05MVPaEsx3p9OP4?rlkey=qxxoray25ab8py6ta35e01dul&st=nbs2w0oj&dl=0)
- **Rain:** [Download](https://www.dropbox.com/scl/fo/xbzn24uvhfdwohmqpd1v7/ADyFTIJNW0_e-NEoJA_q9hE?rlkey=qdflp5h2prm542t5t2qfur98f&st=zs3e6d6d&dl=0)
- **Saturate:** [Download](https://www.dropbox.com/scl/fo/tbfwhaunbhhuaeokcaf5e/AEYH7CGxV2xDeHWygbkK-uk?rlkey=f31sy3ciy6rztztdfo8i0l41f&st=r1495njt&dl=0)
- **Shot Noise:** [Download](https://www.dropbox.com/scl/fo/tbfwhaunbhhuaeokcaf5e/AEYH7CGxV2xDeHWygbkK-uk?rlkey=f31sy3ciy6rztztdfo8i0l41f&st=r1495njt&dl=0)

---

### RobustA VGGISH Features

- **Babble:** [Download](https://www.dropbox.com/scl/fo/inqyoa68a01j2yenwbmrt/ALKzSJ7S9HHb81--tQAx9jk?rlkey=d32kk0wiiaegk8x6zuhnbetg4&st=n7ylwndw&dl=0)
- **Bitrate:** [Download](https://www.dropbox.com/scl/fo/cm3dlhu7lan8zk6jl46gl/AIrFPTPJBCZ8VYbJJT5HqkI?rlkey=cqeb8flypj1f5g8sttvtrj4h7&st=4iwlhahn&dl=0)
- **HFChannel:** [Download](https://www.dropbox.com/scl/fo/gaft2eo66dewy4rgk6d2c/AGlS6Oksjig5TtjC84QKHx8?rlkey=amjetu6s8t298z35abh5gjpun&st=da7xlrhm&dl=0)
- **Pink:** [Download](https://www.dropbox.com/scl/fo/5ijmoyv8m5zz2vboibysd/AAuFtWFJvkFKR8Zl1KGdKlQ?rlkey=xir2tz173il4eatduvg16bwym&st=mlpw7td5&dl=0)
- **Pitch Shift:** [Download](https://www.dropbox.com/scl/fo/b95ji6qilatwzlz07w9gu/AFRGdNMFSUTlP-tfSWdj2eQ?rlkey=cep2nedhxam0xyn94dhiux954&st=m84e8hfg&dl=0)
- **Random Dropout:** [Download](https://www.dropbox.com/scl/fo/rjefn17uilb049n5tq2vh/APZBNF-rH8hdG1CKrBPajb0?rlkey=0w5apn1rgevb71qriouljgogq&st=pcn7q5go&dl=0)
- **Reverb:** [Download](https://www.dropbox.com/scl/fo/ugsvoojjx47bkkv1ey7d8/ABZsHXR0QfnLmNAmOCv--dc?rlkey=vtzzpf0lkgj91jdx8fwk4xreu&st=q5di2lbw&dl=0)
- **White:** [Download](https://www.dropbox.com/scl/fo/lvhj8i8sxobt5arhql7eh/AMHziHedD9vqIzLjmIWFjm0?rlkey=7nk3a7t4c3mzi0slyetwduyrm&st=26uaymyx&dl=0)

---
## Video Corruptions

To apply various corruption types (e.g., brightness, contrast, fog, motion blur, rain, etc.) to videos for robustness evaluation, refer to the [`video-corruption`](video-corruption) directory. This directory contains code and tools for generating corrupted video datasets. See the [`README.md`](video-corruption/README.md) file for detailed instructions on corruption generation, dataset creation, and evaluation benchmarks.

---

## Feature Extraction

### I3D Feature Extraction

To extract I3D features from videos, use the [`i3d-features-extractor`](i3d-features-extractor) directory. This tool supports feature extraction for both clean and corrupted videos. Refer to the [`README.md`](i3d-features-extractor/README.md) file for detailed installation and usage instructions.

---
### Video Features

For comprehensive feature extraction across multiple models (S3D, R(2+1)d, I3D, VGGish, CLIP, TIMM, and RAFT), refer to the [`video_features`](video_features) directory. This tool supports extraction of visual features, optical flow, and audio features. Consult the [`README.md`](video_features/README.md) file for detailed documentation on each extraction method, model configurations, dependencies, and usage instructions.

---
## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@article{AlMarri2025RobustA,
  author = {AlMarri, S. and Liaqat, M. I. and Zaheer, M. Z. and Nawaz, S. and Nandakumar, K. and Schedl, M.},
  title = {RobustA: Robust Anomaly Detection in Multimodal Data},
  journal = {arXiv preprint arXiv:2511.07276},
  year = {2025}
}
```

If you need additional modalities (audio, etc.) or want to add train/test set separation, let me know!
