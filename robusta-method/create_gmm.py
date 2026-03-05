"""
Recreate the GMM models used for dynamic weighting in the paper.

RGB GMM:  PCA(2048->128) + GMM(5 components, full covariance)
          Source: ../I3D-VAL/VAL/ (120 files)
          Output: val_gmm_model_with_pca_128.joblib  -> tuple(pca, gmm)

Audio GMM: GMM(4 components, full covariance) on raw 128-d VGGish
           Source: ../ownExtractedV0.0.2/VGGISHTrain/ (48 of 3954 files)
           Output: val_gmm_model_VGGISH-4-48-6.joblib  -> gmm

Usage:
    python create_gmm.py              # recreate both
    python create_gmm.py --only rgb   # just RGB
    python create_gmm.py --only audio # just Audio
"""

import os
import random
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from joblib import dump


def load_features(folder_path, max_files=None, seed=42):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    if max_files and max_files < len(files):
        random.seed(seed)
        files = random.sample(files, max_files)
    all_feats = []
    for fname in files:
        arr = np.load(os.path.join(folder_path, fname))
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        all_feats.append(arr)
    print(f"  Loaded {len(files)} files from {folder_path}, total frames: {sum(a.shape[0] for a in all_feats)}")
    return np.concatenate(all_feats, axis=0)


def create_rgb_gmm(folder="../I3D-VAL/VAL", n_components=5, pca_dim=128,
                    reg_covar=1e-4, seed=42, output="val_gmm_model_with_pca_128.joblib"):
    print("Creating RGB GMM...")
    feats = load_features(folder)
    print(f"  Feature shape: {feats.shape}")

    pca = PCA(n_components=pca_dim, random_state=seed)
    feats_pca = pca.fit_transform(feats)
    print(f"  PCA: {feats.shape[1]}d -> {pca_dim}d, explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                           reg_covar=reg_covar, random_state=seed)
    gmm.fit(feats_pca)
    print(f"  GMM: {n_components} components, converged={gmm.converged_}")

    dump((pca, gmm), output)
    print(f"  Saved to {output}")


def create_audio_gmm(folder="../ownExtractedV0.0.2/VGGISHTrain", n_components=4,
                      max_files=48, reg_covar=1e-6, seed=42,
                      output="val_gmm_model_VGGISH-4-48-6.joblib"):
    print("Creating Audio GMM...")
    feats = load_features(folder, max_files=max_files, seed=seed)
    print(f"  Feature shape: {feats.shape}")

    gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                           reg_covar=reg_covar, random_state=seed)
    gmm.fit(feats)
    print(f"  GMM: {n_components} components, converged={gmm.converged_}")

    dump(gmm, output)
    print(f"  Saved to {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['rgb', 'audio'], default=None)
    parser.add_argument('--rgb-folder', default='../I3D-VAL/VAL')
    parser.add_argument('--audio-folder', default='../ownExtractedV0.0.2/VGGISHTrain')
    parser.add_argument('--rgb-gmm-components', type=int, default=5)
    parser.add_argument('--audio-gmm-components', type=int, default=4)
    parser.add_argument('--audio-max-files', type=int, default=48)
    args = parser.parse_args()

    if args.only != 'audio':
        create_rgb_gmm(folder=args.rgb_folder, n_components=args.rgb_gmm_components)
    if args.only != 'rgb':
        create_audio_gmm(folder=args.audio_folder, n_components=args.audio_gmm_components,
                          max_files=args.audio_max_files)
