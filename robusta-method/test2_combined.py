# test2_combined.py

from sklearn.metrics import auc, precision_recall_curve, average_precision_score
from joblib import load
import numpy as np
import torch
import os
import math

# Set global PCA for I3D
pca_rgb = None

def load_features_from_folder(folder_path):
    all_feats = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.npy'):
            fpath = os.path.join(folder_path, fname)
            arr = np.load(fpath)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            all_feats.append(arr)
    return np.concatenate(all_feats, axis=0)

def compute_stats(gmm, feats, apply_pca=None):
    if apply_pca is not None:
        feats = apply_pca.transform(feats)
    ll = gmm.score_samples(feats)
    return np.mean(ll), np.std(ll)

def test(dataloader, model, device, gt, args):
    global pca_rgb

    alpha = args.alpha
    beta = args.beta
    c = args.c
    x0 = args.x0

    corruption_percentage = args.corruption_percentage
    corrupt_modality = args.corrupt_modality  # 'AUDIO', 'RGB', or both handled jointly

    # Load GMMs
    pca_rgb, gmm_rgb = load("val_gmm_model_with_pca_128.joblib")
    gmm_audio = load("val_gmm_model_VGGISH-4-48-6.joblib")

    # Compute stats
    mean_rgb, std_rgb = compute_stats(gmm_rgb, load_features_from_folder("../I3D-VAL/VAL"), pca_rgb)
    mean_audio, std_audio = compute_stats(gmm_audio, load_features_from_folder("../VGGISH-VAL/val/vggish"))

    pred, pred2 = torch.zeros(0).to(device), torch.zeros(0).to(device)
    gt_index = 0
    results_file = f"results_combined_{corruption_percentage}.txt"
    with open(results_file, "w") as f: f.write("Starting results...\n")

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            if len(batch) == 3:
                features_rgb, features_audio, filename = batch
                features_rgb = features_rgb.to(device)
                features_audio = features_audio.to(device)

                # RGB log-likelihood
                rgb_np = features_rgb.cpu().numpy()
                if rgb_np.ndim == 3:
                    B, T, D = rgb_np.shape
                    rgb_np = rgb_np.reshape(-1, D)
                rgb_np_pca = pca_rgb.transform(rgb_np)
                log_lik_rgb = gmm_rgb.score(rgb_np_pca)
                sim_rgb = 0.5 / (1 + np.exp(c * (log_lik_rgb + x0)))

                # AUDIO log-likelihood
                audio_np = features_audio.cpu().numpy()
                if audio_np.ndim == 3:
                    B, T, D = audio_np.shape
                    audio_np = audio_np.reshape(-1, D)
                log_lik_audio = gmm_audio.score(audio_np)
                sim_audio = 0.5 / (1 + np.exp(c * (log_lik_audio + x0)))

                # Project AUDIO
                B, T, D = features_audio.shape
                features_audio = features_audio.view(B*T, D)
                features_audio = model.audio_proj(features_audio)
                features_audio = features_audio.view(B, T, 2048)

                features_combined = torch.cat([features_rgb, features_audio], dim=0)
                logits, logits2 = model(inputs=features_combined, seq_len=None)
            else:
                raise ValueError("Expected batch with both RGB and AUDIO")

            logits = torch.squeeze(logits)
            sig = torch.sigmoid(logits)

            B_total = logits.size(0)
            half = B_total // 2
            sig_rgb = sig[:half].mean(0)
            sig_audio = sig[half:].mean(0)

            # Fuse using dynamic weights from similarities
            w_rgb = sim_rgb
            w_audio = sim_audio
            norm_factor = w_rgb + w_audio + 1e-9
            w_rgb /= norm_factor
            w_audio /= norm_factor
            fused_sig = w_rgb * sig_rgb + w_audio * sig_audio
            pred = torch.cat((pred, fused_sig))

            # Print to file
            sig_np = fused_sig.cpu().numpy()
            sig_repeated = np.repeat(sig_np, 16)
            gt_chunk = gt[gt_index:gt_index + len(sig_repeated)]
            gt_index += len(sig_repeated)

            with open(results_file, "a") as f:
                f.write(f"\nFilename: {filename}\n")
                f.write(f"Primary GT slice: {', '.join(map(str, gt_chunk))}\n")
                f.write(f"Primary preds repeated: {', '.join(map(str, sig_repeated))}\n")

            # Second branch logits
            logits2 = torch.squeeze(logits2)
            sig2 = torch.sigmoid(logits2)

            sig2_rgb = sig2[:half].mean(0)
            sig2_audio = sig2[half:].mean(0)

            fused_sig2 = w_rgb * sig2_rgb + w_audio * sig2_audio
            fused_sig2 = torch.unsqueeze(fused_sig2, 1)
            pred2 = torch.cat((pred2, fused_sig2))

        # Final scores
        pred_np = pred.cpu().numpy()
        pred2_np = pred2.cpu().numpy()

        precision, recall, _ = precision_recall_curve(list(gt), np.repeat(pred_np, 16))
        pr_auc = auc(recall, precision)
        ap1 = average_precision_score(list(gt), np.repeat(pred_np, 16))

        precision, recall, _ = precision_recall_curve(list(gt), np.repeat(pred2_np, 16))
        pr_auc2 = auc(recall, precision)
        ap2 = average_precision_score(list(gt), np.repeat(pred2_np, 16))

        return pr_auc, pr_auc2, ap1, ap2
