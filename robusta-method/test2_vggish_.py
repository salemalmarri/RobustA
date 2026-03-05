from sklearn.metrics import auc, precision_recall_curve, average_precision_score
import numpy as np
import torch
import math  # or you can use torch.exp if you prefer

# Disable truncation so large arrays print fully:
np.set_printoptions(threshold=np.inf)

# ADDED:
from joblib import load
import os

def load_i3d_features(folder_path):
    """
    Load all .npy feature files from a given folder,
    and concatenate them into one array of shape [N, D].
    """
    all_features_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            filepath = os.path.join(folder_path, filename)
            features = np.load(filepath)
            if features.ndim == 1:
                features = features[np.newaxis, :]
            all_features_list.append(features)
    all_features = np.concatenate(all_features_list, axis=0)
    return all_features

def compute_val_lik_stats(gmm, folder_val="../VGGISH-VAL/val/vggish"):
    feats_val = load_i3d_features(folder_val)
    ll_array = gmm.score_samples(feats_val)
    mean_lik = np.mean(ll_array)
    std_lik = np.std(ll_array)
    return mean_lik, std_lik

def test(dataloader, model, device, gt, args):
    
    alpha = args.alpha
    beta = args.beta
    
    # print("hyperparameters are ", alpha, "=", beta)
    
    cm = args.corrupt_modality
    corrupt_modality = cm[0] if isinstance(cm, list) and cm else cm
    corruption_percentage = args.corruption_percentage  # in [0,1]

    # ADDED: Load the pre-trained VAL GMM once
    
    # gmm_val = load("val_gmm_model_VGGISH-1-6.joblib")
    
    gmm_val = load("val_gmm_model_VGGISH-4-48-6.joblib")
    # gmm_val = load("val_gmm_model_VGGISH-5-84-6.joblib")

    
    
    # gmm_val = load("val_gmm_model_VGGISH-1-4.joblib")
    # gmm_val = load("val_gmm_model_VGGISH-1-3.joblib")
    
    # ADDED: Compute real mean/std for log-likelihood on VAL
    val_folder_path = "../VGGISH-VAL/val/vggish"  # <-- If VAL is elsewhere, put the full path here
    val_mean_lik, val_std_lik = compute_val_lik_stats(gmm_val, val_folder_path)
    # print("Derived from VAL data => mean_lik:", val_mean_lik, "std_lik:", val_std_lik)
    
    # Now write filename, predictions, and these ground-truth labels
    filename_text = f"results_vggish_white_{corruption_percentage}.txt"
    with open(filename_text, "w") as f:
        f.write("Starting new results file\n")

    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(device)
        pred2 = torch.zeros(0).to(device)
        
        gt_index = 0  # global pointer
        
        for i, batch_data in enumerate(dataloader):
            # You already redefine alpha & beta here, but that’s fine:
            alpha = args.alpha
            beta = args.beta
            
            x0 = args.x0
            c = args.c 

            # ADDED: Default log-likelihood to 0 unless we have features
            log_lik1 = 0.0
            
            # Check how many values are returned
            if len(batch_data) == 2:
                # Format: (features, filename)
                features, filename = batch_data
                features = features.to(device)
                logits, logits2 = model(inputs=features, seq_len=None)

            elif len(batch_data) == 3:
                # Format: (features1, features2, filename)
                features1, features2, filename = batch_data
                features1 = features1.to(device)
                features2 = features2.to(device)

                #
                # -- OLD CODE (commented out) where we took log-likelihood from features1: --
                #
                # features1_np = features1.cpu().numpy()  # shape e.g. [B, T, D]
                # if features1_np.ndim == 3:  # typical case
                #     Bf1, Tf1, Df1 = features1_np.shape
                #     features1_2d = features1_np.reshape(-1, Df1)
                # else:
                #     features1_2d = features1_np
                #
                # log_lik1 = gmm_val.score(features1_2d)
                # z_lik1 = (log_lik1 - val_mean_lik) / (val_std_lik + 1e-9)

                #
                # -- NEW CODE (log_likelihood from features2 instead): --
                #
                features2_np = features2.cpu().numpy()  # shape e.g. [B, T, D]
                if features2_np.ndim == 3:  # typical case
                    Bf2, Tf2, Df2 = features2_np.shape
                    features2_2d = features2_np.reshape(-1, Df2)
                else:
                    features2_2d = features2_np

                log_lik1 = gmm_val.score(features2_2d)
                # print("Raw log-likelihood for features2:", log_lik1)

                z_lik1 = (log_lik1 - val_mean_lik) / (val_std_lik + 1e-9)

                # --------------------------------------------------------------
                # OLD MIN–MAX approach => [0,1] (Commented out now):
                # --------------------------------------------------------------
                # ell_min = val_mean_lik - 4 * val_std_lik
                # ell_max = val_mean_lik + 4 * val_std_lik
                # similarity = 1.0 - (log_lik1 - ell_min) / (ell_max - ell_min)
                # similarity = max(0.0, min(1.0, similarity))
                # similarity = 0.5 * similarity

                # --------------------------------------------------------------
                # NEW "Code 1 style" approach => Sigmoid-based similarity:
                # --------------------------------------------------------------
                similarity = 0.5 / (1 + np.exp(c * (log_lik1 - (-x0))))
                # print("Similarity from Code1's formula: ", similarity)

                # Project and combine audio features
                B, T, D = features2.shape
                features2 = features2.view(B*T, D)
                features2 = model.audio_proj(features2)
                features2 = features2.view(B, T, 2048)

                combined_features = torch.cat([features1, features2], dim=0)
                logits, logits2 = model(inputs=combined_features, seq_len=None)

            else:
                raise ValueError("Unexpected number of items returned by dataset")

            logits = torch.squeeze(logits)
            sig = torch.sigmoid(logits)
            sig = torch.mean(sig, 0)
            
            if len(batch_data) == 3:
                B_total = logits.size(0)
                half = B_total // 2
                sig1 = torch.sigmoid(logits[:half])
                sig2_ = torch.sigmoid(logits[half:])
                sig1_mean = sig1.mean(0)
                sig2_mean = sig2_.mean(0)

                if corrupt_modality == 'AUDIO':
                    w_audio = similarity #0.5 * math.exp(-alpha * corruption_percentage)
                    w_rgb = 1.0 - w_audio
                    sig = w_rgb * sig1_mean + w_audio * sig2_mean
                elif corrupt_modality == 'RGB':
                    w_rgb = similarity  # previously: 0.5 * math.exp(-beta * corruption_percentage)
                    w_audio = 1.0 - w_rgb
                    sig = w_rgb * sig1_mean + w_audio * sig2_mean
                else:
                    sig = alpha * sig1_mean + beta * sig2_mean

            pred = torch.cat((pred, sig))
            
            # Convert to CPU NumPy array
            sig_np = sig.cpu().numpy()

            # Repeat each value 16 times
            sig_repeated = np.repeat(sig_np, 16)
            
            chunk_size = len(sig_repeated)
            
            # Get the slice of gt that corresponds to these predictions
            gt_chunk = gt[gt_index : gt_index + chunk_size]
            gt_index += chunk_size
            

            with open(filename_text, "a") as f:
                f.write(f"Filename: [{filename}]\n")
                gt_str = ", ".join(map(str, gt_chunk))
                f.write(f"Primary GT slice: {gt_str}\n")
                sig_str = ", ".join(map(str, sig_repeated))
                f.write(f"Primary preds repeated: {sig_str}\n")

            logits2 = torch.squeeze(logits2)
            sig2 = torch.sigmoid(logits2)
            sig2 = torch.mean(sig2, 0)

            if len(batch_data) == 3:
                B_total2 = logits2.size(0)
                half2 = B_total2 // 2
                sig1_2 = torch.sigmoid(logits2[:half2])
                sig2_2 = torch.sigmoid(logits2[half2:])
                sig1_mean2 = sig1_2.mean(0)
                sig2_mean2 = sig2_2.mean(0)

                if corrupt_modality == 'AUDIO':
                    w_audio = 0.5 * math.exp(-alpha * corruption_percentage)
                    w_rgb = 1.0 - w_audio
                    sig2 = w_rgb * sig1_mean2 + w_audio * sig2_mean2
                elif corrupt_modality == 'RGB':
                    w_rgb = 0.5 * math.exp(-beta * corruption_percentage)
                    w_audio = 1.0 - w_rgb
                    sig2 = w_rgb * sig1_mean2 + w_audio * sig2_mean2
                else:
                    sig2 = alpha * sig1_mean2 + beta * sig2_mean2

            sig2 = torch.unsqueeze(sig2, 1)
            pred2 = torch.cat((pred2, sig2))

        pred = pred.cpu().detach().numpy()
        pred2 = pred2.cpu().detach().numpy()

        precision, recall, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(recall, precision)
        ap1 = average_precision_score(list(gt), np.repeat(pred, 16))

        precision, recall, _ = precision_recall_curve(list(gt), np.repeat(pred2, 16))
        pr_auc2 = auc(recall, precision)
        ap2 = average_precision_score(list(gt), np.repeat(pred2, 16))

        return pr_auc, pr_auc2, ap1, ap2
