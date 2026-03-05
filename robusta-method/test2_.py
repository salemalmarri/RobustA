from sklearn.metrics import auc, precision_recall_curve, average_precision_score
import numpy as np
import torch
import math  # or you can use torch.exp if you prefer

from joblib import load
import os

# If you want to disable truncation in printed NumPy arrays:
# np.set_printoptions(threshold=np.inf)

# ADDED: We'll keep a global reference to PCA so we don't alter function signatures
pca = None

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

def compute_val_lik_stats(gmm, folder_val="../I3D-VAL/VAL"):
    feats_val = load_i3d_features(folder_val)

    # If PCA is loaded, transform the VAL features before scoring
    if pca is not None:
        feats_val = pca.transform(feats_val)

    ll_array = gmm.score_samples(feats_val)
    mean_lik = np.mean(ll_array)
    std_lik = np.std(ll_array)
    return mean_lik, std_lik

def test(dataloader, model, device, gt, args):
    
    alpha = args.alpha
    beta = args.beta
    ell_multi = args.ell_multi

    cm = args.corrupt_modality
    corrupt_modality = cm[0] if isinstance(cm, list) and cm else cm
    corruption_percentage = args.corruption_percentage  # in [0,1]

    # Load the pre-trained PCA + GMM from file
    global pca
    pca, gmm_val = load("val_gmm_model_with_pca_128.joblib")

    # Compute real mean/std for log-likelihood on VAL
    val_folder_path = "../I3D-VAL/VAL"
    val_mean_lik, val_std_lik = compute_val_lik_stats(gmm_val, val_folder_path)

    with torch.no_grad():
        model.eval()
        
        # We'll accumulate predictions in pred, pred2 if needed
        pred = torch.zeros(0).to(device)
        pred2 = torch.zeros(0).to(device)

        # This pointer tracks how many GT labels we've used so far
        gt_index = 0
        
        # Construct the output file name with corruption_percentage
        filename_text = f"results_vision_brightness_{corruption_percentage}.txt"
        # First, open in WRITE mode once to overwrite any old file:
        with open(filename_text, "w") as f:
            f.write("Starting new results file\n")

        # Now process each batch
        for i, batch_data in enumerate(dataloader):
            alpha = args.alpha
            beta = args.beta
            x0 = args.x0
            c = args.c 

            # Default log-likelihood to 0 unless we have features
            log_lik1 = 0.0

            # Check how many items are returned
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

                # Flatten & transform features1 through PCA for GMM scoring
                features1_np = features1.cpu().numpy()  # [B, T, D] or [T, D]
                if features1_np.ndim == 3:  
                    Bf1, Tf1, Df1 = features1_np.shape
                    features1_2d = features1_np.reshape(-1, Df1)
                else:
                    features1_2d = features1_np

                features1_2d_pca = pca.transform(features1_2d)
                log_lik1 = gmm_val.score(features1_2d_pca)
                z_lik1 = (log_lik1 - val_mean_lik) / (val_std_lik + 1e-9)

                # Example min–max approach or Sigmoid-based approach
                ell_min = (val_mean_lik - 4 * val_std_lik) * ell_multi
                ell_max = val_mean_lik + 4 * val_std_lik

                # Example Sigmoid-based similarity
                similarity = 0.5 / (1 + np.exp(c * (log_lik1 - (-x0))))

                # Project features2
                B, T, D = features2.shape
                features2 = features2.view(B*T, D)
                features2 = model.audio_proj(features2)
                features2 = features2.view(B, T, 2048)

                # Combine
                combined_features = torch.cat([features1, features2], dim=0)
                logits, logits2 = model(inputs=combined_features, seq_len=None)
            else:
                raise ValueError("Unexpected number of items returned by dataset")

            logits = torch.squeeze(logits)
            sig = torch.sigmoid(logits)
            sig = torch.mean(sig, 0)  # average over time dimension

            # Weighted combination for 3-item batches
            if len(batch_data) == 3:
                B_total = logits.size(0)
                half = B_total // 2
                sig1 = torch.sigmoid(logits[:half]).mean(0)
                sig2_ = torch.sigmoid(logits[half:]).mean(0)

                if corrupt_modality == 'AUDIO':
                    w_audio = 0.5 * math.exp(-alpha * corruption_percentage)
                    w_rgb = 1.0 - w_audio
                    sig = w_rgb * sig1 + w_audio * sig2_
                elif corrupt_modality == 'RGB':
                    w_rgb = similarity
                    w_audio = 1.0 - w_rgb
                    sig = w_rgb * sig1 + w_audio * sig2_
                else:
                    sig = alpha * sig1 + beta * sig2_

            pred = torch.cat((pred, sig))

            # --- Now let's repeat each value 16 times & align GT ---
            sig_np = sig.cpu().numpy()
            sig_repeated = np.repeat(sig_np, 16)           # shape: [T * 16]
            chunk_size = len(sig_repeated)                 # e.g. 67*16, etc.
            gt_chunk = gt[gt_index : gt_index + chunk_size]
            gt_index += chunk_size

            # Append to file
            with open(filename_text, "a") as f:
                f.write(f"\nFilename: {filename}\n")
                # f.write(f"Ground truth slice: {gt_chunk}\n")
                # f.write(f"Repeated preds: {sig_repeated}\n")
                gt_str = ", ".join(map(str, gt_chunk))
                f.write(f"Primary GT slice: {gt_str}\n")
                sig_str = ", ".join(map(str, sig_repeated))
                f.write(f"Primary preds repeated: {sig_str}\n")

            # Handle logits2
            logits2 = torch.squeeze(logits2)
            sig2 = torch.sigmoid(logits2)
            sig2 = torch.mean(sig2, 0)

            if len(batch_data) == 3:
                B_total2 = logits2.size(0)
                half2 = B_total2 // 2
                sig1_2 = torch.sigmoid(logits2[:half2]).mean(0)
                sig2_2 = torch.sigmoid(logits2[half2:]).mean(0)

                if corrupt_modality == 'AUDIO':
                    w_audio = 0.5 * math.exp(-alpha * corruption_percentage)
                    w_rgb = 1.0 - w_audio
                    sig2 = w_rgb * sig1_2 + w_audio * sig2_2
                elif corrupt_modality == 'RGB':
                    w_rgb = 0.5 * math.exp(-beta * corruption_percentage)
                    w_audio = 1.0 - w_rgb
                    sig2 = w_rgb * sig1_2 + w_audio * sig2_2
                else:
                    sig2 = alpha * sig1_2 + beta * sig2_2

            sig2 = torch.unsqueeze(sig2, 1)
            pred2 = torch.cat((pred2, sig2))

        # Once done with all batches, compute final metrics
        pred = pred.cpu().detach().numpy()
        pred2 = pred2.cpu().detach().numpy()

        # Repeat each predicted point 16 times for scoring
        precision, recall, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(recall, precision)
        ap1 = average_precision_score(list(gt), np.repeat(pred, 16))

        precision, recall, _ = precision_recall_curve(list(gt), np.repeat(pred2, 16))
        pr_auc2 = auc(recall, precision)
        ap2 = average_precision_score(list(gt), np.repeat(pred2, 16))

        return pr_auc, pr_auc2, ap1, ap2
