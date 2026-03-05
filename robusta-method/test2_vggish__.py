

from sklearn.metrics import auc, precision_recall_curve, average_precision_score
import numpy as np
import torch
 
def test(dataloader, model, device, gt, args):
    
    alpha = args.alpha
    beta = args.beta
    
    # gt = np.concatenate([gt, gt])
    
    cm = args.corrupt_modality
    corrupt_modality = cm[0] if isinstance(cm, list) and cm else cm
    corruption_percentage = args.corruption_percentage  # in [0,1]
    

    # Now write filename, predictions, and these ground-truth labels
    filename_text = f"results_vggish_babble_{corruption_percentage}.txt"
    with open(filename_text, "w") as f:
        f.write("Starting new results file\n")

    
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(device)
        pred2 = torch.zeros(0).to(device)
        gt_index = 0
        
        for i, batch_data in enumerate(dataloader):
            alpha = args.alpha
            beta = args.beta
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

                # # Project and combine audio features
                B, T, D = features2.shape
                features2 = features2.view(B*T, D)


                # Combine (for example by averaging)
                # combined_features = (features1 + features2) / 2.0
                # features2 = features2[:1]
                # print(features2.shape)
                
                # if torch.all(features1 == 0):
                    # alpha = 0

                # elif torch.all(features2 == 0):
                    # beta = 0

                features2 = model.audio_proj(features2)
                features2 = features2.view(B, T, 2048)
                combined_features = torch.cat([features1, features2], dim=0)  # [B+B, T, 2048]
                                    

                logits, logits2 = model(inputs=combined_features, seq_len=None)
            else:
                raise ValueError("Unexpected number of items returned by dataset")

            # Process logits
            logits = torch.squeeze(logits)
            sig = torch.sigmoid(logits)
            sig = torch.mean(sig, 0)
            
            # -----------------------------
            # NEW: alpha–beta weighting if 
            # we have two features
            # -----------------------------
            if len(batch_data) == 3:
                B_total = logits.size(0)  # should be 2B if we concatenated two features
                half = B_total // 2
                sig1 = torch.sigmoid(logits[:half])
                sig2_ = torch.sigmoid(logits[half:])
                sig1_mean = sig1.mean(0)
                sig2_mean = sig2_.mean(0)
                # Recompute `sig` with alpha and beta
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

            # -----------------------------
            # NEW: alpha–beta weighting if 
            # we have two features
            # -----------------------------
            if len(batch_data) == 3:
                B_total2 = logits2.size(0)
                half2 = B_total2 // 2
                sig1_2 = torch.sigmoid(logits2[:half2])
                sig2_2 = torch.sigmoid(logits2[half2:])
                sig1_mean2 = sig1_2.mean(0)
                sig2_mean2 = sig2_2.mean(0)
                # Recompute `sig2` with alpha and beta
                sig2 = alpha * sig1_mean2 + beta * sig2_mean2

            sig2 = torch.unsqueeze(sig2, 1)
            pred2 = torch.cat((pred2, sig2))

        pred = pred.cpu().detach().numpy()
        pred2 = pred2.cpu().detach().numpy()

        # Compute metrics
        precision, recall, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(recall, precision)
        ap1 = average_precision_score(list(gt), np.repeat(pred, 16))

        precision, recall, _ = precision_recall_curve(list(gt), np.repeat(pred2, 16))
        pr_auc2 = auc(recall, precision)
        ap2 = average_precision_score(list(gt), np.repeat(pred2, 16))

        return pr_auc, pr_auc2, ap1, ap2
