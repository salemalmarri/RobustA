from sklearn.metrics import auc, precision_recall_curve, average_precision_score
import numpy as np
import torch

def test(dataloader, model, device, gt, args):
    alpha = args.alpha
    beta = args.beta
    corruption_percentage = args.corruption_percentage
    corrupt_modality = args.corrupt_modality

    filename_text = f"results_combined_nogmm_{corruption_percentage}.txt"
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

            if len(batch_data) == 2:
                features, filename = batch_data
                features = features.to(device)
                logits, logits2 = model(inputs=features, seq_len=None)

            elif len(batch_data) == 3:
                features1, features2, filename = batch_data
                features1 = features1.to(device)
                features2 = features2.to(device)

                B, T, D = features2.shape
                features2 = features2.view(B * T, D)
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
                sig1 = torch.sigmoid(logits[:half]).mean(0)
                sig2 = torch.sigmoid(logits[half:]).mean(0)
                sig = alpha * sig1 + beta * sig2

            pred = torch.cat((pred, sig))

            sig_np = sig.cpu().numpy()
            sig_repeated = np.repeat(sig_np, 16)
            chunk_size = len(sig_repeated)
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
                sig1_2 = torch.sigmoid(logits2[:half]).mean(0)
                sig2_2 = torch.sigmoid(logits2[half:]).mean(0)
                sig2 = alpha * sig1_2 + beta * sig2_2

            sig2 = torch.unsqueeze(sig2, 1)
            pred2 = torch.cat((pred2, sig2))

        pred = pred.cpu().numpy()
        pred2 = pred2.cpu().numpy()

        precision, recall, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(recall, precision)
        ap1 = average_precision_score(list(gt), np.repeat(pred, 16))

        precision, recall, _ = precision_recall_curve(list(gt), np.repeat(pred2, 16))
        pr_auc2 = auc(recall, precision)
        ap2 = average_precision_score(list(gt), np.repeat(pred2, 16))

        return pr_auc, pr_auc2, ap1, ap2
