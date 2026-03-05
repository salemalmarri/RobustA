from sklearn.metrics import auc, precision_recall_curve, average_precision_score
import numpy as np
import torch

def test(dataloader, model, device, gt):#, args):
    
    # gt = np.concatenate([gt, gt])
    
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(device)
        pred2 = torch.zeros(0).to(device)
        
        for i, batch_data in enumerate(dataloader):
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
                features2 = model.audio_proj(features2)
                features2 = features2.view(B, T, 2048)

                # Combine (for example by averaging)
                # combined_features = (features1 + features2) / 2.0
                # features2 = features2[:1]
                # print(features2.shape)
                combined_features = torch.cat([features1, features2], dim=0)  # [B+B, T, 2048]

                logits, logits2 = model(inputs=combined_features, seq_len=None)
            else:
                raise ValueError("Unexpected number of items returned by dataset")

            # Process logits
            logits = torch.squeeze(logits)
            sig = torch.sigmoid(logits)
            sig = torch.mean(sig, 0)
            pred = torch.cat((pred, sig))

            logits2 = torch.squeeze(logits2)
            sig2 = torch.sigmoid(logits2)
            sig2 = torch.mean(sig2, 0)
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
