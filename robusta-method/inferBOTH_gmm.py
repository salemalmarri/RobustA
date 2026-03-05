"""Both RGB+Audio corruption - GMM dynamic weighting.
Uses datasetCMBOTH + test2_combined (GMM-based).
"""
from torch.utils.data import DataLoader
import torch
import numpy as np
from model import Model
from datasetCMBOTH import Dataset
from test2_combined import test
import option
import time


if __name__ == '__main__':
    args = option.parser.parse_args()
    device = torch.device("cuda")
    print(args.test_corrupted_list)
    print(args.test_corrupted_audio_list)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=10, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    model = Model(args)
    model = model.to(device)
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('./ckpt/robusta_best.pkl').items()})

    gt = np.load(args.gt)
    st = time.time()
    pr_auc, pr_auc_online, ap1, ap2 = test(test_loader, model, device, gt, args)
    print('offline pr_auc:{0:.4}; online pr_auc:{1:.4}\n'.format(pr_auc, pr_auc_online))
