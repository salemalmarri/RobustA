import torch
import time
import numpy as np
import random
import os
from model import Model
from dataset2 import Dataset
from train import train
from test import test
import option
from torch.utils.data import DataLoader
try:
    from comet_ml import Experiment
    HAS_COMET = True
except ImportError:
    HAS_COMET = False

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    
    args = option.parser.parse_args()
    
    print(args)
    
    # Fix the seed to 64 at the very start
    seed = args.seed
    setup_seed(seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment = None
    if HAS_COMET and os.environ.get("COMET_API_KEY"):
        experiment = Experiment(
            api_key=os.environ["COMET_API_KEY"],
            project_name="CMIL",
            workspace="salemalmarri"
        )

    comet_run_name = experiment.get_name() if experiment else "default"
    ckpt_folder = f'./ckpt/{comet_run_name}'
    os.makedirs(ckpt_folder, exist_ok=True)

    # Create data loaders
    train_rgb_loader = DataLoader(
        Dataset(args, test_mode=False, modality='RGB'),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    train_audio_loader = DataLoader(
        Dataset(args, test_mode=False, modality='AUDIO'),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = DataLoader(
        Dataset(args, test_mode=True),
        batch_size=10, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # Initialize the model
    model = Model(args).to(device)

    # Separate model parameters
    approximator_param = list(map(id, model.approximator.parameters()))
    approximator_param += list(map(id, model.conv1d_approximator.parameters()))
    base_param = filter(lambda p: id(p) not in approximator_param, model.parameters())

    # Create optimizer
    optimizer = torch.optim.Adam([
        {'params': base_param},
        {'params': model.approximator.parameters(), 'lr': args.lr / 2},
        {'params': model.conv1d_approximator.parameters(), 'lr': args.lr / 2}
    ],
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.scheduler_milestones,
        gamma=0.1
    )

    criterion = torch.nn.BCELoss()
    gt = np.load(args.gt)

    # Evaluate before training
    pr_auc, pr_auc_online, ap1, ap2 = test(test_loader, model, device, gt)
    print(
        f'Random init: offline pr_auc:{pr_auc:.4f}; '
        f'online pr_auc:{pr_auc_online:.4f}; '
        f'offline_ap1:{ap1:.4f}; online_ap2:{ap2:.4f}\n'
    )

    # Track the best PR-AUC so far
    best_pr_auc = pr_auc
    # This is the single best-checkpoint filename used throughout
    best_ckpt_name = f"{ckpt_folder}/{args.model_name}_best.pkl"

    is_topk = True

    # Start training
    for epoch in range(args.max_epoch):
        scheduler.step()
        model.train()

        # Example training loop: combine RGB + Audio
        for (rgb_input, rgb_label, _), (audio_input, audio_label, _) in zip(train_rgb_loader, train_audio_loader):
            rgb_input = rgb_input.to(device)
            rgb_label = rgb_label.to(device)
            audio_input = audio_input.to(device)
            audio_label = audio_label.to(device)

            # # Project audio to 2048-d
            B_a, T_a, D_a = audio_input.shape
            audio_input = audio_input.view(B_a * T_a, D_a)
            audio_input = model.audio_proj(audio_input)
            audio_input = audio_input.view(B_a, T_a, 2048)

            # Combine RGB and audio
            combined_input = torch.cat([rgb_input, audio_input], dim=0)
            combined_label = torch.cat([rgb_label, audio_label], dim=0)

            # Single training step
            train([(combined_input, combined_label)], model, optimizer, criterion, device, is_topk)

        # Evaluate after epoch
        pr_auc, pr_auc_online, ap1, ap2 = test(test_loader, model, device, gt)

        if experiment:
            experiment.log_metric("pr_auc", pr_auc, epoch=epoch)
            experiment.log_metric("pr_auc_online", pr_auc_online, epoch=epoch)
            experiment.log_metric("offline_ap1", ap1, epoch=epoch)
            experiment.log_metric("online_ap2", ap2, epoch=epoch)

        print(
            f'Epoch {epoch}/{args.max_epoch}: '
            f'offline pr_auc:{pr_auc:.4f}; '
            f'online pr_auc:{pr_auc_online:.4f}; '
            f'offline_ap1:{ap1:.4f}; online_ap2:{ap2:.4f}\n'
        )

        # Overwrite the best model if current pr_auc is better
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            torch.save(model.state_dict(), best_ckpt_name)
            print(f"New best model saved (epoch {epoch}): {best_ckpt_name}")
