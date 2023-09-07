import os, math
import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils.meter import Meter
from model import SSNModel
from lib.dataset import bsds, augmentation
from lib.utils.loss import reconstruct_loss_with_cross_etnropy, reconstruct_loss_with_mse
from lib.utils.metrics import achievable_segmentation_accuracy
from eval import eval_model


def update_param(data, model, optimizer, compactness, color_scale, pos_scale, nspix, device):
    inputs, labels = data

    inputs = inputs.to(device)
    labels = labels.to(device)

    height, width = inputs.shape[-2:]

    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)    

    coords = torch.stack(torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij'), 0)
    coords = coords[None].repeat(inputs.shape[0], 1, 1, 1).float()

    inputs = torch.cat([color_scale*inputs, pos_scale*coords], 1)
    _, Q, H, feat = model(inputs, nspix=nspix)

    recons_loss = reconstruct_loss_with_cross_etnropy(Q, labels)
    compact_loss = reconstruct_loss_with_mse(Q, coords.reshape(*coords.shape[:2], -1), H)

    loss = recons_loss + compactness * compact_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {"loss": loss.item(), "reconstruction": recons_loss.item(), "compact": compact_loss.item()}


def train(cfg):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Device is:", device)

    model = SSNModel(deep_feature_dim=cfg.fdim, n_iter=cfg.niter).to(device)

    optimizer = optim.Adam(model.parameters(), cfg.lr)

    augment = augmentation.Compose([augmentation.RandomHorizontalFlip(), augmentation.RandomScale(), augmentation.RandomCrop()])
    train_dataset = bsds.BSDS(cfg.root, split="train", geo_transforms=augment)
    train_loader = DataLoader(train_dataset, cfg.batchsize, shuffle=True, drop_last=True, num_workers=cfg.nworkers)

    test_dataset_bsds = bsds.BSDS(cfg.root, split="val")
    test_loader_bsds = DataLoader(test_dataset_bsds, 1, shuffle=False, drop_last=False)

    meter = Meter()

    iterations = 0
    max_val_bsds_asa = 0
    while iterations < cfg.train_iter:
        for data in train_loader:
            iterations += 1
            metric = update_param(data, model, optimizer, cfg.compactness, cfg.color_scale, cfg.pos_scale, cfg.nspix, device)
            meter.add(metric)
            state = meter.state(f"[{iterations}/{cfg.train_iter}]")
            print(state)
            if (iterations % cfg.test_interval) == 0:
                t0 = time.time()
                bsds_asa, bsds_iou = eval_model(model=model, loader=test_loader_bsds, color_scale=cfg.color_scale, \
                                                                pos_scale=cfg.pos_scale, nspix=cfg.nspix, device=device)
                t1 = time.time()
                print(f"validation asa (BSDS) {bsds_asa}, iou {bsds_iou}, eval time {t1-t0}")

                if bsds_asa > max_val_bsds_asa:
                    max_val_bsds_asa = bsds_asa
                    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "best_model.pth"))
                    print("    (best_model.pth was overwritten)")

            if iterations == cfg.train_iter:
                break

    unique_id = str(int(time.time()))
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model"+unique_id+".pth"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, help="/path/to/BSR")
    parser.add_argument("--out_dir", default="./log", type=str, help="/path/to/output directory")
    parser.add_argument("--batchsize", default=6, type=int)
    parser.add_argument("--nworkers", default=8, type=int, help="number of threads for CPU parallel")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--train_iter", default=1000, type=int)
    parser.add_argument("--fdim", default=15, type=int, help="embedding dimension  (!!! excluding LAB,XY,etc. concatenated at the end !!!")
    parser.add_argument("--niter", default=5, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=1000, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    parser.add_argument("--compactness", default=1e-5, type=float)
    parser.add_argument("--test_interval", default=200, type=int)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Config: ", args)

    train(args)