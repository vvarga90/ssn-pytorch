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
from lib.utils.metrics import achievable_segmentation_accuracy, compute_segmentation_iou

from lib.ssn.ssn import sparse_ssn_iter

@torch.no_grad()
def eval_model(model, loader, color_scale, pos_scale, nspix, device, ignore_background=False):

    model_training_mode = model.training
    model.eval()
    sum_asa = 0
    sum_iou = 0
    for data in loader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        height, width = inputs.shape[-2:]

        nspix_per_axis = int(math.sqrt(nspix)) 
        pos_scale_mod = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)   # INFO: this line was erroneous, now fixed

        coords = torch.stack(torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij'), 0)
        coords = coords[None].repeat(inputs.shape[0], 1, 1, 1).float()

        inputs = torch.cat([color_scale*inputs, pos_scale_mod*coords], 1)

        _, Q, H, feat = model(inputs, nspix=nspix)

        H = H.reshape(height, width)
        labels = labels.argmax(1).reshape(height, width)
        
        H_npy = H.to("cpu").detach().numpy()
        labels_npy = labels.to("cpu").numpy()
        asa = achievable_segmentation_accuracy(H_npy, labels_npy)
        iou, _ = compute_segmentation_iou(H_npy, labels_npy, ignore_background=ignore_background)
        sum_asa += asa
        sum_iou += iou
    model.train(mode=model_training_mode)   # reset model to original state
    return sum_asa / len(loader), sum_iou / len(loader)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, help="/path/to/BSR")
    parser.add_argument("--weight", default="./log/best_model.pth", type=str, help="/path/to/weigh.ts")
    parser.add_argument("--fdim", default=15, type=int, help="embedding dimension  (!!! excluding LAB,XY,etc. concatenated at the end !!!")
    parser.add_argument("--niter", default=5, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=1000, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    parser.add_argument("--eval_iou_ignore_background", default=True, type=bool)
    args = parser.parse_args()
    print("Config: ", args)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Device is:", device)

    model = SSNModel(deep_feature_dim=args.fdim, n_iter=args.niter).to(device)

    model.load_state_dict(torch.load(args.weight))
    model.eval()

    test_dataset_bsds = bsds.BSDS(args.root, split="val")
    test_loader_bsds = DataLoader(test_dataset_bsds, 1, shuffle=False, drop_last=False)

    t0 = time.time()
    bsds_asa, bsds_iou = eval_model(model=model, loader=test_loader_bsds, color_scale=args.color_scale, \
                                                            pos_scale=args.pos_scale, nspix=args.nspix, device=device, \
                                                            ignore_background=args.eval_iou_ignore_background)
    t1 = time.time()
    print(f"validation asa (BSDS) {bsds_asa}, iou {bsds_iou}, eval time {t1-t0}")
