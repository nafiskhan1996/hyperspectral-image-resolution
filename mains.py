import argparse
import os
import sys
import random
import time
import json
import math
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from data import HSTrainingData, HSTestData
from SSPSR import SSPSR
from common import *
from loss import HybridLoss
from metrics import quality_assessment


# ---------------------------
# Utils
# ---------------------------
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0
    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.cnt += n
        self.avg = self.sum / self.cnt if self.cnt > 0 else 0.0


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True


def sum_dict(a, b):
    temp = dict()
    for key in a.keys() | b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp


def adjust_learning_rate(start_lr, optimizer, epoch):
    """Decay LR by 10x every 10 epochs."""
    lr = start_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------
# Main
# ---------------------------
def build_argparser():
    main_parser = argparse.ArgumentParser(description="SSPSR Hyperspectral Super-Resolution")
    subparsers = main_parser.add_subparsers(title="subcommands", dest="subcommand")

    # Shared / common
    main_parser.add_argument("--gpus", type=str, default="0", help="CUDA visible device ids (e.g., '0' or '0,1')")
    main_parser.add_argument("--cuda", type=int, default=1, help="1=GPU (if available), 0=CPU")
    main_parser.add_argument("--seed", type=int, default=3000, help="random seed")
    main_parser.add_argument("--deterministic", action="store_true", help="enable cudnn deterministic (slower)")

    # Paths
    main_parser.add_argument("--data_root", type=str, default="./dataset", help="root folder containing datasets")
    main_parser.add_argument("--result_dir", type=str, default="./result", help="folder to save results/plots/npys")
    main_parser.add_argument("--save_dir", type=str, default="./trained_model", help="folder to save final models")
    main_parser.add_argument("--resume", type=str, default="", help="path to checkpoint .pth to resume training")
    main_parser.add_argument("--weights", type=str, default="", help="path to weights/ckpt for testing")

    # Train
    train_parser = subparsers.add_parser("train", help="training")
    train_parser.add_argument("--dataset_name", type=str, default="Chikusei", choices=["Chikusei", "Cave", "Pavia"])
    train_parser.add_argument("--n_scale", type=int, default=4, help="upscale factor")
    train_parser.add_argument("--colors", type=int, default=32, help="number of spectral bands used in training")
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=40)
    train_parser.add_argument("--learning_rate", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=0.0)
    train_parser.add_argument("--n_feats", type=int, default=256)
    train_parser.add_argument("--n_blocks", type=int, default=3)
    train_parser.add_argument("--n_subs", type=int, default=8)
    train_parser.add_argument("--n_ovls", type=int, default=2)
    train_parser.add_argument("--use_share", type=bool, default=True)
    train_parser.add_argument("--model_title", type=str, default="SSPSR")
    train_parser.add_argument("--log_interval", type=int, default=50)

    # Test
    test_parser = subparsers.add_parser("test", help="testing/evaluation")
    test_parser.add_argument("--dataset_name", type=str, default="Cave", choices=["Chikusei", "Cave", "Pavia"])
    test_parser.add_argument("--n_scale", type=int, default=4)
    test_parser.add_argument("--colors", type=int, default=32)

    return main_parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    # Device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    use_cuda = bool(args.cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Seed
    set_seed(args.seed, deterministic=args.deterministic)

    # Command
    if args.subcommand is None:
        print("ERROR: specify either 'train' or 'test'")
        sys.exit(1)

    if args.subcommand == "train":
        train(args, device)
    else:
        test(args, device)


# ---------------------------
# Train
# ---------------------------
def train(args, device):
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    # Paths
    split_root = os.path.join(args.data_root, f"{args.dataset_name}_x{args.n_scale}")
    train_path  = os.path.join(split_root, "trains")
    eval_path   = os.path.join(split_root, "evals")
    result_path = ensure_dir(os.path.join(args.result_dir, f"{args.dataset_name}_x{args.n_scale}"))
    ensure_dir(args.save_dir)
    ensure_dir("./checkpoints")

    # Datasets/Loaders
    print('===> Loading datasets')
    train_set = HSTrainingData(image_dir=train_path, colors=args.colors, augment=True)
    eval_set  = HSTrainingData(image_dir=eval_path,  colors=args.colors, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)
    eval_loader  = DataLoader(eval_set,  batch_size=args.batch_size, num_workers=4, shuffle=False)

    # Colors adjustment by dataset
    if args.dataset_name == 'Cave':
        colors = args.colors
    elif args.dataset_name == 'Pavia':
        colors = 102
    else:
        colors = args.colors

    # Model
    print(f'===> Building model for {colors} bands')
    net = SSPSR(
        n_subs=args.n_subs, n_ovls=args.n_ovls, n_colors=colors,
        n_blocks=args.n_blocks, n_feats=args.n_feats, n_scale=args.n_scale,
        res_scale=0.1, use_share=args.use_share, conv=default_conv
    )

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print("===> Using", torch.cuda.device_count(), "GPUs.")
        net = torch.nn.DataParallel(net)

    net = net.to(device)

    # Title / checkpoints
    model_title = f"{args.dataset_name}_{args.model_title}_Blocks={args.n_blocks}_Subs{args.n_subs}_Ovls{args.n_ovls}_Feats={args.n_feats}"
    args.model_title = model_title

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"=> Resuming from checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint.get("epoch", 0)
        # checkpoint saved as {"epoch": e, "model": model}; handle DP vs non-DP
        state_model = checkpoint["model"].state_dict() if hasattr(checkpoint["model"], "state_dict") else checkpoint["model"]
        net.load_state_dict(state_model)
    else:
        print("=> Training from scratch")

    # Loss / Optim / Log
    h_loss = HybridLoss(spatial_tv=True, spectral_tv=True)
    L1_loss = torch.nn.L1Loss()
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    writer = SummaryWriter(f"runs/{model_title}_{time.strftime('%Y%m%d_%H%M%S')}")

    print('===> Start training')
    best_val_loss = float('inf')         # FIX 1: correct initialization
    patience = 5
    counter = 0

    for e in range(start_epoch, args.epochs):
        adjust_learning_rate(args.learning_rate, optimizer, e + 1)
        batch_loss_meter = AverageMeter()

        print(f"Start epoch {e + 1}, lr = {optimizer.param_groups[0]['lr']}")
        net.train()
        for iteration, (x, lms, gt) in enumerate(train_loader):
            x, lms, gt = x.to(device), lms.to(device), gt.to(device)
            optimizer.zero_grad()
            y = net(x, lms)
            loss = h_loss(y, gt)
            loss.backward()
            optimizer.step()

            batch_loss_meter.update(loss.item())
            # logging
            if (iteration + 1) % args.log_interval == 0:
                print(f"===> {time.ctime()} B{args.n_blocks} Sub{args.n_subs} Fea{args.n_feats} GPU{args.gpus}\t"
                      f"Epoch[{e + 1}]({iteration + 1}/{len(train_loader)}): Loss: {loss.item():.6f}")
                n_iter = e * len(train_loader) + iteration + 1
                writer.add_scalar('train/loss', loss.item(), n_iter)

        print(f"===> {time.ctime()}\tEpoch {e + 1} Training Complete: Avg. Loss: {batch_loss_meter.avg:.6f}")

        # Validation
        eval_loss = validate(args, eval_loader, net, L1_loss, device)
        writer.add_scalar('val/loss', eval_loss, e + 1)
        writer.add_scalar('train/avg_epoch_loss', batch_loss_meter.avg, e + 1)

        # Early stopping & checkpointing  (FIX 1)
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            counter = 0
            save_checkpoint(args, net, e + 1)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {e + 1} (no improvement for {patience} epochs).")
                break

        # Periodic checkpoint
        if (e + 1) % 5 == 0:
            save_checkpoint(args, net, e + 1)

    # Save final model
    net.eval().cpu()
    ensure_dir(args.save_dir)
    save_model_filename = f"{model_title}_epoch_{args.epochs}_{time.strftime('%Y%m%d_%H%M%S')}.pth"
    save_model_path = os.path.join(args.save_dir, save_model_filename)
    if isinstance(net, torch.nn.DataParallel):
        torch.save(net.module.state_dict(), save_model_path)
    else:
        torch.save(net.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)

    # Optional quick test on held-out set path (if present)
    test_data_dir = os.path.join(args.data_root, f"{args.dataset_name}_x{args.n_scale}", "Cave_test.mat")
    if os.path.isfile(test_data_dir):
        print("Running quick testset evaluation...")
        run_test_loop(net.to(device), args, device, test_data_dir, result_path)


# ---------------------------
# Validate
# ---------------------------
def validate(args, loader, model, criterion, device):
    model.eval()
    meter = AverageMeter()
    with torch.no_grad():
        for i, (ms, lms, gt) in enumerate(loader):
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
            y = model(ms, lms)
            loss = criterion(y, gt)
            meter.update(loss.item())
    print(f"===> {time.ctime()}\tEpoch evaluation Complete: Avg. Loss: {meter.avg:.6f}")
    model.train()
    return meter.avg


# ---------------------------
# Test
# ---------------------------
def test(args, device):
    # Paths
    result_path = ensure_dir(os.path.join(args.result_dir, f"{args.dataset_name}_x{args.n_scale}"))
    test_data_dir = os.path.join(args.data_root, f"{args.dataset_name}_x{args.n_scale}", "Cave_test.mat")

    if not args.weights or not os.path.isfile(args.weights):
        print("ERROR: please provide a valid --weights path to a trained checkpoint/state_dict (.pth)")
        sys.exit(1)

    # Colors by dataset
    if args.dataset_name == 'Pavia':
        colors = 102
    else:
        colors = args.colors

    # Build model from args (FIX 2)
    net = SSPSR(
        n_subs=getattr(args, "n_subs", 8),
        n_ovls=getattr(args, "n_ovls", 2),
        n_colors=colors,
        n_blocks=getattr(args, "n_blocks", 3),
        n_feats=getattr(args, "n_feats", 256),
        n_scale=args.n_scale,
        res_scale=0.1, use_share=True, conv=default_conv
    )
    # Load weights (supports checkpoints saved via save_checkpoint or raw state_dict)
    state = torch.load(args.weights, map_mode='cpu' if device.type == 'cpu' else None)
    if isinstance(state, dict) and "model" in state:
        # our checkpoint format
        state = state["model"].state_dict() if hasattr(state["model"], "state_dict") else state["model"]
    net.load_state_dict(state if isinstance(state, dict) else state)
    net = net.to(device).eval()

    print('===> Loading testset')
    test_set = HSTestData(test_data_dir, colors=colors)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    print('===> Start testing')
    run_test_loop(net, args, device, test_data_dir, result_path)


def run_test_loop(net, args, device, test_data_dir, result_path):
    with torch.no_grad():
        output = []
        test_number = 0  # FIX 2
        mat = [4, 7, 3]

        test_set = HSTestData(test_data_dir, colors=args.colors)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        for i, (ms, lms, gt) in enumerate(test_loader):
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
            y = net(ms, lms)

            # Save quick RGB-ish previews (FIX 3: safe dirs)
            ensure_dir(result_path)
            plt.imsave(os.path.join(result_path, 'gt.png'),
                       gt.squeeze().cpu().numpy().transpose(1, 2, 0)[:, :, mat])
            plt.imsave(os.path.join(result_path, 'ms.png'),
                       ms.squeeze().cpu().numpy().transpose(1, 2, 0)[:, :, mat])
            plt.imsave(os.path.join(result_path, 'y.png'),
                       y.detach().squeeze().cpu().numpy().transpose(1, 2, 0)[:, :, mat])

            # Metrics
            y_np = y.squeeze().cpu().numpy().transpose(1, 2, 0)
            gt_np = gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y_np = y_np[:gt_np.shape[0], :gt_np.shape[1], :]
            if i == 0:
                indices = quality_assessment(gt_np, y_np, data_range=1., ratio=args.n_scale)
            else:
                indices = sum_dict(indices, quality_assessment(gt_np, y_np, data_range=1., ratio=args.n_scale))
            output.append(y_np)
            test_number += 1

        for k in indices:
            indices[k] = indices[k] / max(test_number, 1)

    # Save artifacts
    np.save(os.path.join(result_path, "test.npy"), output)
    print("Test finished, results saved to:", os.path.join(result_path, "test.npy"))
    print(indices)
    with open(os.path.join(result_path, f"QI_{args.model_title if hasattr(args, 'model_title') else 'model'}_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
        json.dump(indices, f, indent=2)


# ---------------------------
# Checkpoint
# ---------------------------
def save_checkpoint(args, model, epoch):
    device = torch.device("cuda" if (bool(args.cuda) and torch.cuda.is_available()) else "cpu")
    model = model.to("cpu").eval()
    checkpoint_model_dir = ensure_dir('./checkpoints')
    ckpt_model_filename = f"{args.dataset_name}_{args.model_title}_ckpt_epoch_{epoch}.pth"
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
    state = {"epoch": epoch, "model": model}
    torch.save(state, ckpt_model_path)
    model.to(device).train()
    print(f"Checkpoint saved to {ckpt_model_path}")


if __name__ == "__main__":
    main()
