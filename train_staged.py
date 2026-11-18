#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分阶段训练 + 损失调优（目标：提升 LOL PSNR，逼近教师 DDNet 21.45 dB）。

阶段1 (--mode recon)：纯 GT 重建预训练，关掉所有蒸馏项。
    损失 = w_mse*MSE + w_l1*L1 + w_ssim*(1-SSIM)
    PSNR 与 MSE 单调相关，因此提高 MSE 权重是压低 MSE、抬升 PSNR 的最直接手段。

阶段2 (--mode distill)：从阶段1权重接续，加入教师蒸馏项做结构/感知对齐，
    但重建项（尤其 MSE）仍保持较高权重，避免被蒸馏项重新带偏 GT。

用法：
  # 阶段1：重建预训练
  python train_staged.py --mode recon --gpu 0 --epochs 80 \
      --w_mse 0.5 --w_l1 0.3 --w_ssim 0.2 --name recon_mse0.5

  # 阶段2：从阶段1的 best 权重接续蒸馏微调
  python train_staged.py --mode distill --gpu 0 --epochs 40 \
      --resume /data/lowlight_paper_revision/models/staged/recon_mse0.5.pth \
      --w_mse 0.3 --w_l1 0.2 --w_ssim 0.1 \
      --w_cos 0.2 --w_per 0.1 --w_bri 0.05 --w_con 0.05 \
      --name distill_after_recon
"""
import os
import sys
import time
import argparse

# 限制 DataLoader 工作进程，避免多任务并行时 CPU 过载
try:
    os.cpu_count = lambda: 8
except Exception:
    pass

# 必须在 import torch 之前设置可见 GPU
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--gpu", type=int, default=0)
_args_tmp, _ = _p.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_args_tmp.gpu)

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

BASE = "/data/lowlight_paper_revision/code/our_model"
sys.path.insert(0, BASE)

from config import PathConfig, TrainingConfig, DEVICE  # noqa: E402
from models import AblationModelFactory  # noqa: E402
from dataset import DataLoaderFactory, LowLightDataset  # noqa: E402
from trainer import DistillationTrainer  # noqa: E402

# 教师 DDNet 路径
sys.path.insert(0, "/home/user/YouSihai/lightEnhancement/DDNet")

CKPT_DIR = "/data/lowlight_paper_revision/models/staged"


def set_seed(seed):
    """固定 random/numpy/torch/cuda 全部随机源，保证按 seed 可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_teacher(weight_path, device):
    from model import Main as TeacherModel
    teacher = TeacherModel()
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    teacher.load_state_dict(state, strict=False)
    return teacher.to(device).eval()


# ---------------- 损失 ----------------
def ssim_loss_fn(s, t):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x = s.mean(dim=[2, 3], keepdim=True)
    mu_y = t.mean(dim=[2, 3], keepdim=True)
    sigma_x = s.std(dim=[2, 3], keepdim=True)
    sigma_y = t.std(dim=[2, 3], keepdim=True)
    sigma_xy = ((s - mu_x) * (t - mu_y)).mean(dim=[2, 3], keepdim=True)
    val = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
          ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))
    return 1 - val.mean()


def rgb_to_y(rgb):
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


class StageLoss(nn.Module):
    """统一损失：mode=recon 只重建，mode=distill 重建+蒸馏。所有权重固定（无 softmax 学习权重）。"""

    def __init__(self, mode, w_mse, w_l1, w_ssim, w_cos=0.0, w_per=0.0, w_bri=0.0, w_con=0.0):
        super().__init__()
        self.mode = mode
        self.w_mse = w_mse
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_cos = w_cos
        self.w_per = w_per
        self.w_bri = w_bri
        self.w_con = w_con

    def _perceptual(self, s, t):
        vgg = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=False), nn.MaxPool2d(2),
        ).to(s.device)
        with torch.no_grad():
            tf = vgg(t)
        sf = vgg(s)
        return F.l1_loss(sf, tf)

    def forward(self, student, teacher, target):
        losses = {}
        if target is not None:
            losses["mse"] = F.mse_loss(student, target)
            losses["l1"] = F.l1_loss(student, target)
            losses["ssim"] = ssim_loss_fn(student, target)
        if self.mode == "distill":
            losses["cos"] = 1 - F.cosine_similarity(
                F.normalize(student.view(student.size(0), -1), dim=1),
                F.normalize(teacher.view(teacher.size(0), -1), dim=1),
                dim=1).mean()
            losses["per"] = self._perceptual(student, teacher)
            losses["bri"] = F.l1_loss(rgb_to_y(student), rgb_to_y(teacher))
            losses["con"] = F.l1_loss(student.std(dim=[2, 3]), teacher.std(dim=[2, 3]))

        total = self.w_mse * losses.get("mse", 0.0) + \
                self.w_l1 * losses.get("l1", 0.0) + \
                self.w_ssim * losses.get("ssim", 0.0)
        if self.mode == "distill":
            total = total + self.w_cos * losses["cos"] + self.w_per * losses["per"] + \
                    self.w_bri * losses["bri"] + self.w_con * losses["con"]
        losses["total"] = total
        return losses


class LossCalculator:
    def __init__(self, device, loss_module):
        self.loss_module = loss_module.to(device)

    def compute(self, student_output, teacher_output, target=None, epoch=None):
        return self.loss_module(student_output, teacher_output, target)


def make_trainer(teacher, student, device, mode, weights, lr):
    trainer = DistillationTrainer(teacher, student, device, learning_rate=lr)
    loss = StageLoss(mode, weights["w_mse"], weights["w_l1"], weights["w_ssim"],
                     weights.get("w_cos", 0.0), weights.get("w_per", 0.0),
                     weights.get("w_bri", 0.0), weights.get("w_con", 0.0))
    trainer.loss_calculator = LossCalculator(device, loss)
    trainer.optimizer = optim.AdamW(
        student.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        trainer.optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    return trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["recon", "distill"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--model_type", type=str, default="full",
                    choices=["full", "without_attention", "without_adaptive_residual",
                             "reduced_channels", "without_skip"],
                    help="AblationModelFactory 模型变体（消融实验用）")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--w_mse", type=float, default=0.5)
    ap.add_argument("--w_l1", type=float, default=0.3)
    ap.add_argument("--w_ssim", type=float, default=0.2)
    ap.add_argument("--w_cos", type=float, default=0.0)
    ap.add_argument("--w_per", type=float, default=0.0)
    ap.add_argument("--w_bri", type=float, default=0.0)
    ap.add_argument("--w_con", type=float, default=0.0)
    ap.add_argument("--img_mode", type=str, default="original",
                    choices=["original", "legacy"],
                    help="original=原始分辨率纯净像素(默认); legacy=旧的256+ColorJitter")
    ap.add_argument("--hflip", action="store_true",
                    help="训练时随机水平翻转(低高图一致)")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子，固定 torch/np/random/cudnn，保证可复现")
    args = ap.parse_args()
    set_seed(args.seed)

    if args.name is None:
        args.name = f"{args.mode}_mse{args.w_mse}"

    os.makedirs(CKPT_DIR, exist_ok=True)
    save_raw = os.path.join(CKPT_DIR, f"{args.name}.pth")
    print(f"[start] mode={args.mode} name={args.name} ckpt={save_raw}", flush=True)

    student = AblationModelFactory.create_model(args.model_type).to(DEVICE)
    teacher = load_teacher(PathConfig.TEACHER_WEIGHT_PATH, DEVICE)

    if args.resume is not None and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        for k in ("student_state_dict", "model_state_dict", "state_dict"):
            if k in ckpt:
                ckpt = ckpt[k]
                break
        student.load_state_dict(ckpt, strict=False)
        print(f"[resume] loaded {args.resume}", flush=True)

    # ---- 数据：按 img_mode 决定增强策略 ----
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    lows = sorted(f for f in os.listdir(PathConfig.TRAIN_LOW_DIR) if f.lower().endswith(exts))
    highs = sorted(f for f in os.listdir(PathConfig.TRAIN_HIGH_DIR) if f.lower().endswith(exts))
    assert lows == highs, "低/高光照文件名不一致"
    random.seed(args.seed)
    indices = list(range(len(lows)))
    random.shuffle(indices)
    n = int(len(indices) * TrainingConfig.TRAIN_RATIO)
    train_idx, val_idx = indices[:n], indices[n:]

    if args.img_mode == "original":
        # 纯净像素：原始分辨率，可选随机水平翻转（低高一致），不做 Resize/ColorJitter
        base = [transforms.ToTensor()]
        if args.hflip:
            base = [transforms.RandomHorizontalFlip(p=0.5)] + base
        train_transform = transforms.Compose(base)
    else:
        train_transform = DataLoaderFactory.create_train_transform()
    val_transform = transforms.Compose([transforms.ToTensor()])

    def make_dataset(idx_list, transform, augment):
        low_f = [os.path.join(PathConfig.TRAIN_LOW_DIR, lows[i]) for i in idx_list]
        high_f = [os.path.join(PathConfig.TRAIN_HIGH_DIR, highs[i]) for i in idx_list]
        return LowLightDataset(low_f, high_f, transform=transform, augment=augment,
                               keep_original_size=True)

    train_dataset = make_dataset(train_idx, train_transform, augment=True)
    val_dataset = make_dataset(val_idx, val_transform, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=TrainingConfig.TRAIN_BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=TrainingConfig.VAL_BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)
    print(f"[data] img_mode={args.img_mode} hflip={args.hflip} "
          f"train={len(train_dataset)} val={len(val_dataset)}", flush=True)

    weights = {"w_mse": args.w_mse, "w_l1": args.w_l1, "w_ssim": args.w_ssim,
               "w_cos": args.w_cos, "w_per": args.w_per,
               "w_bri": args.w_bri, "w_con": args.w_con}
    trainer = make_trainer(teacher, student, DEVICE, args.mode, weights, args.lr)
    trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=args.epochs,
                  save_path=os.path.join(CKPT_DIR, args.name),
                  early_stopping_patience=TrainingConfig.EARLY_STOPPING_PATIENCE)

    torch.save(trainer.student.state_dict(), save_raw)
    print(f"[saved] best student -> {save_raw}  (best_val_psnr[0,1]={trainer.best_psnr:.4f})", flush=True)


if __name__ == "__main__":
    main()