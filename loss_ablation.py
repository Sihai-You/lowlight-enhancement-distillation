#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蒸馏损失项消融实验（P1-3.7）。全部基于 'full' 学生架构，仅损失项不同：

  no_distill    去掉余弦对齐项 L_cos（w_cos·L_cos）
  no_perceptual 去掉感知项 L_per（w_per·L_per）
  no_brightness 去掉亮度一致项 L_bri（w_bri·L_bri）
  no_contrast   去掉对比度项 L_con（w_con·L_con）
  no_kd         纯监督（仅 L1 + MSE 重建，完全不用教师模型）

评估统一协议（与论文 Table II/V 一致）：
  原始分辨率 + pyiqa(PSNR/SSIM) + pyiqa(NIQE) + pypiqe(PIQE) + Tukey 离群值剔除。

用法：
  python loss_ablation.py train --variant no_distill --gpu 0
  python loss_ablation.py eval
"""
import os
import sys
import json
import argparse

# 限制 DataLoader 工作进程数，避免多个并行训练任务过量占用 CPU
try:
    os.cpu_count = lambda: 8
except Exception:
    pass

# 必须在 import torch 之前设置可见 GPU
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--gpu", type=int, default=0)
_args_tmp, _ = _p.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_args_tmp.gpu)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BASE = "/data/lowlight_paper_revision/code/our_model"
sys.path.insert(0, BASE)

from config import PathConfig, TrainingConfig, DEVICE  # noqa: E402
from models import AblationModelFactory  # noqa: E402
from dataset import DataLoaderFactory  # noqa: E402
from trainer import DistillationTrainer  # noqa: E402

# 教师 DDNet：直接从其真实项目路径导入 Main
sys.path.insert(0, "/home/user/YouSihai/lightEnhancement/DDNet")


def load_teacher(weight_path, device):
    from model import Main as TeacherModel
    teacher = TeacherModel()
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    new_state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    teacher.load_state_dict(new_state, strict=False)
    teacher = teacher.to(device).eval()
    return teacher

CKPT_DIR = "/data/lowlight_paper_revision/models/loss_ablation"
ENH_DIR = "/data/lowlight_paper_revision/results/loss_ablation/enhanced"
OUT_JSON = "/data/lowlight_paper_revision/results/loss_ablation.json"

TEST_LOW_DIR = PathConfig.TEST_LOW_DIR
TEST_HIGH_DIR = PathConfig.TEST_HIGH_DIR

# 完整蒸馏参考 checkpoint（论文 Table V 的 full 模型）
FULL_CKPT = "/home/user/YouSihai/lightEnhancement/newStudent/models/ablation_full.pth"

VARIANTS = {
    "no_distill": {"distill"},
    "no_perceptual": {"perceptual"},
    "no_brightness": {"brightness"},
    "no_contrast": {"contrast"},
    "no_kd": {"distill", "perceptual", "brightness", "contrast", "ssim"},
}


class AblationDistillationLoss(nn.Module):
    """与 losses.AdaptiveDistillationLoss 完全一致，仅增加可禁用项。"""

    def __init__(self, disabled, temperature=3.0, init_alpha=0.7, init_beta=0.2,
                 init_gamma=0.05, init_delta=0.05):
        super().__init__()
        self.disabled = set(disabled) if disabled else set()
        self.temperature = temperature
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
        self.delta = nn.Parameter(torch.tensor(init_delta))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, student_output, teacher_output, target=None, epoch=None):
        losses = {}
        losses['distill'] = self.distillation_loss(student_output, teacher_output)
        losses['perceptual'] = self.perceptual_loss(student_output, teacher_output)
        losses['contrast'] = self.contrast_loss(student_output, teacher_output)
        losses['brightness'] = self.brightness_consistency_loss(student_output, teacher_output)
        losses['ssim'] = self.ssim_loss(student_output, teacher_output)
        if target is not None:
            losses['reconstruction'] = F.l1_loss(student_output, target)
            losses['mse'] = F.mse_loss(student_output, target)

        weights = self.softmax(torch.stack([self.alpha, self.beta, self.gamma, self.delta]))
        recon_weight = (0.1 + 0.3 * min(1.0, epoch / 50)) if epoch is not None else 0.1

        def on(k):
            return 0.0 if k in self.disabled else 1.0

        total = (on('distill') * weights[0] * losses['distill'] +
                 on('perceptual') * weights[1] * losses['perceptual'] +
                 on('brightness') * weights[2] * losses['brightness'] +
                 on('contrast') * weights[3] * losses['contrast'])
        if target is not None:
            total = total + on('reconstruction') * recon_weight * losses['reconstruction'] + \
                    on('mse') * 0.05 * losses['mse'] + on('ssim') * 0.1 * losses['ssim']

        losses['total'] = total
        losses['weights'] = weights.detach()
        return losses

    def distillation_loss(self, s, t):
        s_n = F.normalize(s.view(s.size(0), -1), dim=1)
        t_n = F.normalize(t.view(t.size(0), -1), dim=1)
        sim = F.cosine_similarity(s_n, t_n, dim=1)
        return 1 - sim.mean()

    def perceptual_loss(self, s, t):
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

    def contrast_loss(self, s, t):
        return F.l1_loss(s.std(dim=[2, 3]), t.std(dim=[2, 3]))

    def brightness_consistency_loss(self, s, t):
        return F.l1_loss(self.rgb_to_y(s), self.rgb_to_y(t))

    def ssim_loss(self, s, t):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu_x = s.mean(dim=[2, 3], keepdim=True)
        mu_y = t.mean(dim=[2, 3], keepdim=True)
        sigma_x = s.std(dim=[2, 3], keepdim=True)
        sigma_y = t.std(dim=[2, 3], keepdim=True)
        sigma_xy = ((s - mu_x) * (t - mu_y)).mean(dim=[2, 3], keepdim=True)
        val = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
              ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))
        return 1 - val.mean()

    def rgb_to_y(self, rgb):
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b


class AblationTotalLoss:
    def __init__(self, device, disabled):
        self.device = device
        self.distillation_loss = AblationDistillationLoss(disabled).to(device)

    def compute(self, student_output, teacher_output, target=None, epoch=None):
        return self.distillation_loss(student_output, teacher_output, target, epoch)


def make_trainer(teacher, student, device, disabled):
    trainer = DistillationTrainer(teacher, student, device)
    new_loss = AblationTotalLoss(device, disabled)
    trainer.loss_calculator = new_loss
    lr = trainer.optimizer.param_groups[0]['initial_lr']
    trainer.optimizer = optim.AdamW(
        list(trainer.student.parameters()) + list(new_loss.distillation_loss.parameters()),
        lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        trainer.optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    return trainer


def train_variant(name, disabled, epochs=100):
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{name}.pth")
    print(f"[train] variant={name} disabled={disabled} ckpt={ckpt_path}", flush=True)

    student = AblationModelFactory.create_model("full").to(DEVICE)
    teacher = load_teacher(PathConfig.TEACHER_WEIGHT_PATH, DEVICE)

    train_transform = DataLoaderFactory.create_train_transform()
    test_transform = DataLoaderFactory.create_test_transform()
    train_dataset, val_dataset = DataLoaderFactory.create_datasets(
        PathConfig.TRAIN_LOW_DIR, PathConfig.TRAIN_HIGH_DIR,
        train_ratio=TrainingConfig.TRAIN_RATIO, transform=train_transform,
        seed=TrainingConfig.RANDOM_SEED)
    train_loader, val_loader, _ = DataLoaderFactory.create_dataloaders(
        train_dataset, val_dataset, None,
        TrainingConfig.TRAIN_BATCH_SIZE, TrainingConfig.VAL_BATCH_SIZE,
        TrainingConfig.TEST_BATCH_SIZE)

    trainer = make_trainer(teacher, student, DEVICE, disabled)
    trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=epochs,
                  save_path=os.path.join(CKPT_DIR, name),
                  early_stopping_patience=TrainingConfig.EARLY_STOPPING_PATIENCE)

    # 保存最佳学生权重（原始 state_dict，与 unified_evaluate.load_model 兼容）
    torch.save(trainer.student.state_dict(), ckpt_path)
    print(f"[train] saved {ckpt_path}", flush=True)
    return ckpt_path


# ---------------- 评估 ----------------
def load_state(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("student_state_dict", "model_state_dict", "state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def run_inference(model, low_dir, high_dir, out_dir):
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    lows = sorted(f for f in os.listdir(low_dir) if f.lower().endswith(exts))
    highs = sorted(f for f in os.listdir(high_dir) if f.lower().endswith(exts))
    common = [f for f in lows if f in highs]
    saved = []
    with torch.no_grad():
        for fn in common:
            img = Image.open(os.path.join(low_dir, fn)).convert("RGB")
            t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
            t = t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            out = torch.clamp(model(t), 0.0, 1.0)
            out_np = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            sp = os.path.join(out_dir, fn)
            Image.fromarray(out_np).save(sp)
            saved.append((sp, os.path.join(high_dir, fn)))
    return saved


def _tukey(s):
    q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def mean_std(scores, remove=True):
    s = np.array(scores, dtype=float)
    if remove and len(s) >= 4:
        lo, hi = _tukey(s)
        s = s[(s >= lo) & (s <= hi)]
    return float(np.mean(s)), float(np.std(s))


def evaluate_all():
    import cv2
    import pyiqa
    from pypiqe import piqe as _piqe

    psnr_m = pyiqa.create_metric("psnr", device=DEVICE)
    ssim_m = pyiqa.create_metric("ssim", device=DEVICE)
    niqe_m = None
    niqe_path = "/home/user/.cache/torch/hub/pyiqa/niqe_modelparameters.mat"
    if os.path.exists(niqe_path):
        niqe_m = pyiqa.create_metric("niqe", pretrained_model_path=niqe_path, device=DEVICE)

    def imread_rgb(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    entries = [("full", FULL_CKPT, None)]
    for name in VARIANTS:
        ckpt_path = os.path.join(CKPT_DIR, f"{name}.pth")
        entries.append((name, ckpt_path, VARIANTS[name]))

    results = {}
    for name, ckpt_path, disabled in entries:
        if not os.path.exists(ckpt_path):
            print(f"[eval] missing {ckpt_path}, skip", flush=True)
            continue
        model = load_state(AblationModelFactory.create_model("full").to(DEVICE), ckpt_path)
        out_dir = os.path.join(ENH_DIR, name)
        pairs = run_inference(model, TEST_LOW_DIR, TEST_HIGH_DIR, out_dir)
        per = {"psnr": [], "ssim": [], "piqe": [], "niqe": []}
        for dist, ref in pairs:
            d = imread_rgb(dist).to(DEVICE)
            r = imread_rgb(ref).to(DEVICE)
            if d.shape != r.shape:
                r = F.interpolate(r, size=(d.shape[2], d.shape[3]), mode="bilinear", align_corners=False)
            with torch.no_grad():
                per["psnr"].append(psnr_m(d, r).item())
                per["ssim"].append(ssim_m(d, r).item())
                if niqe_m is not None:
                    per["niqe"].append(niqe_m(d).item())
            per["piqe"].append(float(_piqe(cv2.cvtColor(cv2.imread(dist), cv2.COLOR_BGR2RGB))[0]))
        m = {}
        for k in ("psnr", "ssim", "piqe", "niqe"):
            if per[k]:
                mean, std = mean_std(per[k])
                m[k] = {"mean": round(mean, 4), "std": round(std, 4)}
        m["disabled"] = sorted(disabled) if disabled else []
        results[name] = m
        print(f"[eval] {name}: {m}", flush=True)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[eval] wrote", OUT_JSON, flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--variant", type=str)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.mode == "train":
        assert args.variant in VARIANTS, f"unknown variant: {args.variant}"
        train_variant(args.variant, VARIANTS[args.variant], epochs=args.epochs)
    else:
        evaluate_all()