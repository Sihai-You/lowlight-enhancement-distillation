#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 FPGA 部署生成"黄金参考"测试向量与测试图片。

在 PyTorch 侧（远程 GPU 或任意装有 torch 的机器）运行，对每个样本输出：
  <name>_input.npy    float32 [1,3,H,W]，取值 [0,1]，供 bit 级比对
  <name>_output.npy   float32 [1,3,H,W]，PyTorch 参考输出 [0,1]
  <name>_input.png    低光输入图（可直接送 FPGA 推理）
  <name>_output.png   增强后图（可视化/与 FPGA 输出比对）
  <name>_gt.png       GT 高光参考图（仅当提供 --high-dir，用于 PSNR/SSIM）
  manifest.json       所有向量的 shape / dtype / md5

包含 1 组确定性合成向量（seed 固定、可复现）+ 若干真实 LOL 测试低光图。

用法：
  python generate_golden.py --model-dir . --ckpt ablation_full.pth \
      --low-dir /path/to/LOL/eval15/low --high-dir /path/to/LOL/eval15/high \
      --n-real 15 --size 600x400 [--skip-npy]
"""
import os
import sys
import json
import hashlib
import argparse

import numpy as np
import torch
from PIL import Image


def parse_size(s):
    w, h = (s.split("x") if "x" in s else [s, s])[:2]
    return int(h), int(w)


def load_model(model_dir, ckpt_path, device="cpu"):
    sys.path.insert(0, model_dir)
    from models import AblationModelFactory
    m = AblationModelFactory.create_model("full", base_channels=32).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("student_state_dict", "model_state_dict", "state_dict"):
            if k in ckpt:
                ckpt = ckpt[k]
                break
    m.load_state_dict(ckpt, strict=False)
    m.eval()
    return m


def pad_to_multiple(t, m=4):
    _, _, h, w = t.shape
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
    return t


def md5(arr):
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def to_uint8(t):
    if hasattr(t, "cpu"):
        t = t.cpu().numpy()
    t = np.asarray(t)
    if t.shape[0] in (1, 3) and t.ndim == 4:  # NCHW
        t = t[0].transpose(1, 2, 0)
    return (np.clip(t, 0.0, 1.0) * 255.0).astype(np.uint8)


class Exporter:
    def __init__(self, out_dir, size, seed, skip_npy):
        self.out_dir = out_dir
        self.size = size
        self.seed = seed
        self.skip_npy = skip_npy
        self.manifest = []

    def save(self, name, xp, yp, gt=None):
        """xp: input tensor [1,3,H,W]; yp: output tensor; gt: GT ndarray/None."""
        os.makedirs(self.out_dir, exist_ok=True)
        # PNG（可视化 / 实际送 FPGA）
        Image.fromarray(to_uint8(xp)).save(os.path.join(self.out_dir, f"{name}_input.png"))
        Image.fromarray(to_uint8(yp)).save(os.path.join(self.out_dir, f"{name}_output.png"))
        if gt is not None:
            Image.fromarray(to_uint8(gt)).save(os.path.join(self.out_dir, f"{name}_gt.png"))
        # NPY（bit 级黄金比对）
        if not self.skip_npy:
            for tag, arr in (("input", xp), ("output", yp)):
                a = np.ascontiguousarray(arr.cpu().numpy().astype(np.float32)) if hasattr(arr, "cpu") else arr.astype(np.float32)
                p = os.path.join(self.out_dir, f"{name}_{tag}.npy")
                np.save(p, a)
                self.manifest.append({"name": f"{name}_{tag}", "file": os.path.basename(p),
                                      "shape": list(a.shape), "dtype": str(a.dtype), "md5": md5(a)})
        print(f"[OK] {name}  done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--low-dir", default=None)
    ap.add_argument("--high-dir", default=None)
    ap.add_argument("--n-real", type=int, default=15)
    ap.add_argument("--size", default="600x400")
    ap.add_argument("--out", default="golden")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-npy", action="store_true")
    args = ap.parse_args()

    H, W = parse_size(args.size)
    model = load_model(args.model_dir, args.ckpt, "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ex = Exporter(args.out, (H, W), args.seed, args.skip_npy)

    def infer(x):
        with torch.no_grad():
            xp = pad_to_multiple(x, 4)
            y = model(xp)[:, :, :H, :W]
        return x, y

    # 1) 确定性合成向量
    xs = torch.rand(1, 3, H, W)
    xg = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(1, 3, H, W)
    xsyn = 0.8 * xs + 0.2 * (xg ** 2)
    ex.save("synthetic", *infer(xsyn))

    # 2) 真实 LOL 测试图
    if args.low_dir and os.path.isdir(args.low_dir):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = sorted(f for f in os.listdir(args.low_dir) if f.lower().endswith(exts))[: args.n_real]
        for f in files:
            name = os.path.splitext(f)[0]
            low = np.array(Image.open(os.path.join(args.low_dir, f)).convert("RGB"))
            if low.shape[:2] != (H, W):
                low = np.array(Image.open(os.path.join(args.low_dir, f)).convert("RGB").resize((W, H)))
            gt = None
            if args.high_dir and os.path.isdir(args.high_dir):
                hp = os.path.join(args.high_dir, f)
                if os.path.exists(hp):
                    gt = np.array(Image.open(hp).convert("RGB"))
                    if gt.shape[:2] != (H, W):
                        gt = np.array(Image.open(hp).convert("RGB").resize((W, H)))
                    gt = gt.astype(np.float32) / 255.0
            x = torch.from_numpy(low.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            ex.save(name, *infer(x), gt=gt)

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"size": f"{W}x{H}", "seed": args.seed, "vectors": ex.manifest},
                  f, ensure_ascii=False, indent=2)
    print(f"\nDONE -> {args.out}/")


if __name__ == "__main__":
    main()