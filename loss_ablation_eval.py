#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两阶段损失消融评估。在统一协议下评估 full + 5 个损失项消融变体：
    full / no_distill / no_perceptual / no_brightness / no_contrast / no_kd
协议与 ablation_eval.py / unified_evaluate.py 完全一致：
    原始分辨率 + pyiqa(PSNR/SSIM/NIQE) + pypiqe(PIQE) + Tukey 离群剔除 + 亮度 0-255
输出：/data/lowlight_paper_revision/results/loss_ablation_eval_results.json
"""
import os
import sys
import json
from datetime import datetime
import numpy as np
import cv2
import torch
from PIL import Image

BASE = "/data/lowlight_paper_revision"
sys.path.insert(0, os.path.join(BASE, "code", "our_model"))

from models import AblationModelFactory  # noqa: E402

TEST_LOW = "/home/user/YouSihai/lightEnhancement/data/Dataset_test/LOLDataset/eval15/low"
TEST_HIGH = "/home/user/YouSihai/lightEnhancement/data/Dataset_test/LOLDataset/eval15/high"
CKPT_DIR = os.path.join(BASE, "models", "staged")

# (key, 显示名, stage2 checkpoint 文件名)
MODELS = [
    ("full",          "Full",                   "distill_after_l1orig.pth"),
    ("no_distill",    "w/o Cosine (L_cos)",     "s2_loss_no_distill.pth"),
    ("no_perceptual", "w/o Perceptual (L_per)", "s2_loss_no_perceptual.pth"),
    ("no_brightness", "w/o Brightness (L_bri)", "s2_loss_no_brightness.pth"),
    ("no_contrast",   "w/o Contrast (L_con)",   "s2_loss_no_contrast.pth"),
    ("no_kd",         "Pure Supervised (no KD)", "s2_loss_no_kd.pth"),
]
MODEL_TYPE = "full"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REMOVE_OUTLIERS = True


def load_model(ckpt):
    model = AblationModelFactory.create_model(MODEL_TYPE).to(DEVICE)
    path = os.path.join(CKPT_DIR, ckpt)
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    for key in ("student_state_dict", "model_state_dict", "state_dict"):
        if isinstance(ck, dict) and key in ck:
            ck = ck[key]
            break
    model.load_state_dict(ck, strict=False)
    model.eval()
    return model


def count_params(model):
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 4)


def exts(f):
    return f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)


def _tukey(s):
    q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def mean_std(s):
    s = np.array(s, dtype=float)
    if REMOVE_OUTLIERS and len(s) >= 4:
        lo, hi = _tukey(s)
        s = s[(s >= lo) & (s <= hi)]
    return float(np.mean(s)), float(np.std(s))


def main():
    import pyiqa
    from pypiqe import piqe as _piqe
    psnr_m = pyiqa.create_metric("psnr", device=DEVICE)
    ssim_m = pyiqa.create_metric("ssim", device=DEVICE)
    niqe_path = "/home/user/.cache/torch/hub/pyiqa/niqe_modelparameters.mat"
    niqe_m = pyiqa.create_metric("niqe", pretrained_model_path=niqe_path, device=DEVICE) if os.path.exists(niqe_path) else None

    lows = sorted(f for f in os.listdir(TEST_LOW) if exts(f))
    highs = sorted(f for f in os.listdir(TEST_HIGH) if exts(f))
    common = [f for f in lows if f in highs]
    print(f"[eval] test images = {len(common)}", flush=True)

    out = {"timestamp": datetime.now().isoformat(),
           "protocol": "two-stage; pyiqa PSNR/SSIM/NIQE + pypiqe PIQE + Tukey trim + 0-255",
           "models": {}}
    for name, disp, ckpt in MODELS:
        model = load_model(ckpt)
        ps, ss, pq, nq = [], [], [], []
        with torch.no_grad():
            for f in common:
                low = np.array(Image.open(os.path.join(TEST_LOW, f)).convert("RGB")).astype(np.float32) / 255.0
                low_t = torch.from_numpy(low).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                enh = torch.clamp(model(low_t), 0.0, 1.0)
                enh_np = (enh[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                ref_np = to_rgb(cv2.imread(os.path.join(TEST_HIGH, f)))
                d = torch.from_numpy(enh_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                r = torch.from_numpy(ref_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                d, r = d.to(DEVICE), r.to(DEVICE)
                ps.append(psnr_m(d, r).item())
                ss.append(ssim_m(d, r).item())
                pq.append(float(_piqe(enh_np)[0]))
                if niqe_m is not None:
                    nq.append(niqe_m(d).item())
        res = {"display": disp, "params_m": count_params(model)}
        for k, v in (("psnr", ps), ("ssim", ss), ("piqe", pq), ("niqe", nq)):
            m, sd = mean_std(v)
            res[k] = {"mean": round(m, 4), "std": round(sd, 4), "n": len(v)}
        out["models"][name] = res
        print(json.dumps({name: res}, ensure_ascii=False), flush=True)

    outp = os.path.join(BASE, "results", "loss_ablation_eval_results.json")
    with open(outp, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nDONE -> {outp}", flush=True)


if __name__ == "__main__":
    main()