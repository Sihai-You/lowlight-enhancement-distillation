#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多随机种子评估：对 distill_seed42/43/44.pth 在 LOL eval15 上统一评估，
输出每个种子的 PSNR/SSIM/PIQE/NIQE（图像级 mean±std），
并计算跨 3 个种子的 mean±std（种子级，样本标准差 ddof=1）。
协议与 loss_ablation_eval.py / unified_evaluate.py 完全一致：
    原始分辨率 + pyiqa(PSNR/SSIM/NIQE) + pypiqe(PIQE) + Tukey 离群剔除 + 亮度 0-255
输出：/data/lowlight_paper_revision/results/multiseed_eval_results.json
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
    ("seed42", "Seed 42", "distill_seed42.pth"),
    ("seed43", "Seed 43", "distill_seed43.pth"),
    ("seed44", "Seed 44", "distill_seed44.pth"),
]
MODEL_TYPE = "full"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REMOVE_OUTLIERS = True
METRICS = ("psnr", "ssim", "piqe", "niqe")


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
    return float(np.mean(s)), float(np.std(s)), int(len(s))


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
           "n_seeds": len(MODELS),
           "per_seed": {},
           "cross_seed": {}}

    # 收集每个种子在每个图像上的指标，便于跨种子聚合
    seed_vals = {}        # metric -> {seed: [per-image values]}
    seed_summary = {}     # seed -> {metric: {mean, std, n}}
    for name, disp, ckpt in MODELS:
        model = load_model(ckpt)
        acc = {m: [] for m in METRICS}
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
                acc["psnr"].append(psnr_m(d, r).item())
                acc["ssim"].append(ssim_m(d, r).item())
                acc["piqe"].append(float(_piqe(enh_np)[0]))
                if niqe_m is not None:
                    acc["niqe"].append(niqe_m(d).item())

        res = {"display": disp, "params_m": count_params(model)}
        for m in METRICS:
            if not acc[m]:
                continue
            mean, sd, n = mean_std(acc[m])
            res[m] = {"mean": round(mean, 4), "std": round(sd, 4), "n": n}
            seed_vals.setdefault(m, {})[name] = acc[m]
        seed_summary[name] = res
        out["per_seed"][name] = res
        print(json.dumps({name: {m: res.get(m) for m in METRICS}}, ensure_ascii=False), flush=True)

    # 跨种子聚合：每种指标，取各“种子内 Tukey 剔除后的均值”，再求跨种子 mean±std
    for m in METRICS:
        means = [seed_summary[s][m]["mean"] for s, _d, _c in MODELS if m in seed_summary[s]]
        means = np.array(means, dtype=float)
        n = len(means)
        grand_mean = float(np.mean(means))
        # 样本标准差（ddof=1），更能反映小样本下的运行间波动
        grand_std = float(np.std(means, ddof=1))
        out["cross_seed"][m] = {
            "mean": round(grand_mean, 4),
            "std_sample": round(grand_std, 4),
            "std_population": round(float(np.std(means)), 4),
            "n_seeds": n,
            "per_seed_means": [round(x, 4) for x in means],
        }

    outp = os.path.join(BASE, "results", "multiseed_eval_results.json")
    with open(outp, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nCROSS_SEED:")
    for m in METRICS:
        if m in out["cross_seed"]:
            cs = out["cross_seed"][m]
            print(f"  {m}: {cs['mean']} ± {cs['std_sample']} (n={cs['n_seeds']} seeds)", flush=True)
    print(f"\nDONE -> {outp}", flush=True)


if __name__ == "__main__":
    main()