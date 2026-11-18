#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1-3.2 统计显著性检验脚本。
对 15 张 LOL-eval15 测试图，逐图计算 PSNR / SSIM / PIQE / NIQE（不做离群值剔除），
再对 "Our method" 与每个对比方法做配对检验：
  - 配对 t 检验（ttest_rel）
  - Wilcoxon 符号秩检验（非参数）
输出 results/significance_test.json 及可读表格。
"""
import os, sys, json
import numpy as np, cv2, torch
from scipy import stats

BASE = "/data/lowlight_paper_revision"
sys.path.insert(0, os.path.join(BASE, "code", "our_model"))

import pyiqa
from pypiqe import piqe as _piqe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REF = os.path.join(BASE, "data", "test", "high")

METHODS = {
    "Our method": os.path.join(BASE, "results", "final_enh", "distill_after_l1orig"),
    "DDNet (teacher)": os.path.join(BASE, "enhanced_outputs", "myDDNet"),
    "Zero-DCE": os.path.join(BASE, "enhanced_outputs", "Zero-DCE"),
    "SCI": os.path.join(BASE, "enhanced_outputs", "SCI"),
    "RUAS": os.path.join(BASE, "enhanced_outputs", "RUAS"),
    "HE (CLAHE)": os.path.join(BASE, "enhanced_outputs", "CHAHE"),
    "Dong": os.path.join(BASE, "enhanced_outputs", "Dong"),
}

psnr_m = pyiqa.create_metric("psnr", device=DEVICE)
ssim_m = pyiqa.create_metric("ssim", device=DEVICE)
niqe_path = "/home/user/.cache/torch/hub/pyiqa/niqe_modelparameters.mat"
niqe_m = pyiqa.create_metric("niqe", pretrained_model_path=niqe_path, device=DEVICE) if os.path.exists(niqe_path) else None


def rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_tensor(path):
    r = rgb(path)
    return torch.from_numpy(r).float().permute(2, 0, 1).unsqueeze(0) / 255.0


refs = sorted(os.listdir(REF))
scores = {}
for name, d in METHODS.items():
    ps, ss, pq, nq = [], [], [], []
    for f in refs:
        rp, dp = os.path.join(REF, f), os.path.join(d, f)
        with torch.no_grad():
            dt, rt = to_tensor(dp).to(DEVICE), to_tensor(rp).to(DEVICE)
            ps.append(psnr_m(dt, rt).item())
            ss.append(ssim_m(dt, rt).item())
        pq.append(float(_piqe(rgb(dp))[0]))
        if niqe_m is not None:
            with torch.no_grad():
                nq.append(niqe_m(to_tensor(dp).to(DEVICE)).item())
    scores[name] = {"psnr": ps, "ssim": ss, "piqe": pq, "niqe": nq}

BASE_NAME = "Our method"
out = {"n": len(refs), "metrics": {}, "paired_tests": {}}
for m in METHODS:
    out["metrics"][m] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in scores[m].items()}

for metric in ["psnr", "ssim", "piqe", "niqe"]:
    bx = np.array(scores[BASE_NAME][metric], dtype=float)
    out["paired_tests"][metric] = {}
    for m in METHODS:
        if m == BASE_NAME:
            continue
        mx = np.array(scores[m][metric], dtype=float)
        _, tp = stats.ttest_rel(bx, mx)
        wp = stats.wilcoxon(bx, mx)[1]
        out["paired_tests"][metric][m] = {"ttest_p": float(tp), "wilcoxon_p": float(wp)}

os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
json.dump(out, open(os.path.join(BASE, "results", "significance_test.json"), "w"), indent=2)

# ---- 可读输出 ----
def pfmt(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

print("=" * 78)
print("Per-image descriptive statistics (n=15, no outlier removal)")
print("=" * 78)
print(f"{'Method':<20}{'PSNR':>12}{'SSIM':>12}{'PIQE':>12}{'NIQE':>12}")
for m in METHODS:
    mm = out["metrics"][m]
    print(f"{m:<20}{mm['psnr']['mean']:>8.2f}±{mm['psnr']['std']:>4.2f}"
          f"{mm['ssim']['mean']:>8.3f}±{mm['ssim']['std']:>4.3f}"
          f"{mm['piqe']['mean']:>8.2f}±{mm['piqe']['std']:>4.2f}"
          f"{mm['niqe']['mean']:>8.2f}±{mm['niqe']['std']:>4.2f}")

print()
print("=" * 78)
print("Paired significance tests:  Our method  vs  each baseline")
print("=" * 78)
for metric, title in [("psnr", "PSNR (higher=better)"), ("ssim", "SSIM (higher=better)"),
                      ("piqe", "PIQE (lower=better)"), ("niqe", "NIQE (lower=better)")]:
    print(f"\n--- {title} ---")
    print(f"{'Baseline':<20}{'t-test p':>12}{'Wilcoxon p':>12}{'significant?':>14}")
    for m in METHODS:
        if m == BASE_NAME:
            continue
        d = out["paired_tests"][metric][m]
        sig = "YES" if (d["ttest_p"] < 0.05 and d["wilcoxon_p"] < 0.05) else "no"
        print(f"{m:<20}{pfmt(d['ttest_p']):>12}{pfmt(d['wilcoxon_p']):>12}{sig:>14}")

print("\nDONE -> results/significance_test.json")