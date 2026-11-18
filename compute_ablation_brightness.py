#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 ablation_eval.py 已保存的两阶段增强图计算平均亮度（0-255）等亮度指标，
用于回填 Table V 的 Avg. Luminance 列。协议与 unified_evaluate.py 一致（Tukey 剔除）。"""
import os, json
import numpy as np
import cv2

BASE = "/data/lowlight_paper_revision"
ENH_DIR = os.path.join(BASE, "results", "ablation_enhanced")
MODELS = ["without_attention", "without_adaptive_residual", "without_skip", "reduced_channels"]


def _tukey(s):
    q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def mean_std(s):
    s = np.array(s, dtype=float)
    if len(s) >= 4:
        lo, hi = _tukey(s)
        s = s[(s >= lo) & (s <= hi)]
    return float(np.mean(s)), float(np.std(s))


def brightness(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(np.mean(gray.astype(np.float32)))


out = {}
for m in MODELS:
    d = os.path.join(ENH_DIR, m)
    files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
    vals = [brightness(os.path.join(d, f)) for f in files]
    mu, sd = mean_std(vals)
    out[m] = {"mean": round(mu, 2), "std": round(sd, 2), "n": len(vals)}
    print(m, out[m], "raw_n=", len(files))

print(json.dumps(out, ensure_ascii=False, indent=2))