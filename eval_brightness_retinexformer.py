#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复刻 unified_evaluate.py 的 brightness_metrics + mean_std/median_iqr，
对 retinexformer_eval15 增强图计算亮度特性与亮度分布指标。"""
import os, json
import numpy as np
import cv2
from scipy import stats

ENH = "/workspace/lowlight_paper_revision/results/retinexformer_eval15"
REMOVE_OUTLIERS = True

def brightness_metrics(path):
    img = cv2.imread(path)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    b = gray.astype(np.float32)
    max_v, min_v = float(np.max(b)), float(np.min(b))
    michelson = (max_v - min_v) / (max_v + min_v) if (max_v + min_v) > 0 else 0.0
    hist, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
    hist = hist / hist.sum()
    entropy = -float(np.sum(hist * np.log2(hist + 1e-10)))
    over = float(np.sum(gray > 240) / gray.size)
    under = float(np.sum(gray < 15) / gray.size)
    h, w = gray.shape
    block_h, block_w = h // 4, w // 4
    blocks = []
    if block_h > 0 and block_w > 0:
        for i in range(4):
            for j in range(4):
                blk = gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                if blk.size:
                    blocks.append(np.mean(blk))
    uniformity = 1 - (np.std(blocks) / (np.mean(blocks) + 1e-10)) if blocks else 0.0
    skew = float(stats.skew(b.flatten()))
    kurt = float(stats.kurtosis(b.flatten()))
    return {
        "mean_brightness": float(np.mean(b)),
        "std_brightness": float(np.std(b)),
        "dynamic_range": float(max_v - min_v),
        "michelson_contrast": michelson,
        "over_exposed_ratio": over,
        "under_exposed_ratio": under,
        "histogram_entropy": entropy,
        "brightness_uniformity": uniformity,
        "brightness_skewness": skew,
        "brightness_kurtosis": kurt,
    }

def _tukey(s):
    q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
    iqr = q3 - q1
    return q1 - 1.5*iqr, q3 + 1.5*iqr

def mean_std(s, remove):
    s = np.array(s, dtype=float)
    if remove and len(s) >= 4:
        lo, hi = _tukey(s)
        s = s[(s >= lo) & (s <= hi)]
    return float(np.mean(s)), float(np.std(s)), int(len(s))

def median_iqr(s, remove):
    s = np.array(s, dtype=float)
    if remove and len(s) >= 4:
        lo, hi = _tukey(s)
        s = s[(s >= lo) & (s <= hi)]
    return float(np.median(s)), float(np.percentile(s, 25)), float(np.percentile(s, 75)), int(len(s))

files = sorted(f for f in os.listdir(ENH) if f.lower().endswith('.png'))
allb = {k: [] for k in brightness_metrics(os.path.join(ENH, files[0])).keys()}
for f in files:
    bm = brightness_metrics(os.path.join(ENH, f))
    for k, v in bm.items():
        allb[k].append(v)

out = {}
for k in ("mean_brightness", "std_brightness", "michelson_contrast",
          "histogram_entropy", "brightness_skewness", "brightness_kurtosis"):
    m, sd, n = mean_std(allb[k], REMOVE_OUTLIERS)
    out[k] = {"mean": round(m, 4), "std": round(sd, 4), "n": n}
for k in ("brightness_uniformity", "dynamic_range", "over_exposed_ratio", "under_exposed_ratio"):
    med, q1, q3, n = median_iqr(allb[k], REMOVE_OUTLIERS)
    out[k] = {"median": round(med, 4), "q1": round(q1, 4), "q3": round(q3, 4), "n": n}

print(json.dumps(out, ensure_ascii=False, indent=2))
with open("/workspace/lowlight_paper_revision/results/retinexformer_brightness.json", "w") as fp:
    json.dump(out, fp, ensure_ascii=False, indent=2)
print("saved -> retinexformer_brightness.json")