#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 7 个对比方法（DDNet/Zero-DCE/SCI/RUAS/HE/Dong/…）的增强输出图，
用与 unified_evaluate.py 完全相同的统一协议重新评估，
输出 full_unified_metrics.json，供论文 Table II/III/IV 使用。

运行：conda activate py3_9 && python full_unified_eval.py
"""
import os, sys, json
from datetime import datetime
import numpy as np

BASE = "/data/lowlight_paper_revision"
sys.path.insert(0, os.path.join(BASE, "code", "evaluation"))
sys.path.insert(0, os.path.join(BASE, "code", "our_model"))

from unified_evaluate import FullReferenceAssessor, mean_std, median_iqr, DEVICE, REMOVE_OUTLIERS

TEST_HIGH = os.path.join(BASE, "data", "test", "high")
ENH = os.path.join(BASE, "enhanced_outputs")
OUR_METRICS = os.path.join(BASE, "results", "unified_results", "unified_metrics.json")

# 论文里的方法名 -> 增强输出目录名
COMPARISON = {
    "DDNet": "myDDNet",
    "Zero-DCE": "Zero-DCE",
    "SCI": "SCI",
    "RUAS": "RUAS",
    "HE": "CHAHE",      # 论文标 HE，服务器目录名为 CHAHE(CLAHE)
    "Dong": "Dong",
}

def load_our_full():
    with open(OUR_METRICS, "r", encoding="utf-8") as f:
        d = json.load(f)["models"]["full"]
    return d

def compare_run(assessor, name, enh_dir):
    refs = sorted(os.listdir(TEST_HIGH))
    res = {"psnr": [], "ssim": [], "piqe": [], "niqe": []}
    bkeys = ["mean_brightness", "std_brightness", "dynamic_range", "michelson_contrast",
             "over_exposed_ratio", "under_exposed_ratio", "histogram_entropy",
             "brightness_uniformity", "brightness_skewness", "brightness_kurtosis"]
    bright = {k: [] for k in bkeys}
    n = 0
    for fn in refs:
        dist_path = os.path.join(enh_dir, fn)
        ref_path = os.path.join(TEST_HIGH, fn)
        if not os.path.exists(dist_path):
            continue
        n += 1
        psnr, ssim = assessor.psnr_ssim(dist_path, ref_path)
        res["psnr"].append(psnr); res["ssim"].append(ssim)
        res["piqe"].append(assessor.piqe(dist_path))
        niqe = assessor.niqe(dist_path)
        if niqe is not None:
            res["niqe"].append(niqe)
        bm = assessor.brightness_metrics(dist_path)
        for k in bkeys:
            bright[k].append(bm[k])

    out = {"n_images": n}
    for k in ("psnr", "ssim", "piqe", "niqe"):
        if res[k]:
            m, s = mean_std(res[k], REMOVE_OUTLIERS)
            out[k] = {"mean": m, "std": s, "n": len(res[k])}
    for k in ("mean_brightness", "std_brightness", "michelson_contrast",
              "histogram_entropy", "brightness_skewness", "brightness_kurtosis"):
        m, s = mean_std(bright[k], REMOVE_OUTLIERS)
        out[k] = {"mean": m, "std": s}
    for k in ("brightness_uniformity", "dynamic_range", "over_exposed_ratio", "under_exposed_ratio"):
        med, q1, q3 = median_iqr(bright[k], REMOVE_OUTLIERS)
        out[k] = {"median": med, "q1": q1, "q3": q3}
    return out

def main():
    assessor = FullReferenceAssessor(DEVICE)
    out = {"timestamp": datetime.now().isoformat(), "device": DEVICE,
           "remove_outliers": REMOVE_OUTLIERS, "methods": {}}
    out["methods"]["Our method"] = load_our_full()
    for name, d in COMPARISON.items():
        enh_dir = os.path.join(ENH, d)
        print(f"\n===== {name} ({d}) =====")
        out["methods"][name] = compare_run(assessor, name, enh_dir)
        print(json.dumps({k: out["methods"][name][k] for k in ("psnr","ssim","piqe","niqe")}, indent=2))

    outp = os.path.join(BASE, "results", "full_unified_metrics.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写出: {outp}")

if __name__ == "__main__":
    main()