#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对新 checkpoint 做完整统一协议评估（与论文 Table II/III/IV 口径完全一致）：
PSNR/SSIM/PIQE/NIQE + 亮度统计 + 参数量 + V100 稳态耗时/FPS。
输出结构与 full_unified_metrics.json 中 "Our method" 一致。

用法: python eval_final.py --ckpt <path> --name <name> --out <json>
"""
import os, sys, json, argparse
from datetime import datetime
import numpy as np
import torch
from PIL import Image

BASE = "/data/lowlight_paper_revision"
sys.path.insert(0, os.path.join(BASE, "code", "evaluation"))
sys.path.insert(0, os.path.join(BASE, "code", "our_model"))

from unified_evaluate import (FullReferenceAssessor, mean_std, median_iqr,
                              DEVICE, REMOVE_OUTLIERS, list_test_pairs, run_inference,
                              measure_latency)

LATENCY_SIZE = (3, 256, 256)
LATENCY_ITER = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--low", type=str, default="/home/user/YouSihai/lightEnhancement/data/Dataset_test/LOLDataset/eval15/low")
    ap.add_argument("--high", type=str, default="/home/user/YouSihai/lightEnhancement/data/Dataset_test/LOLDataset/eval15/high")
    ap.add_argument("--enh", type=str, default="/data/lowlight_paper_revision/results/final_enh")
    args = ap.parse_args()

    from models import AblationModelFactory
    model = AblationModelFactory.create_model("full").to(DEVICE)
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    for k in ("student_state_dict", "model_state_dict", "state_dict"):
        if k in ckpt:
            ckpt = ckpt[k]
            break
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    out_dir = os.path.join(args.enh, args.name)
    pairs = run_inference(model, args.low, args.high, out_dir)

    assessor = FullReferenceAssessor(DEVICE)
    per = {"psnr": [], "ssim": [], "piqe": [], "niqe": []}
    bkeys = ["mean_brightness", "std_brightness", "dynamic_range", "michelson_contrast",
             "over_exposed_ratio", "under_exposed_ratio", "histogram_entropy",
             "brightness_uniformity", "brightness_skewness", "brightness_kurtosis"]
    bright = {k: [] for k in bkeys}

    for dist_path, ref_path in pairs:
        psnr, ssim = assessor.psnr_ssim(dist_path, ref_path)
        per["psnr"].append(psnr); per["ssim"].append(ssim)
        per["piqe"].append(assessor.piqe(dist_path))
        niqe = assessor.niqe(dist_path)
        if niqe is not None:
            per["niqe"].append(niqe)
        bm = assessor.brightness_metrics(dist_path)
        for k in bkeys:
            bright[k].append(bm[k])

    result = {"name": args.name, "ckpt": args.ckpt,
              "timestamp": datetime.now().isoformat(), "device": DEVICE,
              "remove_outliers": REMOVE_OUTLIERS}
    for k in ("psnr", "ssim", "piqe", "niqe"):
        if per[k]:
            m, s = mean_std(per[k], REMOVE_OUTLIERS)
            result[k] = {"mean": round(m, 4), "std": round(s, 4), "n": len(per[k])}
    for k in ("mean_brightness", "std_brightness", "michelson_contrast",
              "histogram_entropy", "brightness_skewness", "brightness_kurtosis"):
        m, s = mean_std(bright[k], REMOVE_OUTLIERS)
        result[k] = {"mean": round(m, 4), "std": round(s, 4)}
    for k in ("brightness_uniformity", "dynamic_range", "over_exposed_ratio", "under_exposed_ratio"):
        med, q1, q3 = median_iqr(bright[k], REMOVE_OUTLIERS)
        result[k] = {"median": round(med, 6), "q1": round(q1, 6), "q3": round(q3, 6)}

    # 参数量与 V100 稳态耗时（256×256）
    result["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 6)
    lat_mean, lat_std = measure_latency(model, LATENCY_SIZE, LATENCY_ITER)
    result["inference_ms"] = {"mean": round(lat_mean, 4), "std": round(lat_std, 4)}
    result["inference_fps"] = round(1000.0 / lat_mean, 2) if lat_mean > 0 else None

    print(json.dumps(result, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n已写出: {args.out}")


if __name__ == "__main__":
    main()