#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对单个新 checkpoint 用统一协议在 LOL-eval15 上复测 PSNR/SSIM/PIQE/NIQE。
口径与 unified_evaluate.py 完全一致（原始分辨率 + pyiqa + pypiqe + Tukey 剔除）。

用法：
  python eval_checkpoint.py --ckpt /data/.../recon_l1_0.5.pth --name recon_l1_0.5 \
      --out /data/lowlight_paper_revision/results/staged_metrics.json
"""
import os, sys, json, argparse, time
from datetime import datetime
import numpy as np
import torch
from PIL import Image

BASE = "/data/lowlight_paper_revision"
sys.path.insert(0, os.path.join(BASE, "code", "evaluation"))
sys.path.insert(0, os.path.join(BASE, "code", "our_model"))

from unified_evaluate import (FullReferenceAssessor, mean_std, median_iqr,
                              DEVICE, REMOVE_OUTLIERS, list_test_pairs, run_inference)

TEST_LOW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))  # placeholder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--low", type=str, default="/home/user/YouSihai/lightEnhancement/data/Dataset_test/LOLDataset/eval15/low")
    ap.add_argument("--high", type=str, default="/home/user/YouSihai/lightEnhancement/data/Dataset_test/LOLDataset/eval15/high")
    ap.add_argument("--enh", type=str, default="/data/lowlight_paper_revision/results/staged_enh")
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
    for dist_path, ref_path in pairs:
        psnr, ssim = assessor.psnr_ssim(dist_path, ref_path)
        per["psnr"].append(psnr); per["ssim"].append(ssim)
        per["piqe"].append(assessor.piqe(dist_path))
        niqe = assessor.niqe(dist_path)
        if niqe is not None:
            per["niqe"].append(niqe)

    result = {"name": args.name, "ckpt": args.ckpt,
              "timestamp": datetime.now().isoformat(), "device": DEVICE}
    result["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
    for k in ("psnr", "ssim", "piqe", "niqe"):
        if per[k]:
            m, s = mean_std(per[k], REMOVE_OUTLIERS)
            result[k] = {"mean": round(m, 4), "std": round(s, 4), "n": len(per[k])}
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if os.path.exists(args.out):
        with open(args.out) as f:
            allr = json.load(f)
    else:
        allr = {}
    allr[args.name] = result
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(allr, f, ensure_ascii=False, indent=2)
    print(f"\n已写出: {args.out}")


if __name__ == "__main__":
    main()