#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用「增强图像目录 vs GT 目录」统一评估：对已保存的增强 PNG 与 eval15/high 计算
PSNR/SSIM/PIQE/NIQE，协议与 loss_ablation_eval.py / multiseed_eval.py 完全一致：
    原始分辨率 + pyiqa(PSNR/SSIM/NIQE) + pypiqe(PIQE) + Tukey 离群剔除 + 亮度 0-255
用法：python eval_enhanced_dir.py <enh_dir> <gt_dir> <out_json>
"""
import os
import sys
import json
from datetime import datetime
import numpy as np
import cv2
import torch
from PIL import Image

ENH_DIR, GT_DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REMOVE_OUTLIERS = True


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

    enh = sorted(f for f in os.listdir(ENH_DIR) if exts(f))
    gt = sorted(f for f in os.listdir(GT_DIR) if exts(f))
    common = [f for f in enh if f in gt]
    ps, ss, pq, nq = [], [], [], []
    with torch.no_grad():
        for f in common:
            e = np.array(Image.open(os.path.join(ENH_DIR, f)).convert("RGB")).astype(np.float32)
            r = to_rgb(cv2.imread(os.path.join(GT_DIR, f)))
            d = torch.from_numpy(e).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            rr = torch.from_numpy(r).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            d, rr = d.to(DEVICE), rr.to(DEVICE)
            ps.append(psnr_m(d, rr).item())
            ss.append(ssim_m(d, rr).item())
            pq.append(float(_piqe(e.astype(np.uint8))[0]))
            if niqe_m is not None:
                nq.append(niqe_m(d).item())

    res = {}
    for k, v in (("psnr", ps), ("ssim", ss), ("piqe", pq), ("niqe", nq)):
        m, sd, n = mean_std(v)
        res[k] = {"mean": round(m, 4), "std": round(sd, 4), "n": n}
    out = {"timestamp": datetime.now().isoformat(), "enh_dir": ENH_DIR,
           "n_images": len(common), "metrics": res}
    with open(OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    print("DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()