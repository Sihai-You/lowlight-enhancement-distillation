#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨数据集全参考（zero-shot）评估：
  冻结学生模型 distill_after_l1orig.pth，在 LOL-v2-real（100 对）与 MIT-FiveK（子集）
  上做原始分辨率推理，用与论文完全一致的 FullReferenceAssessor 计算 PSNR/SSIM，
  逐图 Tukey 离群值剔除后输出 mean±std。

  口径与 eval_final.py / unified_evaluate.py 一致：原始分辨率输入 + pyiqa(亮度 0-255) PSNR/SSIM。
  用法（远程）：conda activate py3_9 && python cross_dataset_eval.py
"""
import os, sys, json, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from datetime import datetime
import numpy as np
import torch
from PIL import Image

BASE = "/data/lowlight_paper_revision"
sys.path.insert(0, os.path.join(BASE, "code", "evaluation"))
sys.path.insert(0, os.path.join(BASE, "code", "our_model"))

from unified_evaluate import (FullReferenceAssessor, mean_std, DEVICE, REMOVE_OUTLIERS)

CKPT = os.path.join(BASE, "models", "staged", "distill_after_l1orig.pth")
ENH_ROOT = os.path.join(BASE, "results", "cross_dataset_enh")
OUT_JSON = os.path.join(BASE, "results", "cross_dataset_fullref.json")

DT = {
    "lolv2_real": {
        "low": os.path.join(BASE, "data", "lolv2_real", "low"),
        "high": os.path.join(BASE, "data", "lolv2_real", "high"),
        "token_len": 5,   # 取文件名中 5 位数字
        "max_side": None, # 600x400 原分辨率
    },
    "mit_fivek": {
        "low": os.path.join(BASE, "data", "mit_fivek", "input_srgb"),
        "high": os.path.join(BASE, "data", "mit_fivek", "gt_srgb"),
        "token_len": 5,   # a0001 -> 取 5 位编号（含 a 前缀）
        "max_side": 1024, # 3008x2000 超出模型/显存，按长边 1024 等比缩放
    },
}
EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def key_of(fn):
    """从文件名中抽出用于配对的编号（连续数字串）。"""
    digits = "".join(ch for ch in fn if ch.isdigit())
    return digits


def pair_paths(low_dir, high_dir):
    lows = sorted(f for f in os.listdir(low_dir) if f.lower().endswith(EXTS))
    highs = sorted(f for f in os.listdir(high_dir) if f.lower().endswith(EXTS))
    hmap = {key_of(f): f for f in highs}
    pairs = []
    missing = []
    for f in lows:
        k = key_of(f)
        if k in hmap:
            pairs.append((os.path.join(low_dir, f), os.path.join(high_dir, hmap[k])))
        else:
            missing.append((f, k))
    return pairs, missing, (len(lows), len(highs))


def load_model():
    from models import AblationModelFactory
    model = AblationModelFactory.create_model("full").to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("student_state_dict", "model_state_dict", "state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def run_inference(model, pairs, out_dir, max_side=None):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    with torch.no_grad():
        for i, (low_path, high_path) in enumerate(pairs):
            img = Image.open(low_path).convert("RGB")
            if max_side is not None:
                w, h = img.size
                if max(w, h) > max_side:
                    scale = max_side / float(max(w, h))
                    img = img.resize((max(1, int(round(w * scale))),
                                      max(1, int(round(h * scale)))), Image.BILINEAR)
            t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
            t = t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            _, _, H, W = t.shape
            # 模型含两次 stride-2 下采样，需 H/W 为 8 的倍数才可对齐跳跃连接，
            # 对不满足的尺寸做 reflect 填充，推理后裁剪回原尺寸。
            pad_h = ((H + 7) // 8) * 8 - H
            pad_w = ((W + 7) // 8) * 8 - W
            if pad_h or pad_w:
                t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
            out = model(t)
            out = out[:, :, :H, :W]
            out = torch.clamp(out, 0.0, 1.0)
            out_np = out[0].permute(1, 2, 0).cpu().numpy()
            out_np = (out_np * 255.0).astype(np.uint8)
            name = os.path.basename(low_path)
            save_path = os.path.join(out_dir, name)
            Image.fromarray(out_np).save(save_path)
            saved.append((save_path, high_path))
            # 不同图像尺寸会导致缓存块碎片化累积，逐图释放显存
            del t, out, out_np
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (i + 1) % 20 == 0:
                print(f"    infer {i+1}/{len(pairs)}")
    return saved


def main():
    model = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    assessor = FullReferenceAssessor(DEVICE)

    result = {
        "ckpt": CKPT,
        "device": DEVICE,
        "remove_outliers": REMOVE_OUTLIERS,
        "timestamp": datetime.now().isoformat(),
        "params": int(n_params),
        "datasets": {},
    }

    for ds, cfg in DT.items():
        print(f"\n===== {ds} =====")
        pairs, missing, (nlow, nhigh) = pair_paths(cfg["low"], cfg["high"])
        print(f"  low={nlow} high={nhigh} matched={len(pairs)} missing={len(missing)}")
        if missing:
            print("  missing pairs:", missing[:10])
        out_dir = os.path.join(ENH_ROOT, ds)
        saved = run_inference(model, pairs, out_dir, max_side=cfg.get("max_side"))

        psnr, ssim = [], []
        for dist_path, ref_path in saved:
            p, s = assessor.psnr_ssim(dist_path, ref_path)
            psnr.append(p); ssim.append(s)

        psnr_m, psnr_s = mean_std(psnr, REMOVE_OUTLIERS)
        ssim_m, ssim_s = mean_std(ssim, REMOVE_OUTLIERS)
        entry = {
            "n_pairs": len(saved),
            "psnr": {"mean": round(psnr_m, 4), "std": round(psnr_s, 4)},
            "ssim": {"mean": round(ssim_m, 4), "std": round(ssim_s, 4)},
            "psnr_list": [round(v, 4) for v in psnr],
            "ssim_list": [round(v, 4) for v in ssim],
        }
        result["datasets"][ds] = entry
        print(json.dumps({k: entry[k] for k in ("n_pairs", "psnr", "ssim")}, indent=2))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nDONE -> {OUT_JSON}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()