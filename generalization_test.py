#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1-3.5 跨数据集泛化实验（zero-shot）。
用仅在 LOL 上训练的学生模型(full)，直接推理 DICM(64 张) 与 LIME(10 张) 两个
未见过的低光基准测试集，计算无参考指标 PIQE / NIQE（越低越好），
并与"原始低光输入"（Input，无增强）对比，验证模型的跨数据集泛化能力。
输出 results/generalization_test.json + results/generalization/{dicm,lime}/ 增强图。
"""
import os, sys, json
import numpy as np
import cv2
import torch

BASE = "/data/lowlight_paper_revision"
PROJECT = "/home/user/YouSihai/lightEnhancement/newStudent"
sys.path.insert(0, PROJECT)

import pyiqa
from pypiqe import piqe as _piqe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = os.path.join(PROJECT, "models", "ablation_full.pth")
DICM_DIR = "/home/user/Zero-DCE/Zero-DCE/Zero-DCE_code/data/test_data/DICM"
LIME_DIR = "/home/user/Zero-DCE/Zero-DCE/Zero-DCE_code/data/test_data/LIME"
OUT_ROOT = os.path.join(BASE, "results", "generalization")

niqe_path = "/home/user/.cache/torch/hub/pyiqa/niqe_modelparameters.mat"
niqe_m = pyiqa.create_metric("niqe", pretrained_model_path=niqe_path, device=DEVICE) \
        if os.path.exists(niqe_path) else None


def load_model():
    from models import AblationModelFactory
    m = AblationModelFactory.create_model("full").to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("student_state_dict", "model_state_dict", "state_dict"):
            if k in ckpt:
                ckpt = ckpt[k]; break
    m.load_state_dict(ckpt, strict=False)
    m.eval()
    return m


def rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_tensor(path):
    r = rgb(path)
    return torch.from_numpy(r).float().permute(2, 0, 1).unsqueeze(0) / 255.0


def pad_to_multiple(t, m=16):
    """对 H/W 反射填充到 m 的整数倍，避免解码器 skip 连接尺寸不匹配。"""
    _, _, h, w = t.shape
    pad_h = (m - h % m) % m
    pad_w = (m - w % m) % m
    if pad_h or pad_w:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    return t


def crop_to(t, h, w):
    return t[:, :, :h, :w]


def no_ref(path):
    """返回 (PIQE, NIQE)。"""
    r = rgb(path)
    pq = float(_piqe(r)[0])
    nq = None
    if niqe_m is not None:
        t = torch.from_numpy(r).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            nq = niqe_m(t.to(DEVICE)).item()
    return pq, nq


def enhance_and_score(model, dataset_name, low_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = sorted(f for f in os.listdir(low_dir) if f.lower().endswith(exts))
    out_dir = os.path.join(OUT_ROOT, dataset_name.lower())
    os.makedirs(out_dir, exist_ok=True)
    input_piqe, input_niqe, our_piqe, our_niqe = [], [], [], []
    for f in files:
        lp = os.path.join(low_dir, f)
        ip, inq = no_ref(lp)
        input_piqe.append(ip)
        if inq is not None:
            input_niqe.append(inq)
        t = to_tensor(lp)
        _, _, H, W = t.shape
        with torch.no_grad():
            tp = pad_to_multiple(t, 16).to(DEVICE)
            out = model(tp)
            out = crop_to(out, H, W)
            out = torch.clamp(out, 0.0, 1.0)
        out_np = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        sp = os.path.join(out_dir, os.path.splitext(f)[0] + ".png")
        cv2.imwrite(sp, cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR))
        op, onq = no_ref(sp)
        our_piqe.append(op)
        if onq is not None:
            our_niqe.append(onq)
    return {
        "n": len(files),
        "input": {"piqe_mean": float(np.mean(input_piqe)), "piqe_std": float(np.std(input_piqe)),
                  "niqe_mean": float(np.mean(input_niqe)), "niqe_std": float(np.std(input_niqe))},
        "ours":  {"piqe_mean": float(np.mean(our_piqe)),  "piqe_std": float(np.std(our_piqe)),
                  "niqe_mean": float(np.mean(our_niqe)),  "niqe_std": float(np.std(our_niqe))},
    }


def main():
    model = load_model()
    out = {"device": DEVICE, "note": "PIQE|NIQE lower is better; zero-shot, trained on LOL only"}
    for name, d in [("DICM", DICM_DIR), ("LIME", LIME_DIR)]:
        print(f"===== {name} ({d}) =====")
        out[name] = enhance_and_score(model, name, d)
        print(json.dumps(out[name], indent=2))

    os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
    with open(os.path.join(BASE, "results", "generalization_test.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nDONE -> results/generalization_test.json")


if __name__ == "__main__":
    main()