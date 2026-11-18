#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一低光增强评估脚本 —— 论文《Research on a Real-Time Low-Light Image
Enhancement Lightweight Network Based on Knowledge Distillation》配套程序。

解决的问题
============
论文当前存在两套互不一致的评估协议，导致同一模型在不同表格里出现矛盾：

1. ``newStudent/evaluator.py``（用于消融实验 Table V）
   - 使用 skimage.metrics 计算 PSNR / SSIM（data_range=255）；
   - 不做离群值剔除；
   - 亮度/对比度输出为 [0,1] 归一化值。
   - 结果：Full model PSNR=18.70、SSIM=0.7540。

2. ``student/img_accessment_py3.9.py``（用于 Table II/III/IV）
   - 使用 pyiqa 计算 PSNR / SSIM / NIQE，pypiqe 计算 PIQE；
   - 进行离群值剔除（REMOVE_OUTLIERS=True）；
   - 亮度输出为 0–255 自然量纲。
   - 结果：Our method PSNR=18.61、SSIM=0.83。

因此论文 Table II 与 Table V 的 PSNR、SSIM、亮度量纲互相矛盾。

本脚本把上述两套口径统一为**一套**：
   原始分辨率输入 + pyiqa(PSNR/SSIM/NIQE) + pypiqe(PIQE) + 亮度统计 + 离群值剔除
并额外计算参数量与稳态推理耗时 / FPS，输出 ``unified_metrics.json``，
供 ``generate_tables.py`` 重新生成论文表格（Table II~VIII）。

运行环境
========
本脚本需在远程服务器上运行（含 GPU 与 pyiqa/pypiqe/torch/cv2/skimage/scipy）。
推荐 conda 环境：py3_9。

    conda activate py3_9
    python unified_evaluate.py

路径说明
========
把下面的 PROJECT_DIR / DATA_DIR / MODEL_DIR 改成你机器上的真实路径即可。
脚本默认按 /home/user/YouSihai/lightEnhancement 与 /home/user/data 组织。
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np
import cv2
import torch
from PIL import Image

# ---------- 可配置路径 ----------
PROJECT_DIR = "/home/user/YouSihai/lightEnhancement/newStudent"
DATA_DIR = "/home/user/YouSihai/lightEnhancement/data"
MODEL_DIR = "/home/user/YouSihai/lightEnhancement/newStudent/models"

TEST_LOW_DIR = os.path.join(DATA_DIR, "Dataset_test/LOLDataset/eval15/low")
TEST_HIGH_DIR = os.path.join(DATA_DIR, "Dataset_test/LOLDataset/eval15/high")

# 模型类型 -> 对应 checkpoint 文件名
MODEL_TYPES = {
    "full": ("full", os.path.join(MODEL_DIR, "ablation_full.pth")),
    "without_attention": ("without_attention", os.path.join(MODEL_DIR, "ablation_without_attention.pth")),
    "without_adaptive_residual": ("without_adaptive_residual", os.path.join(MODEL_DIR, "ablation_without_adaptive_residual.pth")),
    "reduced_channels": ("reduced_channels", os.path.join(MODEL_DIR, "ablation_reduced_channels.pth")),
    "without_skip": ("without_skip", os.path.join(MODEL_DIR, "ablation_without_skip.pth")),
}

OUTPUT_ROOT = "/home/user/YouSihai/lightEnhancement/unified_results"
OUTPUT_JSON = os.path.join(OUTPUT_ROOT, "unified_metrics.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REMOVE_OUTLIERS = True
LATENCY_ITER = 50          # 稳态推理计时次数（排除预热）
LATENCY_SIZE = (3, 256, 256)

# 让脚本能 import 远程项目里的模型定义
sys.path.insert(0, PROJECT_DIR)


def get_model_factory():
    """延迟导入远程项目的 AblationModelFactory，保证架构与论文一致。"""
    from models import AblationModelFactory
    return AblationModelFactory


def load_model(model_type, checkpoint_path):
    factory = get_model_factory()
    model = factory.create_model(model_type).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("student_state_dict", "model_state_dict", "state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def list_test_pairs(low_dir, high_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    lows = sorted(f for f in os.listdir(low_dir) if f.lower().endswith(exts))
    highs = sorted(f for f in os.listdir(high_dir) if f.lower().endswith(exts))
    common = [f for f in lows if f in highs]
    assert len(common) == 15, f"期望 15 张测试图，实际 {len(common)} 张"
    return [(os.path.join(low_dir, f), os.path.join(high_dir, f)) for f in common]


def run_inference(model, low_dir, high_dir, out_dir):
    """在原始分辨率上推理并保存增强图，返回 (输出路径, 参考路径) 列表。"""
    os.makedirs(out_dir, exist_ok=True)
    pairs = list_test_pairs(low_dir, high_dir)
    saved = []
    with torch.no_grad():
        for low_path, high_path in pairs:
            img = Image.open(low_path).convert("RGB")
            w, h = img.size
            t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
            t = t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # 原始分辨率
            out = model(t)
            out = torch.clamp(out, 0.0, 1.0)
            out_np = out[0].permute(1, 2, 0).cpu().numpy()
            out_np = (out_np * 255.0).astype(np.uint8)
            name = os.path.basename(low_path)
            save_path = os.path.join(out_dir, name)
            Image.fromarray(out_np).save(save_path)
            saved.append((save_path, high_path))
    return saved


class FullReferenceAssessor:
    """与论文 Table II/III/IV 口径一致的评估器。"""

    def __init__(self, device):
        import pyiqa
        from pypiqe import piqe as _piqe
        self._piqe = _piqe
        self.psnr_metric = pyiqa.create_metric("psnr", device=device)
        self.ssim_metric = pyiqa.create_metric("ssim", device=device)
        # NIQE 需要离线模型，若不存在则跳过 NIQE
        niqe_path = "/home/user/.cache/torch/hub/pyiqa/niqe_modelparameters.mat"
        self.niqe_metric = None
        if os.path.exists(niqe_path):
            self.niqe_metric = pyiqa.create_metric("niqe", pretrained_model_path=niqe_path, device=device)
        self.device = device

    @staticmethod
    def _imread_bgr(path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        return img

    @staticmethod
    def _to_rgb(bgr):
        if bgr.ndim == 3 and bgr.shape[2] == 3:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)

    def brightness_metrics(self, path):
        img = self._imread_bgr(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        b = gray.astype(np.float32)
        max_v, min_v = np.max(b), np.min(b)
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
                    blk = gray[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
                    if blk.size:
                        blocks.append(np.mean(blk))
        uniformity = 1 - (np.std(blocks) / (np.mean(blocks) + 1e-10)) if blocks else 0.0
        try:
            from scipy import stats
            skew = float(stats.skew(b.flatten()))
            kurt = float(stats.kurtosis(b.flatten()))
        except Exception:
            skew = kurt = 0.0
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

    def psnr_ssim(self, dist_path, ref_path):
        d = torch.from_numpy(self._to_rgb(self._imread_bgr(dist_path))).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        r = torch.from_numpy(self._to_rgb(self._imread_bgr(ref_path))).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        if d.shape != r.shape:
            r = torch.nn.functional.interpolate(r, size=(d.shape[2], d.shape[3]), mode="bilinear", align_corners=False)
        d, r = d.to(self.device), r.to(self.device)
        with torch.no_grad():
            psnr = self.psnr_metric(d, r).item()
            ssim = self.ssim_metric(d, r).item()
        return psnr, ssim

    def piqe(self, path):
        rgb = self._to_rgb(self._imread_bgr(path))
        score, _, _, _ = self._piqe(rgb)
        return float(score)

    def niqe(self, path):
        if self.niqe_metric is None:
            return None
        rgb = self._to_rgb(self._imread_bgr(path))
        t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        t = t.to(self.device)
        with torch.no_grad():
            return self.niqe_metric(t).item()


def _tukey_fences(scores):
    q1, q3 = np.percentile(scores, 25), np.percentile(scores, 75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def mean_std(scores, remove_outliers):
    s = np.array(scores, dtype=float)
    if remove_outliers and len(s) >= 4:
        lo, hi = _tukey_fences(s)
        s = s[(s >= lo) & (s <= hi)]
    return float(np.mean(s)), float(np.std(s)) if len(s) else 0.0


def median_iqr(scores, remove_outliers):
    s = np.array(scores, dtype=float)
    if remove_outliers and len(s) >= 4:
        lo, hi = _tukey_fences(s)
        s = s[(s >= lo) & (s <= hi)]
    return float(np.median(s)), float(np.percentile(s, 25)), float(np.percentile(s, 75))


def measure_latency(model, size, iters):
    model = model.to(DEVICE)
    x = torch.randn(1, *size, device=DEVICE)
    with torch.no_grad():
        for _ in range(5):  # 预热
            _ = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    return float(np.mean(ts)), float(np.std(ts))


def evaluate_model(model_type, assessor):
    # 1) 推理 + 保存增强图（原始分辨率）
    out_dir = os.path.join(OUTPUT_ROOT, "enhanced", model_type)
    pairs = run_inference(load_model(model_type, MODEL_TYPES[model_type][1]),
                          TEST_LOW_DIR, TEST_HIGH_DIR, out_dir)

    # 2) 逐图计算指标
    per = {"psnr": [], "ssim": [], "piqe": [], "niqe": []}
    bkeys = ["mean_brightness", "std_brightness", "dynamic_range", "michelson_contrast",
             "over_exposed_ratio", "under_exposed_ratio", "histogram_entropy",
             "brightness_uniformity", "brightness_skewness", "brightness_kurtosis"]
    bright = {k: [] for k in bkeys}

    for dist_path, ref_path in pairs:
        psnr, ssim = assessor.psnr_ssim(dist_path, ref_path)
        per["psnr"].append(psnr)
        per["ssim"].append(ssim)
        per["piqe"].append(assessor.piqe(dist_path))
        niqe = assessor.niqe(dist_path)
        if niqe is not None:
            per["niqe"].append(niqe)
        bm = assessor.brightness_metrics(dist_path)
        for k in bkeys:
            bright[k].append(bm[k])

    # 3) 汇总（与论文 Table I 说明一致）
    result = {}
    for k in ("psnr", "ssim", "piqe", "niqe"):
        if per[k]:
            m, s = mean_std(per[k], REMOVE_OUTLIERS)
            result[k] = {"mean": m, "std": s, "n": len(per[k])}
    for k in ("mean_brightness", "std_brightness", "michelson_contrast",
              "histogram_entropy", "brightness_skewness", "brightness_kurtosis"):
        m, s = mean_std(bright[k], REMOVE_OUTLIERS)
        result[k] = {"mean": m, "std": s}
    for k in ("brightness_uniformity", "dynamic_range", "over_exposed_ratio", "under_exposed_ratio"):
        med, q1, q3 = median_iqr(bright[k], REMOVE_OUTLIERS)
        result[k] = {"median": med, "q1": q1, "q3": q3}

    # 4) 参数量与稳态耗时 / FPS
    model = load_model(model_type, MODEL_TYPES[model_type][1])
    result["params_m"] = count_params(model)
    lat_mean, lat_std = measure_latency(model, LATENCY_SIZE, LATENCY_ITER)
    result["inference_ms"] = {"mean": round(lat_mean, 4), "std": round(lat_std, 4)}
    result["inference_fps"] = round(1000.0 / lat_mean, 2) if lat_mean > 0 else None
    return result


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    assessor = FullReferenceAssessor(DEVICE)
    out = {"timestamp": datetime.now().isoformat(), "device": DEVICE,
           "remove_outliers": REMOVE_OUTLIERS, "models": {}}
    for model_type in MODEL_TYPES:
        print(f"\n===== 评估模型: {model_type} =====")
        out["models"][model_type] = evaluate_model(model_type, assessor)
        print(json.dumps(out["models"][model_type], ensure_ascii=False, indent=2))
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写出: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()