#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jetson 边缘设备验证基准脚本（供独立任务使用）。

在 Jetson（JetPack + PyTorch for Jetson）上加载学生模型，测量：
  - 稳态单图推理时延（mean±std, ms），batch=1，预热后
  - FPS
  - 峰值显存（MB）
  - 参数量（M）
并在 FP32 / FP16 两种精度下各测一遍，输出 JSON。

用法：
  python jetson_benchmark.py --ckpt /path/to/ablation_full.pth \
      --model-dir /path/to/dir_containing_models.py [--iters 100] [--size 600x400]

输出：
  results/jetson_benchmark.json    （同时打印到 stdout）
"""
import os
import sys
import json
import time
import argparse

import torch


def parse_size(s):
    """'WxH' -> (H, W)；缺省 '600x400'。"""
    w, h = (s.split("x") if "x" in s else [s, s])[:2]
    return int(h), int(w)


def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def bench(model, size, iters, fp16):
    inputs = torch.randn(1, 3, *size, device="cuda")
    model.eval()
    with torch.no_grad():
        for _ in range(10):          # 预热
            model(inputs)
        torch.cuda.synchronize()
        # 重置峰值显存（预热后）
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(inputs)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    lat = float(sum(ts) / len(ts))
    fps = 1000.0 / lat if lat > 0 else 0.0
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    return {
        "size": f"{size[1]}x{size[0]}",
        "precision": "fp16" if fp16 else "fp32",
        "latency_ms_mean": round(lat, 3),
        "latency_ms_std": round(float(__import__("statistics").pstdev(ts)), 3),
        "fps": round(fps, 1),
        "peak_mem_mb": round(peak_mb, 1),
        "iters": iters,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--size", default="600x400")
    ap.add_argument("--out", default="jetson_benchmark.json")
    args = ap.parse_args()

    sys.path.insert(0, args.model_dir)
    from models import AblationModelFactory

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 显式指定 base_channels=32，避免依赖 config.py（Jetson 上只需 models.py）
    model = AblationModelFactory.create_model("full", base_channels=32).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("student_state_dict", "model_state_dict", "state_dict"):
            if k in ckpt:
                ckpt = ckpt[k]
                break
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    size = parse_size(args.size)
    result = {
        "device": "jetson",
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_available": device == "cuda",
        "params_M": round(count_params(model), 4),
        "runs": [],
    }
    for fp16 in (False, True):
        try:
            if fp16:
                model = model.half()
            result["runs"].append(bench(model, size, args.iters, fp16))
            if fp16:
                model = model.float()
        except Exception as e:
            result["runs"].append({"precision": "fp16" if fp16 else "fp32", "error": str(e)})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()