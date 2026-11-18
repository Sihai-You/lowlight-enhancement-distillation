#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1-3.4 复杂度对比脚本：参数、FLOPs/MACs、稳态推理时延、FPS、峰值显存。
输入分辨率：256x256（训练块尺寸，与论文时延口径一致）+ 400x600（LOL 原生图尺寸）。
模型：our student (full + 4 消融变体) 与 teacher DDNet。
输出 results/complexity_test.json。
"""
import os, sys, json, time
import numpy as np
import torch
from thop import profile

BASE = "/data/lowlight_paper_revision"
sys.path.insert(0, os.path.join(BASE, "code", "our_model"))
sys.path.insert(0, os.path.join(BASE, "code", "baselines", "DDNet"))

from models import AblationModelFactory
from model import Main

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENCY_ITERS = 50


class DDNetOut(torch.nn.Module):
    """只返回 teacher 的增强图 x_out，便于 thop 计数。"""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, xgl):
        return self.m(x, xgl)[2]


def params_m(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def measure_flops(model, inputs, wrapper=None):
    m = wrapper(model) if wrapper else model
    return profile(m, inputs=inputs, verbose=False)[0] / 1e9  # GMac


def measure_latency(model, inputs, iters=LATENCY_ITERS):
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            if isinstance(inputs, tuple):
                model(*inputs)
            else:
                model(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if isinstance(inputs, tuple):
                model(*inputs)
            else:
                model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    mean, std = float(np.mean(ts)), float(np.std(ts))
    return mean, std, 1000.0 / mean if mean > 0 else 0.0


results = {}
factory = AblationModelFactory()
our_types = [("full", "full"), ("without_attention", "without_attention"),
             ("without_adaptive_residual", "without_adaptive_residual"),
             ("without_skip", "without_skip"), ("reduced_channels", "reduced_channels")]

for label, mt in our_types:
    m = factory.create_model(mt).to(DEVICE).eval()
    e = {}
    e["params_M"] = round(params_m(m), 4)
    x = torch.randn(1, 3, 256, 256, device=DEVICE)
    e["gmacs_256"] = round(measure_flops(m, (x,)), 4)
    x2 = torch.randn(1, 3, 400, 600, device=DEVICE)
    e["gmacs_400x600"] = round(measure_flops(m, (x2,)), 4)
    mean, std, fps = measure_latency(m, x)
    e["latency_ms"] = round(mean, 3)
    e["latency_std_ms"] = round(std, 3)
    e["fps"] = round(fps, 1)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        m(x)
    e["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    results[label] = e
    del m

# ---- teacher DDNet ----
t = Main().to(DEVICE).eval()
x = torch.randn(1, 3, 256, 256, device=DEVICE)
xgl = torch.randn(1, 1, 256, 256, device=DEVICE)
x2 = torch.randn(1, 3, 400, 600, device=DEVICE)
xgl2 = torch.randn(1, 1, 400, 600, device=DEVICE)
te = {}
te["params_M"] = round(params_m(t), 4)
te["gmacs_256"] = round(measure_flops(t, (x, xgl), wrapper=DDNetOut), 4)
te["gmacs_400x600"] = round(measure_flops(t, (x2, xgl2), wrapper=DDNetOut), 4)
mean, std, fps = measure_latency(t, (x, xgl))
te["latency_ms"] = round(mean, 3)
te["latency_std_ms"] = round(std, 3)
te["fps"] = round(fps, 1)
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    t(x, xgl)
te["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
results["DDNet_teacher"] = te

os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
json.dump(results, open(os.path.join(BASE, "results", "complexity_test.json"), "w"), indent=2)

# ---- 可读输出 ----
print("=" * 90)
print(f"{'Model':<26}{'Params(M)':>10}{'GMAC@256':>11}{'GMAC@400x600':>14}{'Lat(ms)':>10}{'FPS':>9}{'PeakMB':>9}")
print("=" * 90)
for k, e in results.items():
    print(f"{k:<26}{e['params_M']:>10.4f}{e['gmacs_256']:>11.3f}{e['gmacs_400x600']:>14.3f}"
          f"{e['latency_ms']:>10.3f}{e['fps']:>9.1f}{e['peak_gpu_mb']:>9.1f}")

print("\nParams/Ideal-FLOPS ratio note added to results.")
print("DONE -> results/complexity_test.json")