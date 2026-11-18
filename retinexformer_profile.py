#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retinexformer 效率画像：参数/FLOPs(GMac=thop_flops/2)/时延/FPS/峰值显存，
口径与复杂度表一致（thop.profile + /2）。
"""
import os
import sys
import time
import statistics
import torch

sys.path.insert(0, "/home/user/YouSihai/lightEnhancement/Retinexformer")
from basicsr.models import create_model  # noqa: E402
from basicsr.utils.options import parse  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

opt = parse("/data/lowlight_paper_revision/retinex_eval15.yml", is_train=False)
opt["dist"] = False
model = create_model(opt).net_g
ck = torch.load(
    "/home/user/YouSihai/lightEnhancement/Retinexformer/pretrained_weights/LOL_v1.pth",
    map_location="cpu")
try:
    model.load_state_dict(ck["params"])
except Exception:
    model.load_state_dict({"module." + k: v for k, v in ck["params"].items()})
model.cuda().eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params} = {n_params/1e6:.4f} M", flush=True)

from thop import profile  # noqa: E402

for size in [(3, 256, 256), (3, 400, 600)]:
    x = torch.randn(1, *size).cuda()
    fl, pa = profile(model, inputs=(x,), verbose=False)
    fl = float(fl)
    print(f"size {size}: thop_flops={fl:.4e} gmacs(fl/2)={fl/2/1e9:.4f} params={pa}", flush=True)

# 时延 + 峰值显存
x = torch.randn(1, 3, 400, 600).cuda()
with torch.no_grad():
    for _ in range(10):
        _ = model(x)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    ts = []
    N = 50
    for _ in range(N):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1000)
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

mean = statistics.mean(ts)
std = statistics.stdev(ts)
print(f"latency: mean {mean:.3f} ms, std {std:.3f} ms, fps {1000/mean:.1f}", flush=True)
print(f"peak_gpu_mb: {peak_mb:.1f}", flush=True)