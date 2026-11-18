#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retinexformer 稳态时延 mean±std（400x600, 批大小1, 排除预热）。"""
import os, sys, time, statistics, torch
sys.path.insert(0, "/home/user/YouSihai/lightEnhancement/Retinexformer")
from basicsr.models import create_model
from basicsr.utils.options import parse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
opt = parse("/data/lowlight_paper_revision/retinex_eval15.yml", is_train=False)
opt["dist"] = False
model = create_model(opt).net_g
ck = torch.load("/home/user/YouSihai/lightEnhancement/Retinexformer/pretrained_weights/LOL_v1.pth", map_location="cpu")
try:
    model.load_state_dict(ck["params"])
except Exception:
    model.load_state_dict({"module." + k: v for k, v in ck["params"].items()})
model.cuda().eval()
x = torch.randn(1, 3, 400, 600).cuda()
with torch.no_grad():
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
m = statistics.mean(ts); s = statistics.stdev(ts)
print(f"LATENCY mean={m:.3f} ms std={s:.3f} ms fps={1000/m:.1f}", flush=True)