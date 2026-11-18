#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 Jetson 上对 TensorRT engine 做稳态单图推理基准（batch=1）。

针对 JetPack 5.x / 6.x（TensorRT 8.x，经典 binding API）编写。

用法：
  python bench_trt.py --engine student_full_fp16.engine --iters 200 --precision fp16

输出 JSON（结构与 jetson_benchmark.py 一致），同时打印到 stdout：
  latency_ms_mean / latency_ms_std / fps / peak_mem_mb（近似）/ iters / precision
"""
import os
import json
import argparse
import statistics

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


def get_engine_input_output(engine, context):
    """返回 (input_name, output_name, in_shape, out_shape)。"""
    n = engine.num_bindings
    in_name = out_name = None
    in_shape = out_shape = None
    for i in range(n):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        if engine.binding_is_input(i):
            in_name, in_shape = name, tuple(shape)
        else:
            out_name, out_shape = name, tuple(shape)
    if in_name is None or out_name is None:
        raise RuntimeError("未找到 engine 的输入/输出 binding")
    return in_name, out_name, in_shape, out_shape


def bench(engine_path, iters, warmup=20):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    in_name, out_name, in_shape, out_shape = get_engine_input_output(engine, context)
    print(f"[info] engine={os.path.basename(engine_path)} in={in_name}{in_shape} out={out_name}{out_shape}")

    dtype = np.float32 if engine.get_binding_dtype(engine.get_binding_index(in_name)) == trt.float32 else None
    if dtype is None:
        raise RuntimeError("当前脚本假定输入为 float32")

    n_in = int(np.prod(in_shape))
    n_out = int(np.prod(out_shape))

    d_input = cuda.mem_alloc(n_in * dtype().nbytes)
    d_output = cuda.mem_alloc(n_out * dtype().nbytes)

    h_input = np.full(in_shape, 0.5, dtype=dtype)  # 中性灰度输入（模型期望 [0,1]）
    h_output = np.empty(out_shape, dtype=dtype)
    cuda.memcpy_htod(d_input, np.ascontiguousarray(h_input))

    in_idx = engine.get_binding_index(in_name)
    out_idx = engine.get_binding_index(out_name)
    bindings = [0] * engine.num_bindings
    bindings[in_idx] = int(d_input)
    bindings[out_idx] = int(d_output)

    stream = cuda.Stream()
    start = cuda.Event()
    end = cuda.Event()

    # 预热
    for _ in range(warmup):
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()

    free0 = cuda.mem_get_info()[0]
    peak_alloc_mb = 0.0
    ts = []
    for _ in range(iters):
        start.record(stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        end.record(stream)
        stream.synchronize()
        ts.append(start.time_till(end))  # ms

    # 峰值显存近似：运行前后空闲显存差 + engine/激活占用
    free1 = cuda.mem_get_info()[0]
    used_mb = (free0 - free1) / 1e6
    peak_alloc_mb = max(0.0, used_mb)

    cuda.memcpy_dtoh(h_output, d_output)

    lat = float(np.mean(ts))
    return {
        "precision": None,  # 由调用方填入
        "latency_ms_mean": round(lat, 3),
        "latency_ms_std": round(float(statistics.pstdev(ts)), 3),
        "fps": round(1000.0 / lat, 1) if lat > 0 else 0.0,
        "peak_mem_mb": round(peak_alloc_mb, 1),
        "iters": iters,
        "input_shape": list(in_shape),
        "output_shape": list(out_shape),
        "output_sample_mean": round(float(np.mean(h_output)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--out", default="trt_benchmark.json")
    args = ap.parse_args()

    result = bench(args.engine, args.iters)
    result["precision"] = args.precision
    result["engine"] = args.engine

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n[OK] 已写出: {args.out}")


if __name__ == "__main__":
    main()