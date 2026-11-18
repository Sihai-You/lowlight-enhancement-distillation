#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 Jetson 上用 TensorRT Python API 从 ONNX 构建 engine（FP16 主 + INT8 可选）。

针对 JetPack 5.x / 6.x（TensorRT 8.x）编写。

用法：
  # 仅 FP16（推荐，稳健）
  python build_engine.py --onnx student_full.onnx --fp16 --out student_full_fp16.engine

  # FP16 + INT8（INT8 需要标定数据 --calib-dir 指向一批低光输入图；缺省用合成数据）
  python build_engine.py --onnx student_full.onnx --fp16 --int8 \
      --calib-dir /path/to/lowlight_calib_images --out-stem student_full

说明：
  - 输入 ONNX 来自 export_onnx.py，固定形状 (1,3,H,W)（默认 600x400 -> H=400,W=600）。
  - INT8 采用熵标定（IInt8EntropyCalibrator2），标定图取目录下前 N 张，归一化到 [0,1]。
"""
import os
import sys
import glob
import argparse

import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

CALIB_BATCH = 8
CALIB_MAX = 64


class ImageEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """用低光图片目录做 INT8 熵标定。"""

    def __init__(self, calib_dir, shape, cache_file="int8_calib.cache"):
        super().__init__()
        self.shape = shape  # (1, 3, H, W)
        self.cache_file = cache_file
        self.files = sorted(
            f for f in glob.glob(os.path.join(calib_dir, "*"))
            if f.lower().endswith(EXTS)
        )[:CALIB_MAX]
        self.batch = CALIB_BATCH
        n = self.batch * trt.volume(shape)
        self.device_input = None
        self.cuda = None
        self.n_elems = n

    def get_algorithm(self):
        return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2

    def get_batch_size(self):
        return self.batch

    def get_batch(self, names):
        # 按批读取并预处理图片 -> NCHW float32 [0,1]
        import cv2
        batch_imgs = []
        for p in self.files[self.cur: self.cur + self.batch]:
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # resize 到标定 shape（若目录图尺寸不一致）
            H, W = self.shape[2], self.shape[3]
            if rgb.shape[:2] != (H, W):
                rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            batch_imgs.append(rgb.transpose(2, 0, 1))
        if not batch_imgs:
            return None
        data = np.stack(batch_imgs, 0).reshape(1, -1)
        self.cur += self.batch
        if self.device_input is None or self.device_input.nbytes < data.nbytes:
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa
            self.cuda = cuda
            self.device_input = cuda.mem_alloc(data.nbytes)
        self.cuda.memcpy_htod(self.device_input, np.ascontiguousarray(data))
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_engine(onnx_path, fp16, int8, calib_dir, shape):
    import pycuda.autoinit  # noqa: F401（初始化 CUDA 上下文）
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("[ERR]", parser.get_error(i))
            raise RuntimeError("ONNX 解析失败")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[info] 启用 FP16")
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if calib_dir and os.path.isdir(calib_dir):
            calibrator = ImageEntropyCalibrator(calib_dir, shape)
            config.int8_calibrator = calibrator
            print(f"[info] 启用 INT8（熵标定，目录={calib_dir}）")
        else:
            print("[warn] 未提供 --calib-dir，仍启用 INT8 但由 TRT 使用默认标定（效果可能不稳）")
    elif int8:
        print("[warn] 设备不支持 INT8，跳过")

    print("[info] 开始构建 engine（可能需要几分钟）...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("engine 构建失败（返回 None）")
    return engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--out", default=None, help="输出 engine 路径（缺省按精度命名）")
    args = ap.parse_args()

    # 读取 ONNX 输入 shape
    import onnx
    m = onnx.load(args.onnx)
    shape = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    shape = shape if shape and shape[0] > 0 else [1, 3, 400, 600]
    print(f"[info] ONNX 输入 shape = {shape}")

    engine = build_engine(args.onnx, args.fp16, args.int8, args.calib_dir, shape)

    out = args.out
    if not out:
        tag = ""
        if args.fp16:
            tag += "_fp16"
        if args.int8:
            tag += "_int8"
        out = os.path.splitext(args.onnx)[0] + tag + ".engine"
    with open(out, "wb") as f:
        f.write(engine)
    print(f"[OK] engine 已写出: {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()