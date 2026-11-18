#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把学生模型 full 导出为 ONNX，供 Jetson 上的 TensorRT engine 构建使用。

运行位置：远程 V100 服务器（已装 torch + 模型权重 + models.py）。
  conda activate py3_9 && python export_onnx.py [--size 600x400] [--opset 13]

导出产物：
  /data/lowlight_paper_revision/code/jetson/student_full.onnx

说明：
  - 模型 forward 只用标准算子（Conv/BN/ReLU/MaxPool/AdaptivePool/Linear/
    Upsample(bilinear)/cat/clamp/sigmoid），ONNX 与 TensorRT 均支持。
  - 默认固定输入形状 (1, 3, H, W)，H/W 必须能被 4 整除（两次 MaxPool）。
  - 默认 600x400 -> (H=400, W=600)，与 jetson_benchmark.py 的 LOL 原生分辨率对齐。
  - 若安装了 onnx / onnxruntime，会额外做一次数值一致性校验（可选，不影响导出）。
"""
import os, sys, argparse
import numpy as np
import torch

BASE = "/data/lowlight_paper_revision"
sys.path.insert(0, os.path.join(BASE, "code", "our_model"))

CKPT = "/home/user/YouSihai/lightEnhancement/newStudent/models/ablation_full.pth"
OUT = os.path.join(BASE, "code", "jetson", "student_full.onnx")


def parse_size(s):
    """'WxH' -> (H, W)；缺省 '600x400'。"""
    w, h = (s.split("x") if "x" in s else [s, s])[:2]
    return int(h), int(w)


def load_model():
    from models import AblationModelFactory
    m = AblationModelFactory.create_model("full", base_channels=32).to("cpu")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("student_state_dict", "model_state_dict", "state_dict"):
            if k in ckpt:
                ckpt = ckpt[k]
                break
    m.load_state_dict(ckpt, strict=False)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="600x400", help="WxH, H/W 须能被 4 整除")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    H, W = parse_size(args.size)
    assert H % 4 == 0 and W % 4 == 0, f"输入 H/W 必须能被 4 整除，得到 {(H, W)}"

    model = load_model()
    dummy = torch.randn(1, 3, H, W)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,  # 固定形状，便于 TensorRT 直接建 engine
    )
    print(f"[OK] ONNX 已导出: {args.out}  (shape=1x3x{H}x{W}, opset={args.opset})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] 参数量: {n_params/1e6:.4f} M")

    # ---- 可选：onnxruntime 数值一致性校验 ----
    try:
        import onnx
        onnx_model = onnx.load(args.out)
        onnx.checker.check_model(onnx_model)
        print("[OK] onnx.checker 校验通过")
    except Exception as e:
        print(f"[warn] 无 onnx 库或校验失败（不影响导出）: {e}")

    try:
        import onnxruntime as ort
        with torch.no_grad():
            ref = model(dummy).numpy()
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        got = sess.run(None, {"input": dummy.numpy()})[0]
        err = float(np.max(np.abs(ref - got)))
        print(f"[OK] onnxruntime 校验完成，最大绝对误差 = {err:.3e}")
    except Exception as e:
        print(f"[warn] 无 onnxruntime 库，跳过数值校验: {e}")


if __name__ == "__main__":
    main()