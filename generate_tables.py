#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
由 ``unified_evaluate.py`` 产出的 ``unified_metrics.json`` 重新生成论文表格，
修复论文中的两处问题：

1. Table V（消融实验）与 Table II 使用不同评估协议导致的 PSNR/SSIM/亮度矛盾；
2. Table VI 中 "中" 未翻译为英文（"Medium"），以及引用编号错误（见 paper_revision.md）。

输出均为可直接粘贴的 Markdown 与 LaTeX。
"""

import json
import os
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(HERE, "unified_metrics.json")

# 论文方法名（与 Table V 一致）
NAME_CN_EN = {
    "full": "Full Optimized Model",
    "reduced_channels": "Channels Halved",
    "without_attention": "No Attention Mech.",
    "without_adaptive_residual": "No Adaptive Residual",
    "without_skip": "No Skip Connections",
}


def load(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_ms(d):
    return f"{d['mean']:.2f}±{d['std']:.2f}" if d else "N/A"


def gen_ablation_markdown(data):
    models = data["models"]
    lines = [
        "| Model | PSNR (dB) | SSIM | Avg. Luminance | Michelson Contrast | Params (M) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key in ("full", "reduced_channels", "without_attention",
                "without_adaptive_residual", "without_skip"):
        m = models[key]
        psnr = f"{m['psnr']['mean']:.2f}±{m['psnr']['std']:.2f}"
        ssim = f"{m['ssim']['mean']:.4f}"
        lum = f"{m['mean_brightness']['mean']:.1f}±{m['mean_brightness']['std']:.1f}"
        con = f"{m['michelson_contrast']['mean']:.4f}"
        par = f"{m['params_m']:.2f}"
        lines.append(f"| {NAME_CN_EN[key]} | {psnr} | {ssim} | {lum} | {con} | {par} |")
    return "\n".join(lines)


def gen_ablation_latex(data):
    models = data["models"]
    rows = []
    for key in ("full", "reduced_channels", "without_attention",
                "without_adaptive_residual", "without_skip"):
        m = models[key]
        rows.append(
            f"    {NAME_CN_EN[key]} & "
            f"{m['psnr']['mean']:.2f}$\\pm${m['psnr']['std']:.2f} & "
            f"{m['ssim']['mean']:.4f} & "
            f"{m['mean_brightness']['mean']:.1f}$\\pm${m['mean_brightness']['std']:.1f} & "
            f"{m['michelson_contrast']['mean']:.4f} & {m['params_m']:.2f} \\\\"
        )
    return (
        "\\begin{table}[t]\n"
        "\\caption{Comparison of Ablation Study Results (unified protocol)}\n"
        "\\label{tab:ablation}\n"
        "\\centering\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        " & PSNR (dB) & SSIM & Avg. Luminance & Contrast & Params (M) \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


def gen_component_analysis(data):
    """生成修正后的 Table VI（Importance 列翻译为英文）。"""
    models = data["models"]
    base = models["full"]
    out = [
        "| Component | PSNR Change (dB) | SSIM Change | Params Change (M) | Importance |",
        "| --- | --- | --- | --- | --- |",
    ]
    rows = [
        ("Attention Mechanism", "without_attention", "High", "Significantly affects reconstruction accuracy"),
        ("Skip Connections", "without_skip", "Medium", "Improves PSNR but reduces SSIM"),
        ("Adaptive Residual", "without_adaptive_residual", "Medium", "Moderate impact on PSNR"),
        ("Channels (32→16)", "reduced_channels", "Medium", "Extremely high parameter efficiency"),
    ]
    for name, key, imp, note in rows:
        m = models[key]
        dpsnr = m["psnr"]["mean"] - base["psnr"]["mean"]
        dssim = m["ssim"]["mean"] - base["ssim"]["mean"]
        dpar = m["params_m"] - base["params_m"]
        out.append(f"| {name} | {dpsnr:+.2f} | {dssim:+.4f} | {dpar:+.2f} | {imp} |")
    return "\n".join(out)


def gen_timing(data):
    lines = ["| Model | Inference (ms) | Inference FPS | Params (M) |",
             "| --- | --- | --- | --- |"]
    for key in ("full", "reduced_channels", "without_attention",
                "without_adaptive_residual", "without_skip"):
        m = data["models"][key]
        lines.append(
            f"| {NAME_CN_EN[key]} | {fmt_ms(m['inference_ms'])} | "
            f"{m['inference_fps']} | {m['params_m']:.3f} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=DEFAULT_JSON)
    args = ap.parse_args()
    if not os.path.exists(args.json):
        raise SystemExit(f"未找到 {args.json}，请先运行 unified_evaluate.py")
    data = load(args.json)
    print("### 修正后的 Table V（消融实验，统一评估协议）")
    print(gen_ablation_markdown(data))
    print("\n### LaTeX（Table V）\n")
    print(gen_ablation_latex(data))
    print("\n### 修正后的 Table VI（组件影响分析，Importance 列翻译为英文）")
    print(gen_component_analysis(data))
    print("\n### 模型效率 / FPS 汇总（用于修正 261.27 FPS 与 2.94 ms 的矛盾）")
    print(gen_timing(data))


if __name__ == "__main__":
    main()