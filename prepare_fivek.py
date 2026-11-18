#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MIT-Adobe FiveK 子集下载与预处理（并行下载版）：
  - 输入: 官方 CSAIL 托管的 DNG（原始图）经 rawpy 渲染为 sRGB 8bit（相机白平衡、不自动提亮）。
  - GT:   Expert C 精修 TIFF16（ProPhoto RGB）经内嵌 ICC 转为 sRGB 8bit。
默认取官方页面按 a%%%% 编号排序的前 100 张（a0001–a0100）。
用法（远程）：python prepare_fivek.py --n 100 --workers 8
"""
import os, sys, re, json, subprocess, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

BASE = "/data/lowlight_paper_revision/data/mit_fivek"
RAW_DIR = os.path.join(BASE, "raw_dng")
GT_DIR = os.path.join(BASE, "raw_tiff_c")
IN_DIR = os.path.join(BASE, "input_srgb")
GT_OUT_DIR = os.path.join(BASE, "gt_srgb")
MANIFEST = os.path.join(BASE, "manifest.json")
PAGE = "https://data.csail.mit.edu/graphics/fivek/"


def fetch_list():
    html = subprocess.check_output(
        ["curl", "-sS", "--max-time", "120", PAGE]).decode(errors="replace")
    names = re.findall(r'img/dng/(a\d{4}-[A-Za-z0-9_]+)\.dng', html)
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); ordered.append(n)
    return ordered


def download(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    r = subprocess.run(["curl", "-sS", "-L", "-C", "-", "--retry", "3",
                        "--max-time", "600", "-o", dst, url])
    return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0


def dng_to_srgb(dng_path):
    import rawpy
    raw = rawpy.imread(dng_path)
    return raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8,
                           output_color=rawpy.ColorSpace.sRGB, gamma=(2.222, 4.5))


def tif_to_srgb(tif_path):
    from PIL import Image, ImageCms
    from io import BytesIO
    img = Image.open(tif_path)
    icc = img.info.get("icc_profile")
    if icc is None:
        raise RuntimeError(f"no ICC: {tif_path}")
    src = ImageCms.getOpenProfile(BytesIO(icc))
    srgb = ImageCms.createProfile("sRGB")
    out = ImageCms.profileToProfile(img, src, srgb, outputMode="RGB",
                                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
    return np.asarray(out).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--conv-workers", type=int, default=8)
    args = ap.parse_args()

    for d in (RAW_DIR, GT_DIR, IN_DIR, GT_OUT_DIR):
        os.makedirs(d, exist_ok=True)

    names = fetch_list()[args.start:args.start + args.n]
    print(f"subset size: {len(names)}", flush=True)

    # ---- 并行下载 ----
    jobs = []
    for name in names:
        jobs.append((f"{PAGE}img/dng/{name}.dng", os.path.join(RAW_DIR, f"{name}.dng")))
        jobs.append((f"{PAGE}img/tiff16_c/{name}.tif", os.path.join(GT_DIR, f"{name}.tif")))

    todo = [(u, d) for (u, d) in jobs if not (os.path.exists(d) and os.path.getsize(d) > 0)]
    print(f"to download: {len(todo)} files", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download, u, d): (u, d) for (u, d) in todo}
        done = 0
        for fut in as_completed(futs):
            ok = fut.result()
            done += 1
            if done % 20 == 0 or not ok:
                print(f"  dl {done}/{len(todo)} (ok={ok})", flush=True)

    # ---- 转换（并行，跳过已完成） ----
    from PIL import Image
    manifest = {"source": PAGE, "n": len(names), "names": [],
                "note": "input=DNG(rawpy camera-WB sRGB8, gamma 2.222/4.5); gt=ExpertC TIFF16 ICC ProPhoto->sRGB8 relative colorimetric"}

    conv_jobs = []
    for name in names:
        idx = name[:5]
        in_out = os.path.join(IN_DIR, f"{idx}.png")
        gt_out = os.path.join(GT_OUT_DIR, f"{idx}.png")
        dng_file = os.path.join(RAW_DIR, f"{name}.dng")
        tif_file = os.path.join(GT_DIR, f"{name}.tif")
        if not (os.path.exists(in_out) and os.path.getsize(in_out) > 0):
            conv_jobs.append(("in", dng_file, in_out))
        if not (os.path.exists(gt_out) and os.path.getsize(gt_out) > 0):
            conv_jobs.append(("gt", tif_file, gt_out))

    print(f"to convert: {len(conv_jobs)} files", flush=True)

    def do_conv(job):
        kind, src, dst = job
        rgb = dng_to_srgb(src) if kind == "in" else tif_to_srgb(src)
        Image.fromarray(rgb).save(dst)
        return dst

    done = 0
    with ThreadPoolExecutor(max_workers=args.conv_workers) as ex:
        futs = {ex.submit(do_conv, j): j for j in conv_jobs}
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 20 == 0:
                print(f"  conv {done}/{len(conv_jobs)}", flush=True)

    manifest["names"] = sorted(names)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    n_in = len(os.listdir(IN_DIR)); n_gt = len(os.listdir(GT_OUT_DIR))
    print(f"DONE input_srgb={n_in} gt_srgb={n_gt} manifest={MANIFEST}", flush=True)


if __name__ == "__main__":
    main()