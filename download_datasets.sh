#!/usr/bin/env bash
# 跨数据集全参考评估 —— 数据集下载脚本（远程 /data/lowlight_paper_revision/data 下执行）
# LOL-v2-real : 从 hf-mirror 镜像（CHXD-Liquidity/lol-v2，含官方 2019 原始时间戳）下载完整 zip，
#               仅解压 Real_captured/Test（Low/Normal = 100 对）。
# MIT-FiveK    : 官方 CSAIL 服务器 data.csail.mit.edu/graphics/fivek 直接下载 DNG + Expert C TIFF16，
#               由 prepare_fivek.py 并行下载并转 sRGB。
set -e
DATA=/data/lowlight_paper_revision/data
mkdir -p "$DATA"

# ---------- LOL-v2-real ----------
if [ ! -d "$DATA/lolv2_real/high" ] || [ "$(ls "$DATA/lolv2_real/high" | wc -l)" -lt 100 ]; then
  cd "$DATA"
  [ -f lolv2.zip ] || curl -sS -L -C - --retry 3 -o lolv2.zip \
      'https://hf-mirror.com/datasets/CHXD-Liquidity/lol-v2/resolve/main/LOL-v2.zip'
  unzip -o -q lolv2.zip 'LOL-v2/Real_captured/Test/*' -d lolv2_real_extract
  SRC="$DATA/lolv2_real_extract/LOL-v2/Real_captured/Test"
  mkdir -p "$DATA/lolv2_real/low" "$DATA/lolv2_real/high"
  cp "$SRC"/Low/*.png    "$DATA/lolv2_real/low/"
  cp "$SRC"/Normal/*.png "$DATA/lolv2_real/high/"
  echo "LOL-v2-real: low=$(ls "$DATA/lolv2_real/low"|wc -l) high=$(ls "$DATA/lolv2_real/high"|wc -l)"
fi

# ---------- MIT-FiveK (前 100 张 a0001-a0100) ----------
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate py3_9
cd /data/lowlight_paper_revision
python prepare_fivek.py --n 100 --workers 10

echo "ALL DOWNLOAD DONE"