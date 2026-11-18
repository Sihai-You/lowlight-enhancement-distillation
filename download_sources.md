# 跨数据集下载来源说明（LOL-v2-real / MIT-FiveK）

## 1. LOL-v2-real（100 对测试）
- **来源**：HuggingFace 数据集 `CHXD-Liquidity/lol-v2` 的 `LOL-v2.zip`。
- **下载 URL**：`https://hf-mirror.com/datasets/CHXD-Liquidity/lol-v2/resolve/main/LOL-v2.zip`（hf-mirror 是国内可直连的 HuggingFace 镜像；远程对 huggingface.co 主站不可达）。
- **大小**：1,046,491,030 字节；解压后 3,593 文件。
- **结构**：`LOL-v2/Real_captured/Test/Low/low00690..low00789.png`（100 张输入）与
  `LOL-v2/Real_captured/Test/Normal/normal00690..normal00789.png`（100 张 GT）。
- **口径核对**：zip 内文件时间戳为 2019-11-27，与官方发布一致；Low 输入 100 张两两互异（MD5 全部不同），
  Normal(GT) 仅有 43 个唯一 MD5（存在多张 low 共享同一 normal 的现象，这是官方 LOL-v2-real 本身特性，
  与 HuggingFace 镜像 `okhater/lolv2-real`、`pkrushith/lolv2-real` 的 Test/GT 完全一致）。
  全部按官方 100 对（690–789）评估，与文献口径一致。
- 下载方式：curl + hf-mirror 直链（Google Drive 官方源在远程被墙，故用此镜像）。

## 2. MIT-Adobe FiveK（前 100 张，a0001–a0108）
- **来源**：MIT CSAIL 官方托管站点 `https://data.csail.mit.edu/graphics/fivek/`（远程 curl 返回 200）。
- **输入图定义**：DNG 原始图，rawpy `postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8,
  output_color=sRGB, gamma=(2.222,4.5))` 渲染成 sRGB 8-bit（即"相机原图/未精修"渲染，不自动提亮）。
- **GT 定义**：Expert C 精修图 TIFF16（ProPhoto RGB），用内嵌 ICC 经 Pillow ImageCms
  `RELATIVE_COLORIMETRIC` 转为 sRGB 8-bit。
- **文件 URL**：
  - 输入：`https://data.csail.mit.edu/graphics/fivek/img/dng/<name>.dng`（每个约 5 MB）
  - GT　：`https://data.csail.mit.edu/graphics/fivek/img/tiff16_c/<name>.tif`（每个约 31.6 MB）
- **子集划分**：官方页面按编号顺序取前 100 张；实际编号 a0001–a0108（个别编号缺失，
  如 a0021/a0023/a0033 等），清单以 `/data/lowlight_paper_revision/data/mit_fivek/manifest.json` 为准。
  注：并非 DPE 论文的"500 测试划分"（该划分需额外文件清单），本实验使用明确记录的 100 张子集。
- **尺寸注意**：DNG 渲染为 3040×2014，Expert C 为 3008×2000。评估时因 3MP 原分辨率推理在 V100-32GB 上
  触发 CUDA OOM（模型非为 3MP 设计），故输入（DNG 渲染图）按长边 1024 等比缩放后推理，
  GT 由 FullReferenceAssessor.psnr_ssim 双线性缩放对齐到输入分辨率。
- **许可**：仅限研究用途（官方站点 LicenseAdobe.txt / LicenseAdobeMIT.txt）。

## 3. 下载日志
- LOL-v2.zip：2026-10-31 远程完成下载，1.05 GB，解压无错误。
- MIT-FiveK：`prepare_fivek.py --n 100 --workers 10` 并行下载，日志
  `/data/lowlight_paper_revision/data/fivek_prep.log`。