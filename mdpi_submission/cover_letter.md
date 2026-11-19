# Cover Letter

**Date:** 19 November 2026

**To:** The Editorial Office, *Sensors* (MDPI)

---

Dear Editors,

We are pleased to submit our original research article entitled **"Research on a Real-Time Low-Light Image Enhancement Lightweight Network Based on Knowledge Distillation"** for consideration for publication in *Sensors*.

Low-light image enhancement (LLIE) is a fundamental task in computer vision, yet state-of-the-art deep models are often too large and too slow for deployment on resource-constrained mobile and edge platforms. In this work, we address this practical gap by proposing a knowledge-distillation framework that compresses a high-capacity teacher model (DDNet) into a lightweight encoder–decoder student network. Our main contributions are:

1. **A lightweight student architecture** that achieves a favorable quality–efficiency trade-off, using only 2.7% of the teacher's parameters (0.144 M vs. 5.39 M), 15.3× fewer FLOPs (7.22 vs. 110.51 GMac), and 16.2× faster inference (365 FPS on a datacenter GPU), while retaining comparable PSNR/SSIM and better perceptual quality (PIQE 22.39 vs. 26.81).

2. **A multi-level distillation loss** that jointly aligns outputs, deep features, local contrast, and global brightness, thereby transferring the teacher's enhancement capability to the student with negligible quality loss.

3. **Practical edge deployment validated on an NVIDIA Jetson Orin Nano Super**, achieving 15.02 FPS (FP32) and 21.73 FPS (FP16) with under 0.5 GB of peak memory, approaching the real-time video threshold on low-power hardware.

4. **Comprehensive experimental validation**, including statistical significance tests, cross-dataset generalization, and ablation studies, all conducted under a unified and fully reproducible evaluation protocol.

We believe this work is well suited to *Sensors* because it directly addresses real-time, on-device visual sensing—a core topic of the journal—by demonstrating how a lightweight neural network can deliver near-real-time low-light enhancement on low-power edge hardware typical of smart sensing and embedded vision systems.

This manuscript is original, has not been published previously, and is not under consideration for publication elsewhere. All authors have read and approved the submission and declare no conflicts of interest.

We thank you for your consideration and look forward to your response.

Sincerely,

**Sihai You** (Corresponding Author)
[Affiliation]
[Email]
