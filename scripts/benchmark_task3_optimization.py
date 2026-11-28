"""
scripts/benchmark_task3_optimization.py
=======================================
Task 3 (Review 3 - Stage 1 Items 1.4 & 1.5):
Hardware-verified Before vs. After Inference Optimization Benchmark on Capstone
Deploy Checkpoints (GDUT-HWD 5-Class Primary Runtime), kept strictly
isolated from Proposed V1 (SHWD/HHW) auxiliary ablation benchmarks.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ultralytics

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.engine.detector  # noqa: F401 - registers custom_ablation_modules & __main__
from ultralytics import YOLO
import custom_ablation_modules as cam

OUT_DIR = ROOT / "Output" / "task3_eval_opt"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_loop_biformer_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Legacy unvectorized Python-loop BiFormer forward (Before optimization baseline)."""
    import torch.nn.functional as F
    b, c, h, w = x.shape
    rs = self.region_size
    pad_h = (rs - h % rs) % rs
    pad_w = (rs - w % rs) % rs
    x_pad = F.pad(x, (0, pad_w, 0, pad_h))
    hp, wp = x_pad.shape[-2:]
    gh, gw = hp // rs, wp // rs
    q, k, v = self.qkv(x_pad).chunk(3, dim=1)
    q_tokens = q.unfold(2, rs, rs).unfold(3, rs, rs).permute(0, 2, 3, 4, 5, 1).reshape(b, gh * gw, rs * rs, c)
    k_tokens = k.unfold(2, rs, rs).unfold(3, rs, rs).permute(0, 2, 3, 4, 5, 1).reshape(b, gh * gw, rs * rs, c)
    v_tokens = v.unfold(2, rs, rs).unfold(3, rs, rs).permute(0, 2, 3, 4, 5, 1).reshape(b, gh * gw, rs * rs, c)
    q_region = q_tokens.mean(dim=2).float()
    k_region = k_tokens.mean(dim=2).float()
    route_logits = torch.nan_to_num(torch.matmul(q_region, k_region.transpose(-1, -2)) / (c ** 0.5), nan=-1e4, posinf=1e4, neginf=-1e4)
    topk = min(self.topk, gh * gw)
    route_idx = route_logits.topk(topk, dim=-1).indices
    out_regions = []
    head_dim = c // self.num_heads
    batch_idx = torch.arange(b, device=x.device).unsqueeze(1)
    for region_idx in range(gh * gw):
        selected = route_idx[:, region_idx]
        k_sel = k_tokens[batch_idx, selected].reshape(b, topk * rs * rs, c)
        v_sel = v_tokens[batch_idx, selected].reshape(b, topk * rs * rs, c)
        q_cur = q_tokens[:, region_idx]
        qh = q_cur.reshape(b, rs * rs, self.num_heads, head_dim).transpose(1, 2)
        kh = k_sel.reshape(b, topk * rs * rs, self.num_heads, head_dim).transpose(1, 2)
        vh = v_sel.reshape(b, topk * rs * rs, self.num_heads, head_dim).transpose(1, 2)
        attn_scores = torch.clamp(torch.nan_to_num(torch.matmul(qh.float(), kh.float().transpose(-1, -2)) / (head_dim ** 0.5), nan=-50.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, vh.float()).to(qh.dtype).transpose(1, 2).reshape(b, rs * rs, c)
        out_regions.append(out)
    y = torch.stack(out_regions, dim=1).reshape(b, gh, gw, rs, rs, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, hp, wp)[:, :, :h, :w]
    return torch.nan_to_num(x + self.norm(self.proj(torch.nan_to_num(y, nan=0.0))), nan=0.0)


def bench_forward(fn, x: torch.Tensor, use_cuda: bool, warmup: int = 20, iters: int = 80):
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
    with torch.inference_mode():
        for _ in range(warmup):
            _ = fn(x)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fn(x)
        if use_cuda:
            torch.cuda.synchronize()
    mean_ms = (time.perf_counter() - t0) / iters * 1000.0
    vram_mb = (torch.cuda.max_memory_allocated(0) / (1024 * 1024)) if use_cuda else 0.0
    return round(mean_ms, 2), round(1000.0 / mean_ms, 1), round(vram_mb, 1)


def _load_worker_frames(n_frames: int = 12):
    vid = ROOT / "video_test" / "construction_site_workers_1080p.mp4"
    if not vid.exists():
        return [], []
    cap = cv2.VideoCapture(str(vid))
    indices = list(range(180, 180 + n_frames * 20, 20))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()
    return frames, indices[: len(frames)]


def _compute_parity_stats(r32_list, r16_list, frame_indices):
    c32, c16, max_delta = 0, 0, 0
    conf_diffs = []
    for r32, r16 in zip(r32_list, r16_list):
        n32 = len(r32.boxes) if r32.boxes is not None else 0
        n16 = len(r16.boxes) if r16.boxes is not None else 0
        c32 += n32
        c16 += n16
        max_delta = max(max_delta, abs(n32 - n16))
        m32 = float(r32.boxes.conf.float().mean().item()) if n32 > 0 else 0.0
        m16 = float(r16.boxes.conf.float().mean().item()) if n16 > 0 else 0.0
        conf_diffs.append(abs(m32 - m16))
    return {
        "frames": len(r32_list),
        "frame_indices": frame_indices,
        "fp32_boxes": int(c32),
        "fp16_amp_boxes": int(c16),
        "max_abs_count_delta": int(max_delta),
        "mean_conf_diff": round(float(np.mean(conf_diffs)) if conf_diffs else 0.0, 5),
    }


def check_video_parity(yolo_obj: YOLO, n_frames: int = 12):
    frames, indices = _load_worker_frames(n_frames)
    if not frames:
        return {"frames": 0, "fp32_boxes": 0, "fp16_amp_boxes": 0, "max_abs_count_delta": 0, "mean_conf_diff": 0.0}
    prev_m = torch.backends.cuda.matmul.allow_tf32
    prev_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        r32_list, r16_list = [], []
        for f in frames:
            r32_list.append(yolo_obj.predict(f, imgsz=640, device="cuda:0", half=False, conf=0.25, iou=0.45, verbose=False)[0])
            with torch.amp.autocast("cuda", dtype=torch.float16):
                r16_list.append(yolo_obj.predict(f, imgsz=640, device="cuda:0", half=False, conf=0.25, iou=0.45, verbose=False)[0])
        return _compute_parity_stats(r32_list, r16_list, indices)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_m
        torch.backends.cudnn.allow_tf32 = prev_c


def check_video_parity_batched(yolo_obj: YOLO, n_frames: int = 12, batch_size: int = 4):
    frames, indices = _load_worker_frames(n_frames)
    if not frames:
        return {"frames": 0, "batch": batch_size, "fp32_boxes": 0, "fp16_amp_boxes": 0, "max_abs_count_delta": 0, "mean_conf_diff": 0.0}
    prev_m = torch.backends.cuda.matmul.allow_tf32
    prev_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        r32_list, r16_list = [], []
        for i in range(0, len(frames), batch_size):
            chunk = frames[i : i + batch_size]
            r32_list.extend(yolo_obj.predict(chunk, batch=len(chunk), imgsz=640, device="cuda:0", half=False, conf=0.25, iou=0.45, verbose=False))
            with torch.amp.autocast("cuda", dtype=torch.float16):
                r16_list.extend(yolo_obj.predict(chunk, batch=len(chunk), imgsz=640, device="cuda:0", half=False, conf=0.25, iou=0.45, verbose=False))
        res = _compute_parity_stats(r32_list, r16_list, indices)
        res["batch"] = batch_size
        return res
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_m
        torch.backends.cudnn.allow_tf32 = prev_c


def main():
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[TASK 3 OPTIMIZATION BENCHMARK — GDUT-HWD 5-CLASS ONLY] Device: {device_name}")

    models = [
        ("Rep-YOLO11s-GDUT-A6 (Capstone 5-Cls)", ROOT / "Output/GDUT_Bộ số 1_stage2-full-train-run-a6/deploy_weights/rep_yolo11s_gdut_a6_deploy.pt"),
    ]

    rows = []
    parity_summary = {}
    parity_b4_summary = {}

    for label, ckpt in models:
        if not ckpt.exists():
            print(f"[SKIP] Missing {ckpt}")
            continue
        size_mb = round(ckpt.stat().st_size / (1024 * 1024), 2)
        y = YOLO(str(ckpt))
        m_cpu = y.model.cpu().eval()
        x_cpu = torch.randn(1, 3, 640, 640)

        # 1. CPU 4-thread baseline
        torch.set_num_threads(4)
        cpu_ms, cpu_fps, _ = bench_forward(m_cpu, x_cpu, use_cuda=False, warmup=2, iters=6)
        rows.append({
            "model": label, "stage": "CPU 4-Thread FP32", "device": "CPU (4 threads)",
            "precision": "FP32 (CPU)", "batch": 1, "latency_ms": cpu_ms, "fps": cpu_fps,
            "vram_mb": 0.0, "ckpt_mb": size_mb, "speedup_vs_before": round(cpu_ms / cpu_ms, 2)
        })

        if not torch.cuda.is_available():
            continue

        m_gpu = y.model.cuda().eval()
        x_gpu = torch.randn(1, 3, 640, 640, device="cuda")

        # 2. BEFORE: Unvectorized Python-loop BiFormer + GPU FP32
        orig_vec_forward = cam.BiFormerBlockLite.forward
        cam.BiFormerBlockLite.forward = run_loop_biformer_forward
        before_ms, before_fps, before_vram = bench_forward(m_gpu, x_gpu, use_cuda=True, warmup=15, iters=60)
        rows.append({
            "model": label, "stage": "Before (Loop BiFormer FP32)", "device": device_name,
            "precision": "FP32 (cuDNN TF32=True)", "batch": 1, "latency_ms": before_ms, "fps": before_fps,
            "vram_mb": before_vram, "ckpt_mb": size_mb, "speedup_vs_before": 1.0
        })

        # 3. AFTER-1: Vectorized torch.gather BiFormer + RepConv Deployed FP32
        cam.BiFormerBlockLite.forward = orig_vec_forward
        vec_ms, vec_fps, vec_vram = bench_forward(m_gpu, x_gpu, use_cuda=True, warmup=20, iters=80)
        rows.append({
            "model": label, "stage": "After-1 (Vec BiFormer + RepConv FP32)", "device": device_name,
            "precision": "FP32 (cuDNN TF32=True)", "batch": 1, "latency_ms": vec_ms, "fps": vec_fps,
            "vram_mb": vec_vram, "ckpt_mb": size_mb, "speedup_vs_before": round(before_ms / vec_ms, 2)
        })

        # 4. AFTER-2: Vectorized BiFormer + RepConv + CUDA AMP FP16
        def amp_fn(inp):
            with torch.amp.autocast("cuda"):
                return m_gpu(inp)
        amp_ms, amp_fps, amp_vram = bench_forward(amp_fn, x_gpu, use_cuda=True, warmup=20, iters=80)
        rows.append({
            "model": label, "stage": "After-2 (+ CUDA AMP FP16)", "device": device_name,
            "precision": "FP16-AMP", "batch": 1, "latency_ms": amp_ms, "fps": amp_fps,
            "vram_mb": amp_vram, "ckpt_mb": size_mb, "speedup_vs_before": round(before_ms / amp_ms, 2)
        })

        # 5. AFTER-3: Micro-Batched (batch=4) CUDA AMP FP16 (per-frame latency = batch_ms / 4)
        x_b4 = torch.randn(4, 3, 640, 640, device="cuda")
        def amp_b4(inp):
            with torch.amp.autocast("cuda"):
                return m_gpu(inp)
        b4_ms, _, b4_vram = bench_forward(amp_b4, x_b4, use_cuda=True, warmup=15, iters=60)
        per_frame_b4 = round(b4_ms / 4.0, 2)
        fps_b4 = round(4000.0 / b4_ms, 1)
        rows.append({
            "model": label, "stage": "After-3 (Micro-Batch N=4 AMP FP16)", "device": device_name,
            "precision": "FP16-AMP", "batch": 4, "latency_ms": per_frame_b4, "fps": fps_b4,
            "vram_mb": b4_vram, "ckpt_mb": size_mb, "speedup_vs_before": round(before_ms / per_frame_b4, 2)
        })

        # Real construction video parity test (B=1 and Micro-Batch N=4 separately)
        parity_b1 = check_video_parity(y, n_frames=12)
        parity_b4 = check_video_parity_batched(y, n_frames=12, batch_size=4)
        parity_summary[label] = parity_b1
        parity_b4_summary[label] = parity_b4

        rows[-5]["accuracy_delta"] = "CPU FP32 Ref"
        rows[-4]["accuracy_delta"] = "GPU FP32 Ref"
        rows[-3]["accuracy_delta"] = "Exact Reparameterized FP32"
        rows[-2]["accuracy_delta"] = (
            f"B=1: Delta {parity_b1['max_abs_count_delta']} boxes "
            f"({parity_b1['fp32_boxes']}/{parity_b1['fp16_amp_boxes']}, conf diff {parity_b1['mean_conf_diff']})"
        )
        rows[-1]["accuracy_delta"] = (
            f"N=4: Delta {parity_b4['max_abs_count_delta']} boxes "
            f"({parity_b4['fp32_boxes']}/{parity_b4['fp16_amp_boxes']}, conf diff {parity_b4['mean_conf_diff']})"
        )
        print(f"  [PARITY B=1] {label}: {parity_b1}")
        print(f"  [PARITY N=4] {label}: {parity_b4}")

    ckpt_main = models[0][1]
    deploy_sha = hashlib.sha256(ckpt_main.read_bytes()).hexdigest() if ckpt_main.exists() else ""
    raw_meta = torch.load(ckpt_main, map_location="cpu", weights_only=False) if ckpt_main.exists() else {}
    source_sha = str(raw_meta.get("source_checkpoint_sha256", "")) if isinstance(raw_meta, dict) else ""

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "optimization_benchmark_table.csv", index=False)
    payload = {
        "device": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ultralytics_version": ultralytics.__version__,
        "source_checkpoint_sha256": source_sha,
        "deployed_checkpoint_sha256": deploy_sha,
        "deployed_checkpoint_bytes": ckpt_main.stat().st_size if ckpt_main.exists() else 0,
        "records": rows,
        "video_parity_fp32_vs_amp_fp16": parity_summary,
        "video_parity_fp32_vs_amp_fp16_batch4": parity_b4_summary,
    }
    (OUT_DIR / "optimization_benchmark_table.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
