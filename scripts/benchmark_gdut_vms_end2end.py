"""
RTX 3050 End-to-End GDUT-A6 VMS + Non-Biometric Re-ID Benchmark (Task 3 Review 3 v2.2)
======================================================================================
Measures real GPU latency, throughput, peak VRAM, and precision parity across 12 real
construction-worker frames (frames 180..420, stride 20) using clean isolated subprocesses
per TF32/precision setting (preventing cuDNN handle caching cross-contamination):
  1. Pure GDUT-A6 Detector (Strict FP32: matmul_tf32=False, cudnn_tf32=False)
  2. Pure GDUT-A6 Detector (TF32 FP32 Default: matmul_tf32=True, cudnn_tf32=True, B=1)
  3. Pure GDUT-A6 Detector (Selective AMP FP16, B=1)
  4. GDUT-A6 TF32 + Head ByteTrack + 3-Strip Non-Biometric Upper-Torso Re-ID (use_deep=False)
  5. Full VMS + Multi-Layer MobileNetV3 Non-Biometric Upper-Torso Re-ID (use_deep=True, refresh=30)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_worker_frames(video_path: Path, frame_indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: list[np.ndarray] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.release()
    if len(frames) < len(frame_indices):
        raise RuntimeError(f"Expected {len(frame_indices)} frames, loaded {len(frames)}")
    return frames


def run_worker_tier(mode: str, warmup: int, timed: int) -> dict:
    import os

    use_tf32 = mode in ("tf32_fp32", "e2e_strip", "e2e_deep", "e2e_paired")
    use_sel_amp = mode == "sel_amp"
    os.environ["ALLOW_TF32"] = "True" if use_tf32 else "False"
    os.environ["USE_SELECTIVE_AMP"] = "True" if use_sel_amp else "False"

    torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
    torch.backends.cudnn.allow_tf32 = bool(use_tf32)

    from src.config import DEFAULT_MODEL_PATH, DEFAULT_VIDEO_PATH
    from src.engine.bytetrack import BYTETracker
    from src.engine.detector import SafetyObjectDetector
    from src.engine.gdut_reid import UpperTorsoReIDGallery

    frame_indices = list(range(180, 420, 20))
    frames = load_worker_frames(DEFAULT_VIDEO_PATH, frame_indices)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    is_cuda = device.startswith("cuda")

    detector = SafetyObjectDetector(
        model_path=str(DEFAULT_MODEL_PATH),
        device=device,
        use_half=False,
        use_selective_amp=use_sel_amp,
    )
    detector.use_selective_amp = use_sel_amp
    torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
    torch.backends.cudnn.allow_tf32 = bool(use_tf32)

    def _sync():
        if is_cuda:
            torch.cuda.synchronize()

    if mode in ("strict_fp32", "tf32_fp32", "sel_amp"):
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
        for i in range(warmup):
            detector.predict(frames[i % len(frames)])
        _sync()

        parity_counts: list[int] = []
        parity_confs: list[float] = []
        for fr in frames:
            dets, _, _ = detector.predict(fr)
            parity_counts.append(len(dets))
            confs = sorted([float(d.get("conf", 0.0)) for d in dets], reverse=True)
            parity_confs.append(float(np.mean(confs)) if confs else 0.0)
        _sync()

        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
        lats = []
        for i in range(timed):
            fr = frames[i % len(frames)]
            _sync()
            t0 = time.perf_counter()
            detector.predict(fr)
            _sync()
            lats.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(np.mean(lats))
        return {
            "mode": mode,
            "gpu_model": torch.cuda.get_device_name(0) if is_cuda else "CPU",
            "torch_version": torch.__version__,
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if is_cuda else False,
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32) if is_cuda else False,
            "frame_resolution": f"{frames[0].shape[1]}x{frames[0].shape[0]}",
            "lat_mean_ms": round(mean_ms, 2),
            "lat_p95_ms": round(float(np.percentile(lats, 95)), 2),
            "lat_p99_ms": round(float(np.percentile(lats, 99)), 2),
            "fps": round(1000.0 / max(1e-6, mean_ms), 1),
            "vram_mb": round(float(torch.cuda.max_memory_allocated() / (1024 * 1024)), 1) if is_cuda else 0.0,
            "parity_counts": parity_counts,
            "parity_confs": parity_confs,
        }

    def _run_reid_step(fr: np.ndarray, heads, trk: BYTETracker, gal: UpperTorsoReIDGallery, fidx: int, ts: float):
        tracks = trk.update(heads, frame=None)
        items = [(t.track_id, t.tlbr, str(detector.class_names.get(int(t.cls_id), "none")).lower().strip()) for t in tracks]
        profs = gal.associate_batch(items, frame_bgr=fr, frame_idx=fidx, now_ts=ts)
        gal.update_frame_end([p.worker_id for p in profs if p.worker_id > 0], frame_idx=fidx, now_ts=ts)

    def _run_e2e_step(fr: np.ndarray, trk: BYTETracker, gal: UpperTorsoReIDGallery, fidx: int, ts: float):
        detector.predict(fr)
        heads = detector.get_head_tracks()
        _run_reid_step(fr, heads, trk, gal, fidx, ts)

    if mode == "e2e_paired":
        # Continuous 1080p stream frames (stride=1) so ByteTrack confirms identities &
        # MobileNetV3 deep_refresh_interval=30 triggers authentically on real detections.
        stream_indices = list(range(180, 180 + warmup + timed))
        stream_frames = load_worker_frames(DEFAULT_VIDEO_PATH, stream_indices)

        # Within-frame paired shared-detection AB alternating benchmark (n=timed frames):
        # Measures shared YOLOv11s detection ONCE per frame, then times both Re-ID pipelines on
        # the exact same detections in alternating order (strip->deep on even, deep->strip on odd).
        tracker_strip = BYTETracker(
            track_thresh=0.30, track_low_thresh=0.15, new_track_thresh=0.30,
            track_buffer=60, match_thresh=0.70, grace_period=2,
        )
        gallery_strip = UpperTorsoReIDGallery(
            ttl_seconds=600.0, sim_thresh=0.70, use_deep=False,
            deep_refresh_interval=30, min_confirm_hits=6, min_confirm_frames=24,
            device="cpu",
        )
        ts_base = time.time()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
        for i in range(warmup):
            _run_e2e_step(stream_frames[i], tracker_strip, gallery_strip, i, ts_base + i * 0.04)
        _sync()
        vram_strip_mb = round(float(torch.cuda.max_memory_allocated() / (1024 * 1024)), 1) if is_cuda else 0.0

        tracker_deep = BYTETracker(
            track_thresh=0.30, track_low_thresh=0.15, new_track_thresh=0.30,
            track_buffer=60, match_thresh=0.70, grace_period=2,
        )
        gallery_deep = UpperTorsoReIDGallery(
            ttl_seconds=600.0, sim_thresh=0.70, use_deep=True,
            deep_refresh_interval=30, min_confirm_hits=6, min_confirm_frames=24,
            device=detector.device,
        )
        gallery_deep.prewarm()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
        for i in range(warmup):
            _run_e2e_step(stream_frames[i], tracker_deep, gallery_deep, i, ts_base + i * 0.04)
        _sync()

        det_lats: list[float] = []
        strip_reid_lats: list[float] = []
        deep_reid_lats: list[float] = []
        strip_lats: list[float] = []
        deep_lats: list[float] = []
        deep_extract_lats: list[float] = []
        deep_non_extract_lats: list[float] = []
        calls_start = int(gallery_deep.deep_extractor.extract_calls)

        for k in range(timed):
            fidx = warmup + k
            fr = stream_frames[fidx]
            ts = ts_base + fidx * 0.04

            _sync()
            t0 = time.perf_counter()
            detector.predict(fr)
            heads = detector.get_head_tracks()
            _sync()
            det_ms = (time.perf_counter() - t0) * 1000.0
            det_lats.append(det_ms)

            strip_reid_ms = 0.0
            deep_reid_ms = 0.0
            deep_did_extract = False
            order = ("strip", "deep") if (k % 2 == 0) else ("deep", "strip")
            for branch in order:
                if branch == "strip":
                    _sync()
                    t_s = time.perf_counter()
                    _run_reid_step(fr, heads, tracker_strip, gallery_strip, fidx, ts)
                    _sync()
                    strip_reid_ms = (time.perf_counter() - t_s) * 1000.0
                else:
                    c0 = int(gallery_deep.deep_extractor.extract_calls)
                    _sync()
                    t_d = time.perf_counter()
                    _run_reid_step(fr, heads, tracker_deep, gallery_deep, fidx, ts)
                    _sync()
                    deep_reid_ms = (time.perf_counter() - t_d) * 1000.0
                    deep_did_extract = int(gallery_deep.deep_extractor.extract_calls) > c0

            strip_total = det_ms + strip_reid_ms
            deep_total = det_ms + deep_reid_ms
            strip_reid_lats.append(strip_reid_ms)
            deep_reid_lats.append(deep_reid_ms)
            strip_lats.append(strip_total)
            deep_lats.append(deep_total)
            if deep_did_extract:
                deep_extract_lats.append(deep_total)
            else:
                deep_non_extract_lats.append(deep_total)

        vram_deep_mb = round(float(torch.cuda.max_memory_allocated() / (1024 * 1024)), 1) if is_cuda else 0.0
        deep_extract_calls = int(gallery_deep.deep_extractor.extract_calls) - calls_start
        det_mean = float(np.mean(det_lats))
        strip_reid_mean = float(np.mean(strip_reid_lats))
        deep_reid_mean = float(np.mean(deep_reid_lats))
        strip_mean = float(np.mean(strip_lats))
        deep_mean = float(np.mean(deep_lats))

        return {
            "mode": "e2e_paired",
            "protocol": f"Within-frame paired shared-detection (AB alternating, n={timed})",
            "strip": {
                "mode": "e2e_strip",
                "timed_frames": len(strip_lats),
                "det_mean_ms": round(det_mean, 2),
                "reid_mean_ms": round(strip_reid_mean, 2),
                "lat_mean_ms": round(strip_mean, 2),
                "lat_p95_ms": round(float(np.percentile(strip_lats, 95)), 2),
                "lat_p99_ms": round(float(np.percentile(strip_lats, 99)), 2),
                "fps": round(1000.0 / max(1e-6, strip_mean), 1),
                "vram_mb": vram_strip_mb,
                "deep_active": False,
            },
            "deep": {
                "mode": "e2e_deep",
                "timed_frames": len(deep_lats),
                "det_mean_ms": round(det_mean, 2),
                "reid_mean_ms": round(deep_reid_mean, 2),
                "lat_mean_ms": round(deep_mean, 2),
                "lat_p95_ms": round(float(np.percentile(deep_lats, 95)), 2),
                "lat_p99_ms": round(float(np.percentile(deep_lats, 99)), 2),
                "fps": round(1000.0 / max(1e-6, deep_mean), 1),
                "vram_mb": vram_deep_mb,
                "deep_active": bool(gallery_deep.deep_extractor.available),
                "deep_extract_calls": deep_extract_calls,
                "deep_frame_mean_ms": round(float(np.mean(deep_extract_lats)), 2) if deep_extract_lats else 0.0,
                "deep_frame_p95_ms": round(float(np.percentile(deep_extract_lats, 95)), 2) if deep_extract_lats else 0.0,
                "non_deep_frame_mean_ms": round(float(np.mean(deep_non_extract_lats)), 2) if deep_non_extract_lats else 0.0,
                "paired_delta_mean_ms": round(deep_mean - strip_mean, 2),
            },
        }

    # Tracking + Non-Biometric Upper-Torso Re-ID tiers
    use_deep = mode == "e2e_deep"
    tracker = BYTETracker(
        track_thresh=0.30, track_low_thresh=0.15, new_track_thresh=0.30,
        track_buffer=60, match_thresh=0.70, grace_period=2,
    )
    gallery = UpperTorsoReIDGallery(
        ttl_seconds=600.0, sim_thresh=0.70, use_deep=use_deep,
        deep_refresh_interval=30, min_confirm_hits=6, min_confirm_frames=24,
        device=detector.device if use_deep else "cpu",
    )
    if use_deep:
        gallery.prewarm()

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    ts_base = time.time()
    for i in range(warmup):
        fr = frames[i % len(frames)]
        detector.predict(fr)
        heads = detector.get_head_tracks()
        tracks = tracker.update(heads, frame=None)
        items = [(t.track_id, t.tlbr, str(detector.class_names.get(int(t.cls_id), "none")).lower().strip()) for t in tracks]
        profs = gallery.associate_batch(items, frame_bgr=fr, frame_idx=i, now_ts=ts_base + i * 0.04)
        gallery.update_frame_end([p.worker_id for p in profs if p.worker_id > 0], frame_idx=i, now_ts=ts_base + i * 0.04)
        if (i + 1) % len(frames) == 0:
            tracker.reset()
            gallery.on_video_loop()
    _sync()

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    lats = []
    for i in range(timed):
        fr = frames[i % len(frames)]
        _sync()
        t0 = time.perf_counter()
        detector.predict(fr)
        heads = detector.get_head_tracks()
        tracks = tracker.update(heads, frame=None)
        items = [(t.track_id, t.tlbr, str(detector.class_names.get(int(t.cls_id), "none")).lower().strip()) for t in tracks]
        profs = gallery.associate_batch(items, frame_bgr=fr, frame_idx=warmup + i, now_ts=ts_base + 10.0 + i * 0.04)
        gallery.update_frame_end([p.worker_id for p in profs if p.worker_id > 0], frame_idx=warmup + i, now_ts=ts_base + 10.0 + i * 0.04)
        _sync()
        lats.append((time.perf_counter() - t0) * 1000.0)
        if (i + 1) % len(frames) == 0:
            tracker.reset()
            gallery.on_video_loop()

    mean_ms = float(np.mean(lats))
    return {
        "mode": mode,
        "lat_mean_ms": round(mean_ms, 2),
        "lat_p95_ms": round(float(np.percentile(lats, 95)), 2),
        "lat_p99_ms": round(float(np.percentile(lats, 99)), 2),
        "fps": round(1000.0 / max(1e-6, mean_ms), 1),
        "vram_mb": round(float(torch.cuda.max_memory_allocated() / (1024 * 1024)), 1) if is_cuda else 0.0,
        "deep_active": bool(gallery.deep_extractor.available),
    }


def _invoke_subprocess_tier(mode: str, warmup: int, timed: int) -> dict:
    last_err = ""
    for attempt in range(2):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker-mode", mode, "--warmup", str(warmup), "--timed", str(timed)],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            for line in reversed(proc.stdout.splitlines()):
                if line.startswith("JSON_RESULT="):
                    return json.loads(line[len("JSON_RESULT="):])
        last_err = f"rc={proc.returncode} stdout={proc.stdout[-400:]} stderr={proc.stderr[-600:]}"
        time.sleep(1.5)
    raise RuntimeError(f"Worker {mode} failed: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-mode", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timed", type=int, default=60)
    args = parser.parse_args()

    if args.worker_mode:
        res = run_worker_tier(args.worker_mode, args.warmup, args.timed)
        print("JSON_RESULT=" + json.dumps(res), flush=True)
        return

    t1 = _invoke_subprocess_tier("strict_fp32", warmup=20, timed=60)
    print(f"[TIER 1 Strict FP32] {t1}")
    time.sleep(1.0)
    t2 = _invoke_subprocess_tier("tf32_fp32", warmup=20, timed=60)
    print(f"[TIER 2 TF32 FP32 Default] {t2}")
    time.sleep(1.0)
    t3 = _invoke_subprocess_tier("sel_amp", warmup=20, timed=60)
    print(f"[TIER 3 Selective AMP FP16] {t3}")
    time.sleep(1.0)
    paired = _invoke_subprocess_tier("e2e_paired", warmup=24, timed=120)
    t4 = paired["strip"]
    t5 = paired["deep"]
    print(f"[TIER 4 Paired Strip Re-ID] {t4}")
    print(f"[TIER 5 Paired Deep MobileNetV3 Re-ID] {t5}")

    count_deltas = [abs(a - b) for a, b in zip(t1["parity_counts"], t3["parity_counts"])]
    conf_diffs = [abs(a - b) for a, b in zip(t1["parity_confs"], t3["parity_confs"])]
    parity_summary = {
        "frames": 12,
        "frame_indices": list(range(180, 420, 20)),
        "fp32_boxes": int(sum(t1["parity_counts"])),
        "amp_fp16_boxes": int(sum(t3["parity_counts"])),
        "max_abs_count_delta": int(max(count_deltas) if count_deltas else 0),
        "mean_conf_diff": round(float(np.mean(conf_diffs)) if conf_diffs else 0.0, 5),
    }

    b1_candidates = [
        ("Strict FP32", t1["lat_mean_ms"]),
        ("TF32 FP32", t2["lat_mean_ms"]),
        ("Selective AMP FP16 (TF32=False)", t3["lat_mean_ms"]),
    ]
    fastest_label, fastest_ms = min(b1_candidates, key=lambda kv: kv[1])
    tag_tf32 = " [Fastest B=1]" if fastest_label == "TF32 FP32" else ""
    tag_amp = " [Fastest B=1]" if fastest_label.startswith("Selective AMP") else ""

    out_payload = {
        "gpu_model": t2["gpu_model"],
        "torch_version": t2["torch_version"],
        "default_runtime_mode": "TF32 FP32",
        "matmul_allow_tf32": t2["matmul_allow_tf32"],
        "cudnn_allow_tf32": t2["cudnn_allow_tf32"],
        "allow_tf32": t2["matmul_allow_tf32"] and t2["cudnn_allow_tf32"],
        "fastest_b1_mode": fastest_label,
        "fastest_b1_ms": fastest_ms,
        "frame_resolution": t2["frame_resolution"],
        "warmup_iters": 20,
        "timed_iters": 60,
        "timed_e2e_paired_iters": 120,
        "paired_protocol": paired["protocol"],
        "parity_fp32_vs_amp_fp16": parity_summary,
        "stages": [
            {
                "stage": "1. Pure GDUT-A6 Detector (Strict FP32)",
                "config": "FP32 (matmul_tf32=False, cudnn_tf32=False)",
                "latency_ms": t1["lat_mean_ms"],
                "p95_ms": t1["lat_p95_ms"],
                "p99_ms": t1["lat_p99_ms"],
                "fps": t1["fps"],
                "vram_mb": t1["vram_mb"],
                "parity_note": f"Baseline Ref ({parity_summary['fp32_boxes']}/12 boxes)",
            },
            {
                "stage": "2. Pure GDUT-A6 Detector (TF32 FP32)",
                "config": "TF32 FP32 (matmul_tf32=True, cudnn_tf32=True)",
                "latency_ms": t2["lat_mean_ms"],
                "p95_ms": t2["lat_p95_ms"],
                "p99_ms": t2["lat_p99_ms"],
                "fps": t2["fps"],
                "vram_mb": t2["vram_mb"],
                "parity_note": f"TensorCore TF32 (-{round(t1['lat_mean_ms'] - t2['lat_mean_ms'], 2)}ms vs Strict){tag_tf32}",
            },
            {
                "stage": "3. Pure GDUT-A6 Detector (Selective AMP FP16)",
                "config": "AMP FP16 + CoordConv/BiFormer FP32 guard (TF32=False)",
                "latency_ms": t3["lat_mean_ms"],
                "p95_ms": t3["lat_p95_ms"],
                "p99_ms": t3["lat_p99_ms"],
                "fps": t3["fps"],
                "vram_mb": t3["vram_mb"],
                "parity_note": f"Delta={parity_summary['max_abs_count_delta']} boxes, conf diff={parity_summary['mean_conf_diff']}{tag_amp}",
            },
            {
                "stage": "4. GDUT-A6 TF32 + ByteTrack + 3-Strip Non-Biometric Re-ID",
                "config": "TF32 FP32 + HSV/Lab + Quadrant HOG + Hungarian",
                "latency_ms": t4["lat_mean_ms"],
                "p95_ms": t4["lat_p95_ms"],
                "p99_ms": t4["lat_p99_ms"],
                "det_mean_ms": t4["det_mean_ms"],
                "reid_mean_ms": t4["reid_mean_ms"],
                "fps": t4["fps"],
                "vram_mb": t4["vram_mb"],
                "parity_note": f"Paired Shared-Det (n=120): det={t4['det_mean_ms']}ms + strip_reid={t4['reid_mean_ms']}ms (use_deep=False)",
            },
            {
                "stage": "5. Full VMS + Multi-Layer MobileNetV3 Non-Biometric Re-ID",
                "config": f"TF32 FP32 + Deep Re-ID={'GPU' if t5['deep_active'] else 'OFF'} (refresh=30)",
                "latency_ms": t5["lat_mean_ms"],
                "p95_ms": t5["lat_p95_ms"],
                "p99_ms": t5["lat_p99_ms"],
                "det_mean_ms": t5["det_mean_ms"],
                "reid_mean_ms": t5["reid_mean_ms"],
                "paired_delta_mean_ms": t5["paired_delta_mean_ms"],
                "fps": t5["fps"],
                "vram_mb": t5["vram_mb"],
                "deep_extract_calls": t5["deep_extract_calls"],
                "deep_frame_mean_ms": t5["deep_frame_mean_ms"],
                "deep_frame_p95_ms": t5["deep_frame_p95_ms"],
                "non_deep_frame_mean_ms": t5["non_deep_frame_mean_ms"],
                "parity_note": (
                    f"Paired Shared-Det (n=120): +{t5['paired_delta_mean_ms']}ms avg ({t5['deep_extract_calls']} MobileNetV3 calls; "
                    f"extract={t5['deep_frame_mean_ms']}ms vs steady={t5['non_deep_frame_mean_ms']}ms; >=25 FPS SLA)"
                ),
            },
        ],
    }

    out_path = ROOT / "Output" / "task3_eval_opt" / "gdut_vms_end2end_reid_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"[SAVED] {out_path}")

    opt_path = ROOT / "Output" / "task3_eval_opt" / "optimization_benchmark_table.json"
    if opt_path.exists():
        opt_data = json.loads(opt_path.read_text(encoding="utf-8"))
        vp = opt_data.setdefault("video_parity_fp32_vs_amp_fp16", {})
        vp["Rep-YOLO11s-GDUT-A6 (Capstone 5-Cls)"] = {
            "frames": parity_summary["frames"],
            "frame_indices": parity_summary["frame_indices"],
            "fp32_boxes": parity_summary["fp32_boxes"],
            "fp16_amp_boxes": parity_summary["amp_fp16_boxes"],
            "max_abs_count_delta": parity_summary["max_abs_count_delta"],
            "mean_conf_diff": parity_summary["mean_conf_diff"],
        }
        for rec in opt_data.get("records", []):
            if rec.get("model") == "Rep-YOLO11s-GDUT-A6 (Capstone 5-Cls)" and rec.get("stage") == "After-2 (+ CUDA AMP FP16)":
                rec["accuracy_delta"] = (
                    f"B=1: Delta {parity_summary['max_abs_count_delta']} boxes "
                    f"({parity_summary['fp32_boxes']}/{parity_summary['amp_fp16_boxes']}, conf diff {parity_summary['mean_conf_diff']})"
                )
        opt_path.write_text(json.dumps(opt_data, indent=2), encoding="utf-8")
        print(f"[UPDATED PARITY] {opt_path}")


if __name__ == "__main__":
    main()
