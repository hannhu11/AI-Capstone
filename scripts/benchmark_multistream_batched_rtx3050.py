"""
Batched + Serialized Multi-Stream Operational Test for AAIML 2027 (RTX 3050).
Compares serialized single-frame GPU dispatch vs cross-stream micro-batching (batch=N)
across N in {1, 2, 4, 8} concurrent 1080p video streams.
"""
import sys, time, json, subprocess, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.benchmark_multistream_rtx3050 import build_deploy_rep_yolo11s, poll_nvidia_smi, run_concurrent_streams


def run_batched_streams(yolo_model, video_paths: list[Path], n_streams: int, steps: int = 120, use_half: bool = True):
    caps = [cv2.VideoCapture(str(video_paths[s % len(video_paths)])) for s in range(n_streams)]
    dummy = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(n_streams)]
    for _ in range(10):
        _ = yolo_model.predict(dummy, imgsz=640, device=0, half=use_half, verbose=False)
    torch.cuda.synchronize()

    smi_samples = []
    stop_evt = threading.Event()
    mon_thread = threading.Thread(target=poll_nvidia_smi, args=(stop_evt, smi_samples), daemon=True)
    mon_thread.start()

    def read_resize(cap):
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        return cv2.resize(frame, (640, 640))

    latencies_ms = []
    t_wall_0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(n_streams, 8)) as pool:
        for step in range(steps):
            t0 = time.perf_counter()
            imgs = list(pool.map(read_resize, caps))
            results = yolo_model.predict(imgs, imgsz=640, device=0, half=use_half, conf=0.25, iou=0.45, verbose=False)
            torch.cuda.synchronize()
            for idx, r in enumerate(results):
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for b in boxes.xyxy.cpu().numpy():
                        cv2.rectangle(imgs[idx], (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 100), 2)
            t1 = time.perf_counter()
            if step >= 10:
                latencies_ms.append((t1 - t0) * 1000.0)
    t_wall_1 = time.perf_counter()

    stop_evt.set()
    mon_thread.join(timeout=2)
    for c in caps:
        c.release()

    wall_s = t_wall_1 - t_wall_0
    total_frames = steps * n_streams
    agg_fps = total_frames / wall_s
    per_stream_fps = agg_fps / n_streams
    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "mode": "batched",
        "streams": n_streams,
        "frames_total": int(total_frames),
        "wall_s": round(wall_s, 2),
        "agg_fps": round(agg_fps, 1),
        "per_stream_fps": round(per_stream_fps, 1),
        "lat_mean_ms": round(float(arr.mean()), 2),
        "lat_p50_ms": round(float(np.percentile(arr, 50)), 2),
        "lat_p95_ms": round(float(np.percentile(arr, 95)), 2),
        "lat_p99_ms": round(float(np.percentile(arr, 99)), 2),
        "gpu_util_pct": round(float(np.mean([s["gpu_util"] for s in smi_samples])), 1) if smi_samples else 0.0,
        "vram_peak_mb": round(float(np.max([s["vram_mb"] for s in smi_samples])), 0) if smi_samples else 0.0,
        "power_mean_w": round(float(np.mean([s["power_w"] for s in smi_samples])), 1) if smi_samples else 0.0,
        "sla_25fps_ok": bool(per_stream_fps >= 25.0),
    }


def main():
    vids = [
        ROOT / "video_test" / "construction_site_workers_1080p.mp4",
        ROOT / "video_test" / "10810476-hd_1920_1080_30fps.mp4",
        ROOT / "video_test" / "19832492-hd_1920_1080_25fps.mp4",
        ROOT / "video_test" / "19832500-hd_1920_1080_25fps.mp4",
    ]
    vids = [v for v in vids if v.exists()]
    custom_ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if custom_ckpt is not None:
        assert custom_ckpt.exists(), f"Missing checkpoint: {custom_ckpt}"
    yolo = build_deploy_rep_yolo11s(custom_ckpt)
    yolo.model.to("cuda:0").eval()

    out = []
    for n in (1, 2, 4, 8):
        rb = run_batched_streams(yolo, vids, n_streams=n, steps=120, use_half=True)
        out.append(rb)
        print(f"[BATCHED N={n}] {rb}", flush=True)

    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else (ROOT / "Output" / "operational_multistream" / "rtx3050_batched_multistream_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[SAVED] {out_path}", flush=True)


if __name__ == "__main__":
    main()
