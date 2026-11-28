"""
Operational Multi-Stream Stress Test Helper on NVIDIA RTX 3050 Laptop GPU
=========================================================================
Tests concurrent real-world 1080p video decoding + preprocessing + GPU FP16/FP32
inference + NMS + overlay rendering across N in {1, 2, 4, 8} simultaneous streams.
Records per-stream FPS, aggregate FPS, latency distribution (mean/p50/p95/p99),
VRAM usage, GPU utilization, and power draw via nvidia-smi.
"""

import os
import sys
import time
import json
import subprocess
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from custom_ablation_modules import CoordConv, RepConv, BiFormerBlockLite, BiFormerSequentialWrapper

# Vectorized BiFormer patch to match deployed Rep-YOLO11s GDUT-A6 architecture
def _vec_biformer_forward(self, x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    rs = self.region_size
    pad_h = (rs - h % rs) % rs
    pad_w = (rs - w % rs) % rs
    x_pad = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
    hp, wp = x_pad.shape[-2:]
    gh, gw = hp // rs, wp // rs
    R, T = gh * gw, rs * rs
    q, k, v = self.qkv(x_pad).chunk(3, dim=1)
    q_tokens = q.view(b, c, gh, rs, gw, rs).permute(0, 2, 4, 3, 5, 1).reshape(b, R, T, c)
    k_tokens = k.view(b, c, gh, rs, gw, rs).permute(0, 2, 4, 3, 5, 1).reshape(b, R, T, c)
    v_tokens = v.view(b, c, gh, rs, gw, rs).permute(0, 2, 4, 3, 5, 1).reshape(b, R, T, c)
    q_reg = q_tokens.mean(dim=2).float()
    k_reg = k_tokens.mean(dim=2).float()
    route_logits = torch.nan_to_num(torch.matmul(q_reg, k_reg.transpose(-1, -2)) / (c ** 0.5), nan=-1e4, posinf=1e4, neginf=-1e4)
    topk = min(self.topk, R)
    route_idx = route_logits.topk(topk, dim=-1).indices
    idx_exp = route_idx.unsqueeze(-1).unsqueeze(-1).expand(b, R, topk, T, c)
    k_sel = torch.gather(k_tokens.unsqueeze(1).expand(b, R, R, T, c), 2, idx_exp).reshape(b, R, topk * T, c)
    v_sel = torch.gather(v_tokens.unsqueeze(1).expand(b, R, R, T, c), 2, idx_exp).reshape(b, R, topk * T, c)
    head_dim = c // self.num_heads
    qh = q_tokens.view(b, R, T, self.num_heads, head_dim).permute(0, 1, 3, 2, 4)
    kh = k_sel.view(b, R, topk * T, self.num_heads, head_dim).permute(0, 1, 3, 2, 4)
    vh = v_sel.view(b, R, topk * T, self.num_heads, head_dim).permute(0, 1, 3, 2, 4)
    attn_scores = torch.clamp(torch.nan_to_num(torch.matmul(qh.float(), kh.float().transpose(-1, -2)) / (head_dim ** 0.5), nan=-50.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    attn = torch.softmax(attn_scores, dim=-1)
    out = torch.matmul(attn, vh.float()).to(qh.dtype).permute(0, 1, 3, 2, 4).reshape(b, R, T, c)
    y = out.view(b, gh, gw, rs, rs, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, hp, wp)[:, :, :h, :w]
    return torch.nan_to_num(x + self.norm(self.proj(torch.nan_to_num(y, nan=0.0))), nan=0.0)

BiFormerBlockLite.forward = _vec_biformer_forward


def build_deploy_rep_yolo11s(custom_pt: Path | None = None) -> YOLO:
    if custom_pt is not None and custom_pt.exists():
        y = YOLO(str(custom_pt))
        for mod in y.model.modules():
            if hasattr(mod, "switch_to_deploy"):
                mod.switch_to_deploy()
        return y

    y = YOLO(str(ROOT / "yolo11s.pt"))
    m = DetectionModel("yolo11s.yaml", nc=2, verbose=False)
    sd_s, sd_d = y.model.state_dict(), m.state_dict()
    sd_d.update({k: v for k, v in sd_s.items() if k in sd_d and sd_d[k].shape == v.shape})
    m.load_state_dict(sd_d)
    m.names = {0: "hat", 1: "person"}

    # Stem CoordConv
    orig0 = m.model[0]
    cc = CoordConv(orig0.conv.in_channels, orig0.conv.out_channels, k=3, s=2)
    cc.i, cc.f, cc.type = orig0.i, orig0.f, "CoordConv"
    m.model[0] = cc

    # Neck-only RepConv -> fused deploy
    for idx in range(11, 23):
        layer = m.model[idx]
        if hasattr(layer, "conv") and hasattr(layer.conv, "kernel_size") and layer.conv.kernel_size == (3, 3):
            rc = RepConv(layer.conv.in_channels, layer.conv.out_channels, k=3, s=layer.conv.stride[0], deploy=True)
            rc.i, rc.f, rc.type = layer.i, layer.f, "RepConv"
            m.model[idx] = rc
        for name, sub in list(layer.named_modules()):
            if type(sub).__name__ == "Conv" and hasattr(sub, "conv") and sub.conv.kernel_size == (3, 3) and sub.conv.groups == 1:
                rc = RepConv(sub.conv.in_channels, sub.conv.out_channels, k=3, s=sub.conv.stride[0], deploy=True)
                parts = name.split(".")
                parent = layer
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], rc)

    # BiFormer P4/P5
    for idx, c_out in [(19, 256), (22, 512)]:
        layer = m.model[idx]
        bf = BiFormerBlockLite(channels=c_out, num_heads=4, region_size=8, topk=4)
        wrap = BiFormerSequentialWrapper(layer, bf)
        wrap.i, wrap.f, wrap.type = getattr(layer, "i", idx), getattr(layer, "f", -1), "BiFormerSequentialWrapper"
        m.model[idx] = wrap

    y.model = m
    return y


def poll_nvidia_smi(stop_event: threading.Event, samples: list):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                text=True,
                timeout=2,
            ).strip()
            parts = [float(x.strip()) for x in out.split(",")]
            if len(parts) == 3:
                samples.append({"gpu_util": parts[0], "vram_mb": parts[1], "power_w": parts[2]})
        except Exception:
            pass
        time.sleep(0.25)


def run_concurrent_streams(yolo_model: YOLO, video_paths: list[Path], n_streams: int, frames_per_stream: int = 120, use_half: bool = True):
    gpu_lock = threading.Lock()
    latencies_ms = []
    lat_lock = threading.Lock()
    frame_counts = [0] * n_streams

    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(15):
        _ = yolo_model.predict(dummy, imgsz=640, device=0, half=use_half, verbose=False)
    torch.cuda.synchronize()

    smi_samples = []
    stop_evt = threading.Event()
    mon_thread = threading.Thread(target=poll_nvidia_smi, args=(stop_evt, smi_samples), daemon=True)
    mon_thread.start()

    barrier = threading.Barrier(n_streams)

    def stream_worker(sid: int):
        vp = video_paths[sid % len(video_paths)]
        cap = cv2.VideoCapture(str(vp))
        barrier.wait()
        local_lats = []
        done = 0
        while done < frames_per_stream:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break
            resized = cv2.resize(frame, (640, 640))
            with gpu_lock:
                res = yolo_model.predict(resized, imgsz=640, device=0, half=use_half, conf=0.25, iou=0.45, verbose=False)
                torch.cuda.synchronize()
            # Lightweight industrial bounding-box overlay (matches SCADA render)
            boxes = res[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                for b in xyxy:
                    x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 200, 100), 2)
            t1 = time.perf_counter()
            if done >= 10:  # exclude initial 10 frames per stream
                local_lats.append((t1 - t0) * 1000.0)
            done += 1
        cap.release()
        frame_counts[sid] = done
        with lat_lock:
            latencies_ms.extend(local_lats)

    t_wall_0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_streams) as pool:
        futs = [pool.submit(stream_worker, s) for s in range(n_streams)]
        for f in futs:
            f.result()
    t_wall_1 = time.perf_counter()

    stop_evt.set()
    mon_thread.join(timeout=2)

    wall_s = t_wall_1 - t_wall_0
    total_frames = sum(frame_counts)
    agg_fps = total_frames / wall_s
    per_stream_fps = agg_fps / n_streams
    arr = np.array(latencies_ms, dtype=np.float64)
    gpu_util = float(np.mean([s["gpu_util"] for s in smi_samples])) if smi_samples else 0.0
    vram_max = float(np.max([s["vram_mb"] for s in smi_samples])) if smi_samples else 0.0
    pwr_mean = float(np.mean([s["power_w"] for s in smi_samples])) if smi_samples else 0.0

    return {
        "streams": n_streams,
        "frames_total": int(total_frames),
        "wall_s": round(wall_s, 2),
        "agg_fps": round(agg_fps, 1),
        "per_stream_fps": round(per_stream_fps, 1),
        "lat_mean_ms": round(float(arr.mean()), 2),
        "lat_p50_ms": round(float(np.percentile(arr, 50)), 2),
        "lat_p95_ms": round(float(np.percentile(arr, 95)), 2),
        "lat_p99_ms": round(float(np.percentile(arr, 99)), 2),
        "gpu_util_pct": round(gpu_util, 1),
        "vram_peak_mb": round(vram_max, 0),
        "power_mean_w": round(pwr_mean, 1),
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
    assert vids, "No test videos found in video_test/"

    custom_ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    yolo = build_deploy_rep_yolo11s(custom_ckpt)
    yolo.model.to("cuda:0").eval()
    print(f"[MODEL] Deploy Rep-YOLO11s ready | params={sum(p.numel() for p in yolo.model.parameters()):,}", flush=True)

    results = []
    for n in (1, 2, 4, 8):
        rec = run_concurrent_streams(yolo, vids, n_streams=n, frames_per_stream=120, use_half=True)
        results.append(rec)
        print(f"[STREAM N={n}] {rec}", flush=True)

    out_dir = ROOT / "Output" / "operational_multistream"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "rtx3050_multistream_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[SAVED] {out_json}", flush=True)


if __name__ == "__main__":
    main()
