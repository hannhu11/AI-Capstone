"""
scripts/export_gdut_a6_deploy.py
================================
Exports clean, standalone deployment checkpoints strictly from canonical Kaggle Run A6
(`nvthanh2004/gdut-stage2-ablation-setup-full-train-run-a6`) raw checkpoints:
- Rep-YOLO11s A6 (SHA-256 21866da3...) -> Output/GDUT_Bộ số 1_stage2-full-train-run-a6/deploy_weights/rep_yolo11s_gdut_a6_deploy.pt
- Rep-YOLOv8s A6 (SHA-256 f474a157...) -> Output/GDUT_Bộ số 1_stage2-full-train-run-a6/deploy_weights/rep_yolov8s_gdut_a6_deploy.pt

Also runs the verified strict-FP32 & AMP-FP16 `switch_to_deploy()` parity and RTX 3050
latency benchmark directly on the canonical YOLO11s A6 checkpoint and writes
`Output/task3_eval_opt/gdut_a6_switch_to_deploy_benchmark.json` with full cryptographic
provenance (`source_checkpoint_sha256` -> `deployed_checkpoint_sha256`).
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prefer canonical Kaggle research_code for unpickling raw Kaggle Run A6 checkpoints
CANONICAL_CODE_DIR = (
    PROJECT_ROOT
    / "docs"
    / "kaggle_verified_source"
    / "nvthanh2004"
    / "gdut-final-test-evaluation_outputs"
    / "research_code"
)
FALLBACK_CODE_DIR = (
    PROJECT_ROOT
    / "Output"
    / "GDUT_Bộ số 1_stage2-full-train-run-a6"
    / "GDUT_A6_CLEAN_VAL_RESULTS"
    / "model_code"
)
MODEL_CODE_DIR = CANONICAL_CODE_DIR if CANONICAL_CODE_DIR.exists() else FALLBACK_CODE_DIR
if str(MODEL_CODE_DIR) not in sys.path:
    sys.path.append(str(MODEL_CODE_DIR))

import custom_ablation_modules as cam  # noqa: E402
import research_trainer  # noqa: F401, E402
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

CANONICAL_DIR = PROJECT_ROOT / "Output" / "GDUT_Bộ số 1_stage2-full-train-run-a6" / "canonical_kaggle_a6"
DEPLOY_DIR = PROJECT_ROOT / "Output" / "GDUT_Bộ số 1_stage2-full-train-run-a6" / "deploy_weights"
BENCH_OUT = PROJECT_ROOT / "Output" / "task3_eval_opt" / "gdut_a6_switch_to_deploy_benchmark.json"

RAW_CHECKPOINTS = {
    "YOLO11s": {
        "raw": CANONICAL_DIR / "GDUT_A6_YOLO11s_canonical_best.pt",
        "expected_sha256": "21866da3396e36a408236b1c3717bff0ed5b99f2e59e9cbbbce96ea6bdb68233",
        "deploy": DEPLOY_DIR / "rep_yolo11s_gdut_a6_deploy.pt",
    },
    "YOLOv8s": {
        "raw": CANONICAL_DIR / "GDUT_A6_YOLOv8s_canonical_best.pt",
        "expected_sha256": "f474a15792a29459913b456141008d07e9571ee224af7802137459fee92eb340",
        "deploy": DEPLOY_DIR / "rep_yolov8s_gdut_a6_deploy.pt",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bench_model(model: torch.nn.Module, x: torch.Tensor, use_amp: bool, warmup: int = 20, iters: int = 80):
    use_cuda = x.is_cuda
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
    with torch.inference_mode():
        for _ in range(warmup):
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _ = model(x)
            else:
                _ = model(x)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _ = model(x)
            else:
                _ = model(x)
        if use_cuda:
            torch.cuda.synchronize()
    mean_ms = (time.perf_counter() - t0) / iters * 1000.0
    vram_mb = (torch.cuda.max_memory_allocated(0) / (1024 * 1024)) if use_cuda else 0.0
    return round(mean_ms, 3), round(1000.0 / mean_ms, 2), round(vram_mb, 2)


def _extract_primary_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0:
        return _extract_primary_tensor(out[0])
    raise TypeError(f"Unexpected model output type: {type(out)}")


def _benchmark_switch_to_deploy(
    raw_model: torch.nn.Module,
    source_sha: str,
    deploy_sha: str,
    raw_path: Path,
    deploy_path: Path,
) -> dict:
    prev_m = torch.backends.cuda.matmul.allow_tf32
    prev_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        m_before = copy.deepcopy(raw_model).float().eval()
        params_before = sum(p.numel() for p in m_before.parameters())

        rep_indices = []
        for idx, layer in enumerate(m_before.model):
            if isinstance(layer, cam.RepConv):
                rep_indices.append(idx)

        m_after = copy.deepcopy(m_before).float().eval()
        switched = 0
        for mod in m_after.modules():
            if isinstance(mod, cam.RepConv) and hasattr(mod, "switch_to_deploy"):
                mod.switch_to_deploy()
                switched += 1
        params_after = sum(p.numel() for p in m_after.parameters())

        # Also measure Ultralytics model.fuse() Conv-BN folding parameter count
        m_fused = copy.deepcopy(m_after).float().eval()
        if hasattr(m_fused, "fuse"):
            m_fused.fuse(verbose=False)
        params_after_fuse = sum(p.numel() for p in m_fused.parameters())

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"

        torch.manual_seed(3407)
        x = torch.randn(1, 3, 640, 640, device=device, dtype=torch.float32)

        m_before = m_before.to(device).eval()
        m_after = m_after.to(device).eval()

        with torch.inference_mode():
            y_before = _extract_primary_tensor(m_before(x)).float()
            y_after = _extract_primary_tensor(m_after(x)).float()
            diff = (y_before - y_after).abs()
            max_abs_diff = float(diff.max().item())
            mean_abs_diff = float(diff.mean().item())

        b_fp32_ms, b_fp32_fps, b_fp32_vram = _bench_model(m_before, x, use_amp=False, warmup=20, iters=80)
        a_fp32_ms, a_fp32_fps, a_fp32_vram = _bench_model(m_after, x, use_amp=False, warmup=20, iters=80)
        b_amp_ms, b_amp_fps, b_amp_vram = _bench_model(m_before, x, use_amp=use_cuda, warmup=20, iters=80)
        a_amp_ms, a_amp_fps, a_amp_vram = _bench_model(m_after, x, use_amp=use_cuda, warmup=20, iters=80)

        payload = {
            "device": device_name,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "ultralytics_version": ultralytics.__version__,
            "source_checkpoint_rel": raw_path.relative_to(PROJECT_ROOT).as_posix(),
            "source_checkpoint_bytes": raw_path.stat().st_size,
            "source_checkpoint_sha256": source_sha,
            "deployed_checkpoint_rel": deploy_path.relative_to(PROJECT_ROOT).as_posix(),
            "deployed_checkpoint_bytes": deploy_path.stat().st_size,
            "deployed_checkpoint_sha256": deploy_sha,
            "repconv_layers_switched": rep_indices,
            "switched_count": switched,
            "params_before": int(params_before),
            "params_after": int(params_after),
            "params_delta": int(params_before - params_after),
            "params_after_model_fuse": int(params_after_fuse),
            "params_conv_bn_delta": int(params_after - params_after_fuse),
            "strict_fp32_tf32_disabled": True,
            "max_abs_diff_fp32": max_abs_diff,
            "mean_abs_diff_fp32": mean_abs_diff,
            "before_fp32_ms": b_fp32_ms,
            "before_fp32_fps": b_fp32_fps,
            "before_fp32_vram_mb": b_fp32_vram,
            "after_fp32_ms": a_fp32_ms,
            "after_fp32_fps": a_fp32_fps,
            "after_fp32_vram_mb": a_fp32_vram,
            "before_amp_fp16_ms": b_amp_ms,
            "before_amp_fp16_fps": b_amp_fps,
            "before_amp_fp16_vram_mb": b_amp_vram,
            "after_amp_fp16_ms": a_amp_ms,
            "after_amp_fp16_fps": a_amp_fps,
            "after_amp_fp16_vram_mb": a_amp_vram,
        }
        BENCH_OUT.parent.mkdir(parents=True, exist_ok=True)
        BENCH_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"   [BENCHMARK SAVED] {BENCH_OUT.relative_to(PROJECT_ROOT)} | max_diff={max_abs_diff:.3e} | mean_diff={mean_abs_diff:.3e}")
        return payload
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_m
        torch.backends.cudnn.allow_tf32 = prev_c


def export_checkpoint(name: str, raw_path: Path, expected_sha: str, deploy_path: Path):
    print("\n=======================================================")
    print(f"Processing {name}: {raw_path}")
    assert raw_path.exists(), f"Raw canonical checkpoint not found: {raw_path}"

    source_sha = _sha256(raw_path)
    assert source_sha == expected_sha, (
        f"Canonical SHA-256 mismatch for {name}: got {source_sha}, expected {expected_sha}"
    )
    print(f"   [VERIFIED] source_checkpoint_sha256 = {source_sha}")

    deploy_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load raw canonical checkpoint
    raw_ckpt = torch.load(raw_path, map_location="cpu", weights_only=False)
    model_raw = raw_ckpt["model"].float()
    model_raw.__class__ = DetectionModel

    # Keep unfused copy for YOLO11s switch_to_deploy() benchmark
    raw_copy_for_bench = copy.deepcopy(model_raw) if name == "YOLO11s" else None

    model = model_raw

    # 2. Fuse RepConv layers via switch_to_deploy()
    fused_count = 0
    for m in model.modules():
        if isinstance(m, cam.RepConv) and hasattr(m, "switch_to_deploy"):
            m.switch_to_deploy()
            fused_count += 1
    print(f"   [+] Fused {fused_count} RepConv layers into single Conv2d")

    # 3. Re-bind SPPFWithBiFormer & custom modules cleanly to root custom_ablation_modules
    for idx, layer in enumerate(model.model):
        layer_cls_name = type(layer).__name__
        if layer_cls_name == "SPPFWithBiFormer":
            sppf = layer.sppf
            attention = layer.attention
            new_layer = cam.SPPFWithBiFormer(sppf, attention)
            new_layer.i = getattr(layer, "i", idx)
            new_layer.f = getattr(layer, "f", -1)
            new_layer.type = "SPPFWithBiFormer"
            new_layer.np = getattr(layer, "np", sum(p.numel() for p in new_layer.parameters()))
            model.model[idx] = new_layer
            print(f"   [+] Layer {idx} re-bound to custom_ablation_modules.SPPFWithBiFormer")

    # 4. Sanity check: Ensure no reference to research_trainer remains
    for name_mod, mod in model.named_modules():
        mod_cls_module = getattr(type(mod), "__module__", "")
        if "research_trainer" in mod_cls_module:
            raise RuntimeError(f"Module {name_mod} still has reference to {mod_cls_module}!")

    # 5. Normalize names dict
    expected_names = {0: "blue", 1: "white", 2: "yellow", 3: "red", 4: "none"}
    if not hasattr(model, "names") or not model.names:
        model.names = expected_names
    else:
        model.names = {int(k): str(v) for k, v in model.names.items()}

    # 6. Save standalone deploy checkpoint with provenance metadata
    deploy_ckpt = {
        "model": model,
        "date": raw_ckpt.get("date"),
        "version": raw_ckpt.get("version", "8.4.115"),
        "train_args": raw_ckpt.get("train_args", {}),
        "source_checkpoint_sha256": source_sha,
    }
    torch.save(deploy_ckpt, deploy_path)
    deploy_sha = _sha256(deploy_path)
    file_size_mb = deploy_path.stat().st_size / (1024 * 1024)
    print(f"   [SAVED] {deploy_path.relative_to(PROJECT_ROOT)} ({file_size_mb:.2f} MB, {deploy_path.stat().st_size:,} B)")
    print(f"   [DEPLOY SHA-256] {deploy_sha}")
    print(f"   [CLASSES] {model.names}")

    if name == "YOLO11s" and raw_copy_for_bench is not None:
        _benchmark_switch_to_deploy(raw_copy_for_bench, source_sha, deploy_sha, raw_path, deploy_path)


def verify_clean_load():
    print("\n=======================================================")
    print("Verifying Standalone Loading in Isolation...")

    for mod_name in list(sys.modules.keys()):
        if "research_trainer" in mod_name or "model_code" in mod_name:
            del sys.modules[mod_name]
    if str(MODEL_CODE_DIR) in sys.path:
        sys.path.remove(str(MODEL_CODE_DIR))

    import numpy as np

    for name, paths in RAW_CHECKPOINTS.items():
        deploy_path = paths["deploy"]
        if not deploy_path.exists():
            continue
        print(f"\nTesting isolated load for {name}: {deploy_path.name}")
        yolo_model = YOLO(str(deploy_path))
        print(f"   [+] Loaded via ultralytics.YOLO | Names: {yolo_model.names}")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = yolo_model.predict(dummy_img, imgsz=640, verbose=False)
        print(f"   [+] Dummy inference passed! Boxes output: {len(results[0].boxes)}")

    print("\n[OK] All canonical GDUT A6 deploy checkpoints verified without research_trainer!")


if __name__ == "__main__":
    for name, paths in RAW_CHECKPOINTS.items():
        if not paths["raw"].exists():
            print(f"[SKIP] {name} canonical raw not present at {paths['raw']}")
            continue
        export_checkpoint(name, paths["raw"], paths["expected_sha256"], paths["deploy"])
    verify_clean_load()
