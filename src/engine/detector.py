"""
src/engine/detector.py
======================
Decoupled YOLO / Rep-YOLO11s Inference Engine.
- Injects custom modules (CoordConv, RepConv, BiFormerSequentialWrapper, ConvBNAct) into namespace.
- Supports GDUT-HWD 5-Class Pure Head/Helmet Taxonomy (`blue, white, yellow, red, none`) with Head-Track extraction.
- Supports hot-swapping checkpoints cleanly without restarting the web service.
"""

from __future__ import annotations
import sys
import time
import warnings
import torch
import numpy as np
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Silence ultralytics FP16 deprecation warning flood
warnings.filterwarnings("ignore", message=".*'half' is deprecated.*")
try:
    from ultralytics.utils import LOGGER
    import logging

    class UltralyticsHalfFilter(logging.Filter):
        def filter(self, record):
            return "is deprecated" not in record.getMessage()

    LOGGER.addFilter(UltralyticsHalfFilter())
except Exception:
    pass

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Register custom research modules for PyTorch unpickler compatibility
import custom_ablation_modules as cam
import ultralytics.nn.modules as un_mod
import ultralytics.nn.tasks as un_tasks
from custom_ablation_modules import BiFormerSequentialWrapper

setattr(sys.modules["__main__"], "BiFormerSequentialWrapper", BiFormerSequentialWrapper)
setattr(un_mod, "BiFormerSequentialWrapper", BiFormerSequentialWrapper)
setattr(un_tasks, "BiFormerSequentialWrapper", BiFormerSequentialWrapper)

for name in dir(cam):
    obj = getattr(cam, name)
    if isinstance(obj, type):
        setattr(sys.modules["__main__"], name, obj)
        setattr(un_mod, name, obj)
        setattr(un_tasks, name, obj)

from ultralytics import YOLO
from src.config import config


class SafetyObjectDetector:
    """YOLO Detection Wrapper with decoupled model loading, taxonomy adaptation, and head-track emission."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: Optional[str] = None,
        use_half: bool = False,
        use_selective_amp: Optional[bool] = None,
    ):
        self.model_path = model_path or config.model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"

        self.use_half = use_half and self.device.startswith("cuda")
        sel_amp_cfg = getattr(config, "use_selective_amp", True) if use_selective_amp is None else use_selective_amp
        self.use_selective_amp = bool(sel_amp_cfg) and self.device.startswith("cuda")
        if self.device.startswith("cuda"):
            allow_tf32_cfg = getattr(config, "allow_tf32", False) and (not self.use_selective_amp)
            torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32_cfg)
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32_cfg)
        self.infer_lock = threading.Lock()
        self.yolo: Optional[YOLO] = None

        # Default stub taxonomy aligns with GDUT-HWD 5-class Enterprise mode
        self.class_names: Dict[int, str] = {
            0: "blue",
            1: "white",
            2: "yellow",
            3: "red",
            4: "none",
        }
        self.has_person_class = False
        self.has_vest_class = False
        self.is_helmet_only = True
        self.last_head_tracks: np.ndarray = np.empty((0, 6), dtype=np.float32)

        if Path(self.model_path).exists() or (root_dir / self.model_path).exists():
            self.load_model(self.model_path)
        else:
            print(f"⚠️ Weights not found at startup ({self.model_path}); running in weightless GDUT 5-class CI stub mode until switched.")

    def load_model(self, model_path: str):
        """Loads or hot-swaps model weights cleanly."""
        target_path = Path(model_path)
        if not target_path.exists():
            target_path = root_dir / model_path

        if not target_path.exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        lock = getattr(self, "infer_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            self.yolo = YOLO(str(target_path))
            self.class_names = self.yolo.names

            # Dynamic Taxonomy Detection (Supports GDUT 5-class, CHV 7-class, SHWD 2-class)
            names_lower = [str(n).lower() for n in self.class_names.values()]
            self.has_person_class = any("person" in n or "worker" in n or "human" in n for n in names_lower)
            self.has_vest_class = any("vest" in n for n in names_lower)
            self.is_helmet_only = (not self.has_person_class) and any(
                n in ("blue", "white", "yellow", "red", "green", "none", "no_helmet", "helmet") for n in names_lower
            )
            if self.is_helmet_only:
                print("🪖 [Taxonomy Adapter] Detected GDUT-HWD / Helmet-Only model. Activating Head-Level ByteTrack + Upper-Torso Re-ID Mode.")

            has_custom_modules = False
            if hasattr(self.yolo, "model") and hasattr(self.yolo.model, "modules"):
                for m in self.yolo.model.modules():
                    m_type = type(m).__name__
                    if m_type in ("CoordConv", "BiFormerBlockLite", "BiFormerBlock", "BiFormerSequentialWrapper"):
                        has_custom_modules = True
                        break
            if has_custom_modules and self.use_half:
                print("⚠️ Disabling raw FP16 weights (enforcing Selective CUDA AMP FP16 with FP32 guards inside CoordConv/BiFormer).")
                self.use_half = False

            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            with torch.inference_mode():
                if self.use_selective_amp and "cuda" in str(self.device).lower():
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        self.yolo.predict(
                            dummy_img,
                            device=self.device,
                            half=False,
                            conf=self.conf_thresh,
                            verbose=False,
                        )
                else:
                    self.yolo.predict(
                        dummy_img,
                        device=self.device,
                        half=self.use_half,
                        conf=self.conf_thresh,
                        verbose=False,
                    )
            self.model_path = str(target_path)
            self.last_head_tracks = np.empty((0, 6), dtype=np.float32)
            print(
                f"✅ Model loaded successfully on {self.device.upper()} "
                f"(FP16={self.use_half}, SelectiveAMP={self.use_selective_amp}) | Classes: {self.class_names}"
            )
        finally:
            if lock is not None:
                lock.release()

    def get_head_tracks(self) -> np.ndarray:
        """Returns cached (N, 6) [x1, y1, x2, y2, conf, cls_id] head boxes from last predict()."""
        return getattr(self, "last_head_tracks", np.empty((0, 6), dtype=np.float32))

    def predict(
        self,
        frame_bgr: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: int = 640,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, float]:
        """
        Runs inference on an image / video frame.
        Returns 3-tuple for backward compatibility:
            all_detections: list of dicts filtered at conf_val
            person_boxes: numpy array (N, 6) for full-body ByteTrack when has_person_class=True (empty in GDUT mode)
            latency_ms: pure inference latency in milliseconds
        Also populates `self.last_head_tracks` (N, 6) down to `track_low_thresh` when `is_helmet_only=True`.
        """
        conf_val = conf if conf is not None else self.conf_thresh
        iou_val = iou if iou is not None else self.iou_thresh
        infer_conf = min(conf_val, getattr(config, "track_low_thresh", 0.15))

        if self.yolo is None:
            self.last_head_tracks = np.empty((0, 6), dtype=np.float32)
            return [], np.empty((0, 6), dtype=np.float32), 0.0

        t0 = time.perf_counter()
        predict_kwargs = {
            "conf": infer_conf,
            "iou": iou_val,
            "imgsz": imgsz,
            "device": self.device,
            "verbose": False,
        }
        if self.use_half and "cuda" in str(self.device).lower():
            predict_kwargs["half"] = True

        with self.infer_lock:
            with torch.inference_mode():
                if (self.use_half or self.use_selective_amp) and "cuda" in str(self.device).lower():
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        results = self.yolo.predict(frame_bgr, **predict_kwargs)
                else:
                    results = self.yolo.predict(frame_bgr, **predict_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        all_detections: List[Dict[str, Any]] = []
        person_raw: List[List[float]] = []
        head_raw: List[List[float]] = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf_score = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                c_name = self.class_names.get(cls_id, str(cls_id)).lower()

                det_dict = {
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "cls": cls_id,
                    "conf": round(conf_score, 4),
                    "name": c_name,
                }

                if conf_score >= conf_val:
                    all_detections.append(det_dict)

                if self.has_person_class:
                    is_person = ("person" in c_name or "worker" in c_name or "human" in c_name)
                    if is_person:
                        person_raw.append([x1, y1, x2, y2, conf_score, cls_id])
                elif self.is_helmet_only:
                    if "vest" not in c_name and "person" not in c_name:
                        head_raw.append([x1, y1, x2, y2, conf_score, cls_id])

        person_arr = np.array(person_raw, dtype=np.float32) if person_raw else np.empty((0, 6), dtype=np.float32)
        self.last_head_tracks = np.array(head_raw, dtype=np.float32) if head_raw else np.empty((0, 6), dtype=np.float32)
        return all_detections, person_arr, latency_ms
