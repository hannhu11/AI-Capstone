"""
src/engine/gdut_reid.py
=======================
Privacy-Safe Head-Anchored Upper-Torso Persistent Re-Identification Engine v2.3
for GDUT-HWD 5-Class (`blue, white, yellow, red, none`) Surveillance.

Upgrades over v2.0–v2.2:
1. Anti-Wall-Bleed Anatomical ROI & Gaussian Spatial Weighting:
   - Narrowed torso projection + anatomical per-band horizontal crops (collar/chest/mid-torso).
   - Border background sampling & median-wall distance suppression so identical walls
     behind different people on webcam C960 Ultra do not inflate similarity.
2. Same-Uniform Discriminative Descriptors:
   - Weighted 3-strip HSV + Lab + BGR appearance + 4-quadrant local gradient texture
     (HOG-8 x 4 quadrants + luminance grids, 520-D total: 474-D appearance + 46-D texture).
   - Multi-layer MobileNetV3-Small feature extraction (`features[6]` + `features[11]` +
     final `avgpool` -> 712-D) capturing fine straps, seams, badges, and fabric texture
     while remaining 100% non-biometric (no facial recognition).
3. Confirm-Before-Mint & Global Hungarian Assignment:
   - Provisional `_tentative_tracks` buffer prevents transient 1–5 frame noise detections
     from ever minting canonical IDs or inflating `self.next_id`; promotion runs Hungarian +
     re-match at `min_confirm_hits >= 6` before minting `cid = self.next_id`.
   - `linear_sum_assignment`, Coexistence Exclusion Graph, Ambiguity Margin Guard
     (`>= 0.06` tie margin), Pre-Warm, and Reference Enrollment.
"""

from __future__ import annotations
import cv2
import time
import numpy as np
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from scipy.optimize import linear_sum_assignment

from src.config import DEFAULT_ROLE_HIERARCHY


def project_upper_torso_roi(
    head_box: np.ndarray | List[float],
    frame_shape: Tuple[int, ...],
) -> Tuple[int, int, int, int]:
    """
    Projects a tightened anatomical upper-torso ROI below a head/helmet bounding box
    while excluding outer wall background and head cranium.
    """
    h_img, w_img = int(frame_shape[0]), int(frame_shape[1])
    if h_img <= 1 or w_img <= 1:
        return 0, 0, 1, 1

    hx1, hy1, hx2, hy2 = [float(v) for v in head_box[:4]]
    hw = max(2.0, hx2 - hx1)
    hh = max(2.0, hy2 - hy1)
    cx = 0.5 * (hx1 + hx2)

    tx1 = int(max(0, min(w_img - 1, round(cx - 1.05 * hw))))
    tx2 = int(max(tx1 + 1, min(w_img, round(cx + 1.05 * hw))))
    ty1 = int(max(0, min(h_img - 1, round(hy2 + 0.15 * hh))))
    ty2 = int(max(ty1 + 1, min(h_img, round(hy2 + 2.8 * hh))))

    # Bottom-border fallback: capture visible shoulder/collar band when head sits at bottom edge
    if (ty2 - ty1) < 8:
        fb_y1 = int(max(0, min(h_img - 1, round(hy1 + 0.65 * hh))))
        fb_y2 = int(max(fb_y1 + 1, h_img))
        fb_x1 = int(max(0, min(w_img - 1, round(cx - 1.0 * hw))))
        fb_x2 = int(max(fb_x1 + 1, min(w_img, round(cx + 1.0 * hw))))
        if (fb_y2 - fb_y1) > (ty2 - ty1):
            return fb_x1, fb_y1, fb_x2, fb_y2

    return tx1, ty1, tx2, ty2


def _sample_border_wall_bgr(
    frame_bgr: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    """Samples left/right exterior margins adjacent to the torso ROI to estimate background wall color."""
    h_img, w_img = frame_bgr.shape[:2]
    tx1, ty1, tx2, ty2 = roi
    y_step = max(1, (ty2 - ty1) // 48)
    samples = []
    if tx1 >= 8:
        lx1 = max(0, tx1 - 18)
        lx2 = max(lx1 + 1, tx1 - 2)
        if lx2 > lx1 and ty2 > ty1:
            samples.append(frame_bgr[ty1:ty2:y_step, lx1:lx2].reshape(-1, 3))
    if (w_img - tx2) >= 8:
        rx1 = min(w_img - 1, tx2 + 2)
        rx2 = min(w_img, tx2 + 18)
        if rx2 > rx1 and ty2 > ty1:
            samples.append(frame_bgr[ty1:ty2:y_step, rx1:rx2].reshape(-1, 3))
    if not samples:
        return None
    cat = np.concatenate(samples, axis=0)
    if cat.shape[0] < 16:
        return None
    return np.median(cat, axis=0).astype(np.float32)


_GAUSS_MASK_64x96 = np.exp(
    -0.5
    * (
        (np.linspace(-1.0, 1.0, 64, dtype=np.float32)[None, :] / 0.55) ** 2
        + (np.linspace(-1.0, 1.0, 96, dtype=np.float32)[:, None] / 0.85) ** 2
    )
).astype(np.float32)


def _compute_quadrant_hog_texture(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Computes a 46-D structural & spatial layout texture descriptor:
    - 2x2 spatial quadrants x 8 HOG orientation bins + 8 edge/contrast stats (40-D)
    - 3x2 zero-mean normalized spatial luminance layout (6-D) to distinguish horizontal
      reflective bars from diagonal straps/vertical pockets on identical-color uniforms.
    """
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 8:
        return np.zeros(46, dtype=np.float32)

    gray = cv2.cvtColor(cv2.resize(crop_bgr, (48, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_smooth = cv2.GaussianBlur(gray, (3, 3), 0.7)
    gx = cv2.Sobel(gray_smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi  # unsigned [0, pi)

    bins = np.floor((ang / np.pi) * 8.0).astype(np.int32)
    bins = np.clip(bins, 0, 7)

    feats: List[np.ndarray] = []
    stats: List[float] = []
    qh, qw = 32, 24
    for ry in (0, 1):
        for rx in (0, 1):
            y0, y1 = ry * qh, (ry + 1) * qh
            x0, x1 = rx * qw, (rx + 1) * qw
            q_mag = mag[y0:y1, x0:x1]
            q_bin = bins[y0:y1, x0:x1]
            q_gray = gray[y0:y1, x0:x1]

            hog = np.bincount(q_bin.ravel(), weights=q_mag.ravel(), minlength=8).astype(np.float32)
            hn = float(np.linalg.norm(hog))
            if hn > 1e-6:
                hog /= hn
            feats.append(hog)
            stats.append(float(q_mag.mean()) / 255.0)
            stats.append(float(q_gray.std()) / 128.0)

    # 3x2 spatial luminance pattern (centered & scaled so solid shirts produce ~0, textured seams separate cleanly)
    grid = cv2.resize(gray_smooth, (2, 3), interpolation=cv2.INTER_AREA).flatten()
    grid_centered = (grid - float(grid.mean())) / 64.0

    tex = np.concatenate([*feats, np.asarray(stats, dtype=np.float32), grid_centered.astype(np.float32)]).astype(np.float32)
    tn = float(np.linalg.norm(tex))
    if tn > 1e-6:
        tex /= tn
    return tex


def _weighted_hist_1d(vals: np.ndarray, weights: np.ndarray, bins: int, hi: float) -> np.ndarray:
    scaled = np.clip((vals.astype(np.float32) / hi) * bins, 0, bins - 1e-4).astype(np.int32)
    h = np.bincount(scaled.ravel(), weights=weights.ravel(), minlength=bins).astype(np.float32)
    n = float(np.linalg.norm(h))
    if n > 1e-6:
        h /= n
    return h


def compute_three_strip_torso_signature(
    frame_bgr: Optional[np.ndarray],
    roi: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Computes a 520-D descriptor packed as two independently normalized subspaces:
    - [0:474]: L2-normalized 3 anatomical horizontal strips (158-D each), weighted by
      2D Gaussian core mask + exterior wall-color suppression.
    - [474:520]: L2-normalized 46-D 4-quadrant HOG + spatial layout texture.
    """
    feat_dim = 520
    if frame_bgr is None or frame_bgr.size == 0:
        return np.zeros(feat_dim, dtype=np.float32)

    h_img, w_img = frame_bgr.shape[:2]
    tx1, ty1, tx2, ty2 = roi
    tx1 = max(0, min(w_img - 1, int(tx1)))
    ty1 = max(0, min(h_img - 1, int(ty1)))
    tx2 = max(tx1 + 1, min(w_img, int(tx2)))
    ty2 = max(ty1 + 1, min(h_img, int(ty2)))

    crop = frame_bgr[ty1:ty2, tx1:tx2]
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
        return np.zeros(feat_dim, dtype=np.float32)

    wall_bg = _sample_border_wall_bgr(frame_bgr, (tx1, ty1, tx2, ty2))

    # Bound torso crop to canonical (64x96) so 1080p full-frame Re-ID stays O(1) fast on CPU
    if crop.shape[0] > 96 or crop.shape[1] > 64:
        crop = cv2.resize(crop, (64, 96), interpolation=cv2.INTER_AREA)

    ch, cw = crop.shape[:2]

    # Build 2D Gaussian core mask favoring foreground body core over outer fringes
    if ch == 96 and cw == 64:
        gauss_mask = _GAUSS_MASK_64x96.copy()
    else:
        yy = np.linspace(-1.0, 1.0, ch, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, cw, dtype=np.float32)[None, :]
        gauss_mask = np.exp(-0.5 * ((xx / 0.55) ** 2 + (yy / 0.85) ** 2)).astype(np.float32)

    if wall_bg is not None:
        diff = np.linalg.norm(crop.astype(np.float32) - wall_bg[None, None, :], axis=2)
        fg_ratio = float(np.mean(diff >= 18.0))
        if fg_ratio >= 0.25:
            fg_weight = np.where(diff < 18.0, 0.15, 1.0).astype(np.float32)
            gauss_mask = gauss_mask * fg_weight

    if ch >= 6:
        b1 = max(1, ch // 3)
        b2 = max(b1 + 1, (2 * ch) // 3)
        band_defs = [
            (0, b1, 0.14, 0.86),     # Strip 0: collar & inner shoulders (narrower to avoid walls)
            (b1, b2, 0.02, 0.98),    # Strip 1: full upper chest
            (b2, ch, 0.06, 0.94),    # Strip 2: lower torso
        ]
    else:
        band_defs = [(0, ch, 0.0, 1.0), (0, ch, 0.0, 1.0), (0, ch, 0.0, 1.0)]

    strips: List[np.ndarray] = []
    for y0, y1, xr0, xr1 in band_defs:
        x0 = max(0, min(cw - 1, int(round(xr0 * cw))))
        x1 = max(x0 + 1, min(cw, int(round(xr1 * cw))))
        band = crop[y0:y1, x0:x1]
        b_mask = gauss_mask[y0:y1, x0:x1]
        if band.size == 0:
            strips.append(np.zeros(158, dtype=np.float32))
            continue

        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        h_idx = np.clip((hsv[:, :, 0].astype(np.float32) / 180.0) * 12.0, 0, 11.99).astype(np.int32)
        s_idx = np.clip((hsv[:, :, 1].astype(np.float32) / 256.0) * 8.0, 0, 7.99).astype(np.int32)
        hs_flat = (h_idx * 8 + s_idx).ravel()
        hist_hs = np.bincount(hs_flat, weights=b_mask.ravel(), minlength=96).astype(np.float32)
        n_hs = float(np.linalg.norm(hist_hs))
        if n_hs > 1e-6:
            hist_hs /= n_hs

        lab = cv2.cvtColor(band, cv2.COLOR_BGR2Lab)
        hist_a = _weighted_hist_1d(lab[:, :, 1], b_mask, 16, 256.0)
        hist_b = _weighted_hist_1d(lab[:, :, 2], b_mask, 16, 256.0)
        n_ab = float(np.linalg.norm(np.concatenate([hist_a, hist_b])))
        if n_ab > 1e-6:
            hist_a /= n_ab
            hist_b /= n_ab

        bgr_hists = [_weighted_hist_1d(band[:, :, c_i], b_mask, 8, 256.0) for c_i in range(3)]

        w_sum = max(1e-6, float(b_mask.sum()))
        w_3 = b_mask[:, :, None]
        bf = band.astype(np.float32)
        mean_bgr = (bf * w_3).sum(axis=(0, 1)) / (w_sum * 255.0)
        var_bgr = (((bf - (mean_bgr * 255.0)[None, None, :]) ** 2) * w_3).sum(axis=(0, 1)) / w_sum
        std_bgr = np.sqrt(np.maximum(0.0, var_bgr)) / 128.0
        moments = np.concatenate([mean_bgr, std_bgr]).astype(np.float32) * 0.5

        band_vec = np.concatenate([hist_hs, hist_a, hist_b, *bgr_hists, moments]).astype(np.float32)
        strips.append(band_vec)

    color_vec = np.concatenate(strips).astype(np.float32)
    cn = float(np.linalg.norm(color_vec))
    if cn > 1e-6:
        color_vec /= cn
    tex_vec = _compute_quadrant_hog_texture(crop).astype(np.float32)
    return np.concatenate([color_vec, tex_vec]).astype(np.float32)


def cosine_sim(v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> float:
    """Computes cosine similarity for L2-normalized vectors in [-1, 1]."""
    if v1 is None or v2 is None or v1.size == 0 or v2.size == 0:
        return 0.0
    if v1.size != v2.size:
        m = min(v1.size, v2.size)
        v1 = v1[:m]
        v2 = v2[:m]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


class MobileNetV3SmallExtractor:
    """
    Non-biometric Multi-Layer Upper-Torso Extractor (`features[6]` + `features[11]` + `avgpool`).
    Extracts 712-D L2-normalized descriptor capturing fine fabric texture, seams, and stripes
    for identical-uniform discrimination without any facial embedding.
    """

    def __init__(self, enabled: bool = True, device: Optional[str] = None):
        import threading

        self.enabled = bool(enabled)
        self.device = device
        self._init_lock = threading.Lock()
        self._initialized = False
        self._available = False
        self._features = None
        self._avgpool = None
        self._torch = None
        self._mean = None
        self._std = None
        self.extract_calls: int = 0

    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        if not self._initialized:
            with self._init_lock:
                if not self._initialized:
                    self._lazy_init()
        return self._available

    def prewarm(self) -> bool:
        """Eagerly initializes weights & runs a dummy forward pass outside the frame lock."""
        if not self.available:
            return False
        try:
            dummy = self._torch.zeros((1, 3, 128, 128), device=self.device, dtype=self._torch.float32)
            with self._torch.no_grad():
                _ = self._forward_multiscale(dummy)
            return True
        except Exception:
            return False

    def _lazy_init(self):
        self._initialized = True
        if not self.enabled:
            self._available = False
            return
        try:
            import os
            import torch
            import torchvision.models as tv_models

            cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
            has_cached = cache_dir.exists() and any(cache_dir.glob("mobilenet_v3_small*.pth"))
            allow_dl = os.getenv("GDUT_REID_ALLOW_DOWNLOAD", "1").lower() in ("1", "true", "yes")
            if not has_cached and not allow_dl:
                self._available = False
                return

            weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT
            model = tv_models.mobilenet_v3_small(weights=weights)
            dev = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            if dev.startswith("cuda") and not torch.cuda.is_available():
                dev = "cpu"
            self.device = dev
            model.eval().to(dev)
            self._features = model.features
            self._avgpool = model.avgpool
            self._torch = torch
            self._mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
            self._std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
            self._available = True
        except Exception:
            self._available = False
            self._features = None

    def _forward_multiscale(self, x_norm):
        import torch.nn.functional as F

        feat_mid6 = None
        feat_mid11 = None
        h = x_norm
        for idx, layer in enumerate(self._features):
            h = layer(h)
            if idx == 6:
                feat_mid6 = F.normalize(self._avgpool(h).flatten(1), dim=1)
            elif idx == 11:
                feat_mid11 = F.normalize(self._avgpool(h).flatten(1), dim=1)
        feat_high = F.normalize(self._avgpool(h).flatten(1), dim=1)

        parts = []
        if feat_mid6 is not None:
            parts.append(0.45 * feat_mid6)
        if feat_mid11 is not None:
            parts.append(0.55 * feat_mid11)
        parts.append(1.0 * feat_high)
        combined = self._torch.cat(parts, dim=1)
        return F.normalize(combined, dim=1).squeeze(0)

    def extract(self, frame_bgr: Optional[np.ndarray], roi: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        if not self.available or frame_bgr is None or self._features is None or self._torch is None:
            return None
        tx1, ty1, tx2, ty2 = roi
        crop = frame_bgr[ty1:ty2, tx1:tx2]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        try:
            rgb = cv2.cvtColor(cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0
            tensor = (tensor - self._mean) / self._std
            with self._torch.no_grad():
                vec = self._forward_multiscale(tensor).detach().cpu().numpy().astype(np.float32)
            self.extract_calls += 1
            norm = float(np.linalg.norm(vec))
            if norm > 1e-6:
                vec /= norm
            return vec
        except Exception:
            return None


class UpperTorsoIdentityProfile:
    """Persistent Re-ID profile bound to a canonical personnel ID."""

    def __init__(
        self,
        worker_id: int,
        head_box: np.ndarray,
        helmet_class: str,
        strip_sig: np.ndarray,
        deep_sig: Optional[np.ndarray],
        torso_roi: Tuple[int, int, int, int],
        frame_idx: int,
        now_ts: float,
    ):
        self.worker_id = worker_id
        self.custom_label: Optional[str] = None
        self.current_helmet_color: str = helmet_class
        self.min_history_votes: int = 10
        self.helmet_color_votes: Dict[str, int] = {}
        self._color_streak_cls: Optional[str] = None
        self._color_streak_len: int = 0
        # Hysteresis guard: require >= 10 confirmed frames before recording PRIOR <COLOR>
        self.last_known_helmet_color: Optional[str] = None
        if helmet_class and helmet_class != "none":
            self.helmet_color_votes[helmet_class] = 1
            self._color_streak_cls = helmet_class
            self._color_streak_len = 1
        self.status: str = "VIOLATION" if helmet_class == "none" else "COMPLIANT"
        self.violations: List[str] = ["MISSING_SAFETY_HELMET"] if helmet_class == "none" else []

        self.last_box = np.asarray(head_box[:4], dtype=np.float32).copy()
        self.last_torso_roi = torso_roi
        self.first_frame = frame_idx
        self.last_seen_frame = frame_idx
        self.first_seen_ts = now_ts
        self.last_seen_ts = now_ts
        self.total_frames_seen = 1
        self.consecutive_hits = 1
        self.is_active = True
        self.is_confirmed = False
        self.last_deep_frame = frame_idx
        self.coexisted_ids: Set[int] = set()

        hw = max(2.0, float(head_box[2] - head_box[0]))
        tw = max(2.0, float(torso_roi[2] - torso_roi[0]))
        self.shoulder_head_ratio: float = tw / hw

        self.strip_gallery: deque[np.ndarray] = deque(maxlen=5)
        self.strip_ema: Optional[np.ndarray] = None
        if strip_sig is not None and np.linalg.norm(strip_sig) > 1e-6:
            self.strip_gallery.append(strip_sig.copy())
            self.strip_ema = strip_sig.copy()

        self.deep_gallery: deque[np.ndarray] = deque(maxlen=7)
        self.deep_ema: Optional[np.ndarray] = None
        if deep_sig is not None and np.linalg.norm(deep_sig) > 1e-6:
            self.deep_gallery.append(deep_sig.copy())
            self.deep_ema = deep_sig.copy()

        cx = float(0.5 * (head_box[0] + head_box[2]))
        cy = float(0.5 * (head_box[1] + head_box[3]))
        self.trajectory: deque[Tuple[float, float]] = deque(maxlen=90)
        self.trajectory.append((cx, cy))

    def update(
        self,
        head_box: np.ndarray,
        helmet_class: str,
        strip_sig: np.ndarray,
        torso_roi: Tuple[int, int, int, int],
        frame_idx: int,
        now_ts: float,
        deep_sig: Optional[np.ndarray] = None,
        min_confirm_hits: int = 6,
        min_confirm_frames: int = 24,
    ):
        self.last_box = np.asarray(head_box[:4], dtype=np.float32).copy()
        self.last_torso_roi = torso_roi
        self.last_seen_frame = frame_idx
        self.last_seen_ts = now_ts
        self.total_frames_seen += 1
        self.consecutive_hits += 1
        if self.consecutive_hits >= min_confirm_hits or self.total_frames_seen >= min_confirm_frames:
            self.is_confirmed = True
        self.is_active = True

        hw = max(2.0, float(head_box[2] - head_box[0]))
        tw = max(2.0, float(torso_roi[2] - torso_roi[0]))
        self.shoulder_head_ratio = 0.8 * self.shoulder_head_ratio + 0.2 * (tw / hw)

        self.current_helmet_color = helmet_class
        if helmet_class and helmet_class != "none":
            self.helmet_color_votes[helmet_class] = self.helmet_color_votes.get(helmet_class, 0) + 1
            if helmet_class == self._color_streak_cls:
                self._color_streak_len += 1
            else:
                self._color_streak_cls = helmet_class
                self._color_streak_len = 1
            if self._color_streak_len >= self.min_history_votes or (
                self.last_known_helmet_color is None
                and self.helmet_color_votes[helmet_class] >= self.min_history_votes
            ):
                self.last_known_helmet_color = helmet_class
            self.status = "COMPLIANT"
            self.violations = []
        else:
            self._color_streak_cls = None
            self._color_streak_len = 0
            self.status = "VIOLATION"
            self.violations = ["MISSING_SAFETY_HELMET"]

        if strip_sig is not None and np.linalg.norm(strip_sig) > 1e-6:
            if self.strip_ema is None:
                self.strip_ema = strip_sig.copy()
                self.strip_gallery.append(strip_sig.copy())
            else:
                blended = 0.85 * self.strip_ema + 0.15 * strip_sig
                n = float(np.linalg.norm(blended))
                if n > 1e-6:
                    self.strip_ema = blended / n
                if self.max_strip_sim(strip_sig) < 0.90:
                    self.strip_gallery.append(strip_sig.copy())

        if deep_sig is not None and np.linalg.norm(deep_sig) > 1e-6:
            self.last_deep_frame = frame_idx
            if self.deep_ema is None:
                self.deep_ema = deep_sig.copy()
                self.deep_gallery.append(deep_sig.copy())
            else:
                blended = 0.85 * self.deep_ema + 0.15 * deep_sig
                n = float(np.linalg.norm(blended))
                if n > 1e-6:
                    self.deep_ema = blended / n
                if self.max_deep_sim(deep_sig) < 0.85:
                    self.deep_gallery.append(deep_sig.copy())

        cx = float(0.5 * (head_box[0] + head_box[2]))
        cy = float(0.5 * (head_box[1] + head_box[3]))
        self.trajectory.append((cx, cy))

    def max_strip_sim(self, cand_strip: Optional[np.ndarray]) -> float:
        color_sim, tex_sim, has_tex = self.max_subspace_sims(cand_strip)
        if not has_tex:
            return color_sim
        return 0.55 * color_sim + 0.45 * tex_sim

    def max_subspace_sims(self, cand_strip: Optional[np.ndarray]) -> Tuple[float, float, bool]:
        """Returns (max_color_sim, max_texture_sim, both_have_texture)."""
        if cand_strip is None or cand_strip.size < 474 or np.linalg.norm(cand_strip[:474]) < 1e-6:
            return 0.0, 0.0, False
        cand_col = cand_strip[:474]
        cand_tex = cand_strip[474:] if cand_strip.size > 474 else None
        cand_has_tex = cand_tex is not None and float(np.linalg.norm(cand_tex)) > 0.1

        color_sims = [cosine_sim(v[:474], cand_col) for v in self.strip_gallery if v.size >= 474]
        if self.strip_ema is not None and self.strip_ema.size >= 474:
            color_sims.append(cosine_sim(self.strip_ema[:474], cand_col))
        best_col = max(color_sims) if color_sims else 0.0

        best_tex = 0.0
        prof_has_tex = False
        if cand_has_tex:
            tex_sims = []
            for v in list(self.strip_gallery) + ([self.strip_ema] if self.strip_ema is not None else []):
                if v.size > 474 and float(np.linalg.norm(v[474:])) > 0.1:
                    prof_has_tex = True
                    tex_sims.append(cosine_sim(v[474:], cand_tex))
            if tex_sims:
                best_tex = max(tex_sims)
        return best_col, best_tex, (cand_has_tex and prof_has_tex)

    def max_deep_sim(self, cand_deep: Optional[np.ndarray]) -> float:
        if cand_deep is None or np.linalg.norm(cand_deep) < 1e-6:
            return 0.0
        sims = [cosine_sim(v, cand_deep) for v in self.deep_gallery]
        if self.deep_ema is not None:
            sims.append(cosine_sim(self.deep_ema, cand_deep))
        return max(sims) if sims else 0.0


class UpperTorsoReIDGallery:
    """
    Persistent Re-ID gallery for GDUT 5-class head tracking.
    Uses Hungarian optimal assignment (`linear_sum_assignment`), coexistence exclusion,
    ambiguity margin gating, anatomical wall-suppressed ROI, and multi-layer MobileNetV3 features.
    """

    def __init__(
        self,
        ttl_seconds: float = 600.0,
        sim_thresh: float = 0.70,
        use_deep: bool = True,
        deep_refresh_interval: int = 30,
        min_confirm_hits: int = 6,
        min_confirm_frames: int = 24,
        device: Optional[str] = None,
    ):
        self.ttl_seconds = float(ttl_seconds)
        self.sim_thresh = float(sim_thresh)
        self.deep_refresh_interval = max(1, int(deep_refresh_interval))
        self.min_confirm_hits = max(1, int(min_confirm_hits))
        self.min_confirm_frames = max(1, int(min_confirm_frames))
        self.deep_extractor = MobileNetV3SmallExtractor(enabled=use_deep, device=device)

        self.profiles: Dict[int, UpperTorsoIdentityProfile] = {}
        self._tentative_tracks: Dict[int, UpperTorsoIdentityProfile] = {}
        self.raw_to_canonical: Dict[int, int] = {}
        self.raw_conflict_streak: Dict[int, int] = {}
        self.hijack_excluded_ids: Dict[int, Tuple[int, int]] = {}
        self.next_id: int = 1

    def prewarm(self) -> bool:
        return self.deep_extractor.prewarm()

    def reset(self):
        self.profiles.clear()
        self._tentative_tracks.clear()
        self.raw_to_canonical.clear()
        self.raw_conflict_streak.clear()
        self.hijack_excluded_ids.clear()
        self.next_id = 1

    def on_video_loop(self):
        """Preserves persistent identity profiles on video loop while clearing raw tracker IDs."""
        self._tentative_tracks.clear()
        self.raw_to_canonical.clear()
        self.raw_conflict_streak.clear()
        self.hijack_excluded_ids.clear()
        for p in self.profiles.values():
            p.is_active = False
            p.consecutive_hits = 0

    def set_label(self, canonical_id: int, label: str) -> bool:
        p = self.profiles.get(int(canonical_id))
        if p is None:
            return False
        cleaned = str(label).strip()[:40]
        p.custom_label = cleaned if cleaned else None
        return True

    def enroll_reference(
        self,
        canonical_id: int,
        frame_bgr: np.ndarray,
        head_box: Optional[np.ndarray | List[float]] = None,
        label: Optional[str] = None,
    ) -> bool:
        """Captures an explicit multi-angle reference view for a tracked personnel ID during demo setup."""
        p = self.profiles.get(int(canonical_id))
        if p is None or frame_bgr is None or frame_bgr.size == 0:
            return False
        hbox = np.asarray(head_box[:4], dtype=np.float32) if head_box is not None else p.last_box
        roi = project_upper_torso_roi(hbox, frame_bgr.shape)
        strip_sig = compute_three_strip_torso_signature(frame_bgr, roi)
        deep_sig = self.deep_extractor.extract(frame_bgr, roi) if self.deep_extractor.available else None
        if np.linalg.norm(strip_sig) > 1e-6:
            p.strip_gallery.append(strip_sig)
        if deep_sig is not None and np.linalg.norm(deep_sig) > 1e-6:
            p.deep_gallery.append(deep_sig)
        if label is not None and str(label).strip():
            p.custom_label = str(label).strip()[:40]
        return True

    def _compute_appear_subscores(
        self,
        profile: UpperTorsoIdentityProfile,
        strip_sig: np.ndarray,
        deep_sig: Optional[np.ndarray],
        skip_hard_veto: bool = False,
        dt_seconds: Optional[float] = None,
    ) -> Tuple[float, bool]:
        """Returns (appearance_similarity, vetoed_by_structural_mismatch)."""
        sim_color, sim_tex, has_tex = profile.max_subspace_sims(strip_sig)
        sim_strip = (0.52 * sim_color + 0.48 * sim_tex) if has_tex else sim_color

        if deep_sig is not None and profile.deep_ema is not None:
            sim_deep = profile.max_deep_sim(deep_sig)
            if has_tex:
                base_sim = 0.50 * sim_deep + 0.30 * sim_color + 0.20 * sim_tex
            else:
                base_sim = 0.52 * sim_deep + 0.48 * sim_color
            if not skip_hard_veto:
                short_term_reentry = (dt_seconds is not None and dt_seconds <= 8.0)
                strong_pattern_color = (sim_color >= 0.78 and (not has_tex or sim_tex >= 0.65))
                deep_veto_floor = 0.50 if (strong_pattern_color or (short_term_reentry and sim_color >= 0.78)) else 0.62
                if sim_deep < deep_veto_floor:
                    return float(min(base_sim, sim_deep)), True
            return float(base_sim), False

        if not skip_hard_veto and has_tex and sim_tex < 0.58:
            return float(min(sim_strip, 0.60 * sim_tex + 0.25 * sim_color)), True
        return float(sim_strip), False

    def _compute_candidate_score(
        self,
        profile: UpperTorsoIdentityProfile,
        strip_sig: np.ndarray,
        deep_sig: Optional[np.ndarray],
        cls_norm: str,
        cand_ratio: float,
        skip_hard_veto: bool = False,
        dt_seconds: Optional[float] = None,
    ) -> float:
        base_sim, vetoed = self._compute_appear_subscores(
            profile, strip_sig, deep_sig, skip_hard_veto=skip_hard_veto, dt_seconds=dt_seconds
        )
        if vetoed:
            return float(base_sim)

        # Geometric shoulder/head proportion check
        ratio_diff = abs(cand_ratio - profile.shoulder_head_ratio)
        ratio_adj = 0.02 if ratio_diff <= 0.18 else (-0.04 if ratio_diff > 0.45 else 0.0)

        # Helmet role continuity:
        # - If bare-headed ("none"), apply 0 penalty so helmet removal preserves canonical ID.
        # - If two distinct positive helmet colors mismatch, strongly penalize cross-swap.
        color_adj = 0.0
        if cls_norm != "none" and profile.last_known_helmet_color and profile.last_known_helmet_color != "none":
            if cls_norm == profile.last_known_helmet_color:
                color_adj = 0.04
            else:
                color_adj = -0.24

        return float(base_sim + ratio_adj + color_adj)

    def associate_batch(
        self,
        items: List[Tuple[int, np.ndarray | List[float], str]],
        frame_bgr: np.ndarray,
        frame_idx: int,
        now_ts: Optional[float] = None,
    ) -> List[UpperTorsoIdentityProfile]:
        """
        Globally associates a batch of (raw_track_id, head_box, helmet_class) tuples via:
        1. Direct raw-to-canonical binding with anti-hijack clothing divergence guard.
        2. Short-Term Spatio-Temporal Re-Link (<=3.0s, <=0.18 diag, no deep hard veto)
           and Long-Term Hungarian matching against inactive profiles within TTL,
           enforcing coexistence exclusion & ambiguity margin guard.
        3. Novel identity minting for genuinely new arrivals.
        """
        if not items:
            return []

        ts = float(now_ts if now_ts is not None else time.time())
        fidx_int = int(frame_idx)
        self.hijack_excluded_ids = {
            k: v for k, v in self.hijack_excluded_ids.items() if fidx_int <= int(v[1])
        }
        n_items = len(items)
        results: List[Optional[UpperTorsoIdentityProfile]] = [None] * n_items
        unmatched_indices: List[int] = []
        claimed_cids: Set[int] = set()
        claimed_boxes: Dict[int, Tuple[float, float]] = {}

        prepared: List[Tuple[int, np.ndarray, str, Tuple[int, int, int, int], np.ndarray, float]] = []
        for raw_track_id, head_box, helmet_class in items:
            cls_norm = str(helmet_class or "none").strip().lower()
            hbox = np.asarray(head_box[:4], dtype=np.float32)
            roi = project_upper_torso_roi(hbox, frame_bgr.shape)
            strip_sig = compute_three_strip_torso_signature(frame_bgr, roi)
            hw = max(2.0, float(hbox[2] - hbox[0]))
            tw = max(2.0, float(roi[2] - roi[0]))
            cand_ratio = tw / hw
            prepared.append((raw_track_id, hbox, cls_norm, roi, strip_sig, cand_ratio))

        # 1. Existing ByteTrack raw_id bindings with clothing anti-hijack guard + tentative continuation
        promoted_candidates: Dict[int, UpperTorsoIdentityProfile] = {}
        for idx, (raw_track_id, hbox, cls_norm, roi, strip_sig, _) in enumerate(prepared):
            if raw_track_id in self.raw_to_canonical:
                cid = self.raw_to_canonical[raw_track_id]
                p = self.profiles.get(cid)
                if p is not None and cid not in claimed_cids:
                    sim_strip = p.max_strip_sim(strip_sig)
                    should_break = False
                    if p.is_confirmed and np.linalg.norm(strip_sig) > 1e-6 and p.strip_ema is not None:
                        if sim_strip < 0.22:
                            should_break = True
                        elif sim_strip < 0.34:
                            self.raw_conflict_streak[raw_track_id] = self.raw_conflict_streak.get(raw_track_id, 0) + 1
                            if self.raw_conflict_streak[raw_track_id] >= 4:
                                should_break = True
                        else:
                            self.raw_conflict_streak[raw_track_id] = 0

                    if not should_break:
                        deep_sig = None
                        if self.deep_extractor.available and (
                            p.deep_ema is None or (frame_idx - p.last_deep_frame) >= self.deep_refresh_interval
                        ):
                            deep_sig = self.deep_extractor.extract(frame_bgr, roi)
                        p.update(
                            hbox,
                            cls_norm,
                            strip_sig,
                            roi,
                            frame_idx,
                            ts,
                            deep_sig=deep_sig,
                            min_confirm_hits=self.min_confirm_hits,
                            min_confirm_frames=self.min_confirm_frames,
                        )
                        results[idx] = p
                        claimed_cids.add(cid)
                        claimed_boxes[cid] = (0.5 * float(hbox[0] + hbox[2]), 0.5 * float(hbox[1] + hbox[3]))
                        continue
                    else:
                        self.raw_to_canonical.pop(raw_track_id, None)
                        self.raw_conflict_streak.pop(raw_track_id, None)
                        self.hijack_excluded_ids[raw_track_id] = (cid, fidx_int + 15)
            elif raw_track_id in self._tentative_tracks:
                tent = self._tentative_tracks[raw_track_id]
                tent.update(
                    hbox,
                    cls_norm,
                    strip_sig,
                    roi,
                    frame_idx,
                    ts,
                    deep_sig=None,
                    min_confirm_hits=self.min_confirm_hits,
                    min_confirm_frames=self.min_confirm_frames,
                )
                if tent.consecutive_hits < self.min_confirm_hits:
                    tent.is_confirmed = False
                    results[idx] = tent
                    continue
                # Reached min_confirm_hits: queue for Step 2 re-check before minting canonical ID
                promoted_candidates[idx] = tent
            unmatched_indices.append(idx)

        # 2. Optimal Hungarian Matching against eligible INACTIVE confirmed profiles
        if unmatched_indices:
            inactive_candidates = sorted(
                [
                    p for p in self.profiles.values()
                    if (not p.is_active)
                    and (p.worker_id not in claimed_cids)
                    and (ts - p.last_seen_ts) <= self.ttl_seconds
                ],
                key=lambda prof: (not prof.is_confirmed, -prof.last_seen_ts),
            )

            diag = float(np.hypot(frame_bgr.shape[1], frame_bgr.shape[0])) + 1e-5
            cand_deeps: Dict[int, Optional[np.ndarray]] = {}
            if inactive_candidates and self.deep_extractor.available:
                for orig_idx in unmatched_indices:
                    raw_id, hbox, _, roi, strip_sig, _ = prepared[orig_idx]
                    is_promoted = orig_idx in promoted_candidates
                    if is_promoted:
                        cand_deeps[orig_idx] = self.deep_extractor.extract(frame_bgr, roi)
                        continue
                    c_cx = 0.5 * float(hbox[0] + hbox[2])
                    c_cy = 0.5 * float(hbox[1] + hbox[3])
                    excl_entry = self.hijack_excluded_ids.get(raw_id)
                    excluded_cid = (
                        excl_entry[0]
                        if (excl_entry is not None and fidx_int <= int(excl_entry[1]))
                        else None
                    )
                    coarse_eligible = False
                    for prof in inactive_candidates:
                        if excluded_cid is not None and prof.worker_id == excluded_cid:
                            continue
                        p_cx = 0.5 * float(prof.last_box[0] + prof.last_box[2])
                        p_cy = 0.5 * float(prof.last_box[1] + prof.last_box[3])
                        dist_norm = float(np.hypot(p_cx - c_cx, p_cy - c_cy)) / diag
                        if dist_norm <= 0.25 or prof.max_strip_sim(strip_sig) >= 0.45:
                            coarse_eligible = True
                            break
                    cand_deeps[orig_idx] = (
                        self.deep_extractor.extract(frame_bgr, roi) if coarse_eligible else None
                    )

            if inactive_candidates:
                num_u = len(unmatched_indices)
                num_p = len(inactive_candidates)
                score_mat = np.full((num_u, num_p), -1.0, dtype=np.float32)
                thresh_mat = np.full((num_u, num_p), self.sim_thresh, dtype=np.float32)
                cost_mat = np.full((num_u, num_p), 2.0, dtype=np.float32)

                for r_i, orig_idx in enumerate(unmatched_indices):
                    raw_id, hbox, cls_norm, _, strip_sig, cand_ratio = prepared[orig_idx]
                    tent = promoted_candidates.get(orig_idx)
                    match_strip = (
                        tent.strip_ema
                        if (tent is not None and tent.strip_ema is not None and np.linalg.norm(tent.strip_ema) > 1e-6)
                        else strip_sig
                    )
                    match_ratio = tent.shoulder_head_ratio if tent is not None else cand_ratio
                    match_cls = (
                        tent.last_known_helmet_color
                        if (tent is not None and tent.last_known_helmet_color)
                        else cls_norm
                    )
                    excl_entry = self.hijack_excluded_ids.get(raw_id)
                    excluded_cid = (
                        excl_entry[0]
                        if (excl_entry is not None and fidx_int <= int(excl_entry[1]))
                        else None
                    )
                    c_deep = cand_deeps.get(orig_idx)
                    c_cx = 0.5 * float(hbox[0] + hbox[2])
                    c_cy = 0.5 * float(hbox[1] + hbox[3])

                    for c_j, prof in enumerate(inactive_candidates):
                        if excluded_cid is not None and prof.worker_id == excluded_cid:
                            continue
                        if prof.worker_id in claimed_cids:
                            continue

                        dt = max(0.0, ts - prof.last_seen_ts)
                        p_cx = 0.5 * float(prof.last_box[0] + prof.last_box[2])
                        p_cy = 0.5 * float(prof.last_box[1] + prof.last_box[3])
                        dist_norm = float(np.hypot(p_cx - c_cx, p_cy - c_cy)) / diag

                        has_pos_color_conflict = (
                            match_cls != "none"
                            and prof.last_known_helmet_color is not None
                            and prof.last_known_helmet_color != "none"
                            and match_cls != prof.last_known_helmet_color
                        )

                        if dt <= 3.0 and dist_norm <= 0.18 and not has_pos_color_conflict:
                            appear_sim, _ = self._compute_appear_subscores(
                                prof, match_strip, c_deep, skip_hard_veto=True
                            )
                            spatial_sim = float(np.exp(-dist_norm / 0.12))
                            color_adj = 0.04 if (match_cls != "none" and match_cls == prof.last_known_helmet_color) else 0.0
                            score = float(0.45 * spatial_sim + 0.55 * appear_sim + color_adj)
                            pair_thresh = 0.52
                        else:
                            score = self._compute_candidate_score(
                                prof, match_strip, c_deep, match_cls, match_ratio, skip_hard_veto=False, dt_seconds=dt
                            )
                            pair_thresh = self.sim_thresh

                        score_mat[r_i, c_j] = score
                        thresh_mat[r_i, c_j] = pair_thresh
                        cost_mat[r_i, c_j] = -(score - pair_thresh)

                row_ind, col_ind = linear_sum_assignment(cost_mat)
                still_unmatched: List[int] = list(unmatched_indices)

                for r_i, c_j in zip(row_ind, col_ind):
                    orig_idx = unmatched_indices[r_i]
                    best_score = float(score_mat[r_i, c_j])
                    pair_thresh = float(thresh_mat[r_i, c_j])
                    if best_score < pair_thresh:
                        continue

                    prof = inactive_candidates[c_j]
                    # Ambiguity Margin Guard: inspect second-best competing candidate
                    other_scores = [float(score_mat[r_i, k]) for k in range(num_p) if k != c_j]
                    second_best = max(other_scores) if other_scores else -1.0
                    raw_id, hbox, cls_norm, roi, strip_sig, _ = prepared[orig_idx]

                    if second_best >= self.sim_thresh and (best_score - second_best) < 0.06:
                        p_cx = 0.5 * (prof.last_box[0] + prof.last_box[2])
                        p_cy = 0.5 * (prof.last_box[1] + prof.last_box[3])
                        c_cx = 0.5 * (hbox[0] + hbox[2])
                        c_cy = 0.5 * (hbox[1] + hbox[3])
                        dist_norm = float(np.hypot(p_cx - c_cx, p_cy - c_cy)) / diag
                        if (best_score - second_best) < 0.03 and dist_norm > 0.35:
                            continue

                    tent = promoted_candidates.pop(orig_idx, None)
                    if tent is not None:
                        prof.total_frames_seen += max(0, tent.total_frames_seen - 1)
                        self._tentative_tracks.pop(raw_id, None)
                    prof.update(
                        hbox,
                        cls_norm,
                        strip_sig,
                        roi,
                        frame_idx,
                        ts,
                        deep_sig=cand_deeps.get(orig_idx),
                        min_confirm_hits=self.min_confirm_hits,
                        min_confirm_frames=self.min_confirm_frames,
                    )
                    self.raw_to_canonical[raw_id] = prof.worker_id
                    claimed_cids.add(prof.worker_id)
                    claimed_boxes[prof.worker_id] = (0.5 * float(hbox[0] + hbox[2]), 0.5 * float(hbox[1] + hbox[3]))
                    results[orig_idx] = prof
                    if orig_idx in still_unmatched:
                        still_unmatched.remove(orig_idx)

                unmatched_indices = still_unmatched

            # 3. Confirm-Before-Mint: tentative tracks accumulate hits WITHOUT incrementing next_id;
            #    only tracks promoted at >= min_confirm_hits mint a canonical ID.
            for orig_idx in unmatched_indices:
                raw_id, hbox, cls_norm, roi, strip_sig, _ = prepared[orig_idx]
                tent = promoted_candidates.get(orig_idx)
                if tent is None and self.min_confirm_hits > 1:
                    temp_id = -(abs(int(raw_id)) + 1)
                    tent_prof = UpperTorsoIdentityProfile(
                        worker_id=temp_id,
                        head_box=hbox,
                        helmet_class=cls_norm,
                        strip_sig=strip_sig,
                        deep_sig=None,
                        torso_roi=roi,
                        frame_idx=frame_idx,
                        now_ts=ts,
                    )
                    tent_prof.is_confirmed = False
                    self._tentative_tracks[raw_id] = tent_prof
                    results[orig_idx] = tent_prof
                    continue

                cid = self.next_id
                self.next_id += 1
                c_deep = cand_deeps.get(orig_idx)
                if c_deep is None and self.deep_extractor.available:
                    c_deep = self.deep_extractor.extract(frame_bgr, roi)

                if tent is not None:
                    self._tentative_tracks.pop(raw_id, None)
                    tent.worker_id = cid
                    tent.is_confirmed = True
                    if c_deep is not None and np.linalg.norm(c_deep) > 1e-6:
                        tent.last_deep_frame = frame_idx
                        tent.deep_ema = c_deep.copy()
                        tent.deep_gallery.append(c_deep.copy())
                    new_profile = tent
                else:
                    new_profile = UpperTorsoIdentityProfile(
                        worker_id=cid,
                        head_box=hbox,
                        helmet_class=cls_norm,
                        strip_sig=strip_sig,
                        deep_sig=c_deep,
                        torso_roi=roi,
                        frame_idx=frame_idx,
                        now_ts=ts,
                    )
                    new_profile.is_confirmed = True

                self.profiles[cid] = new_profile
                self.raw_to_canonical[raw_id] = cid
                claimed_cids.add(cid)
                claimed_boxes[cid] = (0.5 * float(hbox[0] + hbox[2]), 0.5 * float(hbox[1] + hbox[3]))
                results[orig_idx] = new_profile

        return [r for r in results if r is not None]

    def consolidate_profiles(self) -> int:
        """
        Merges fragmented profiles that never coexisted in the same frame and share high
        appearance + helmet similarity (>= 0.78). Strictly enforces `coexisted_ids` exclusion.
        """
        merged_count = 0
        cids = sorted(self.profiles.keys())
        dropped: Set[int] = set()

        for i, wid_a in enumerate(cids):
            if wid_a in dropped:
                continue
            pa = self.profiles.get(wid_a)
            if pa is None or not pa.is_confirmed:
                continue

            for wid_b in cids[i + 1 :]:
                if wid_b in dropped:
                    continue
                pb = self.profiles.get(wid_b)
                if pb is None or not pb.is_confirmed:
                    continue
                if pa.is_active and pb.is_active:
                    continue
                # Hard exclusion: two identities seen together in the same frame MUST NEVER merge
                if wid_b in pa.coexisted_ids or wid_a in pb.coexisted_ids:
                    continue

                if (
                    pa.last_known_helmet_color
                    and pb.last_known_helmet_color
                    and pa.last_known_helmet_color != "none"
                    and pb.last_known_helmet_color != "none"
                    and pa.last_known_helmet_color != pb.last_known_helmet_color
                ):
                    continue

                if pb.strip_ema is None or pa.strip_ema is None:
                    continue
                strip_sim = pa.max_strip_sim(pb.strip_ema)
                if strip_sim < 0.78:
                    continue
                if pa.deep_ema is not None and pb.deep_ema is not None:
                    deep_sim = pa.max_deep_sim(pb.deep_ema)
                    if deep_sim < 0.78:
                        continue

                # Consolidate pb into older wid_a
                pa.total_frames_seen += pb.total_frames_seen
                for v in pb.strip_gallery:
                    if pa.max_strip_sim(v) < 0.94:
                        pa.strip_gallery.append(v.copy())
                for dv in pb.deep_gallery:
                    if pa.max_deep_sim(dv) < 0.94:
                        pa.deep_gallery.append(dv.copy())
                pa.coexisted_ids.update(pb.coexisted_ids)
                pa.coexisted_ids.discard(pa.worker_id)
                if not pa.custom_label and pb.custom_label:
                    pa.custom_label = pb.custom_label
                if pb.last_seen_frame > pa.last_seen_frame:
                    pa.last_box = pb.last_box.copy()
                    pa.last_torso_roi = pb.last_torso_roi
                    pa.last_seen_frame = pb.last_seen_frame
                    pa.last_seen_ts = pb.last_seen_ts
                    pa.is_active = pb.is_active
                    pa.current_helmet_color = pb.current_helmet_color
                    if pb.last_known_helmet_color and not pa.last_known_helmet_color:
                        pa.last_known_helmet_color = pb.last_known_helmet_color

                for raw_id, mapped_cid in list(self.raw_to_canonical.items()):
                    if mapped_cid == wid_b:
                        self.raw_to_canonical[raw_id] = wid_a
                self.profiles.pop(wid_b, None)
                dropped.add(wid_b)
                merged_count += 1

        return merged_count

    def associate(
        self,
        raw_track_id: int,
        head_box: np.ndarray | List[float],
        helmet_class: str,
        frame_bgr: np.ndarray,
        frame_idx: int,
        now_ts: Optional[float] = None,
    ) -> UpperTorsoIdentityProfile:
        """Single-detection compatibility wrapper delegating to associate_batch()."""
        return self.associate_batch([(raw_track_id, head_box, helmet_class)], frame_bgr, frame_idx, now_ts=now_ts)[0]

    def update_frame_end(
        self,
        active_canonical_ids: List[int],
        frame_idx: int,
        now_ts: Optional[float] = None,
    ):
        """Updates active states, coexistence exclusion edges, prunes transient tentative tracks, and expires stale profiles."""
        ts = float(now_ts if now_ts is not None else time.time())
        active_set = {int(wid) for wid in active_canonical_ids if isinstance(wid, (int, np.integer)) and int(wid) > 0}

        if len(active_set) >= 2:
            for wid in active_set:
                p = self.profiles.get(wid)
                if p is not None:
                    p.coexisted_ids.update(active_set - {wid})

        for wid, p in self.profiles.items():
            if wid in active_set:
                p.is_active = True
                p.last_seen_frame = frame_idx
                p.last_seen_ts = ts
            else:
                p.is_active = False
                p.consecutive_hits = 0

        # Prune stale tentative tracks (> 2 missed frames) without ever touching self.next_id
        dead_tent = [
            rid for rid, tp in self._tentative_tracks.items()
            if (frame_idx - tp.last_seen_frame) > 2
        ]
        for rid in dead_tent:
            self._tentative_tracks.pop(rid, None)

        if frame_idx % 10 == 0:
            for raw_id, cid in list(self.raw_to_canonical.items()):
                prof = self.profiles.get(cid)
                if prof is None or (not prof.is_active and (frame_idx - prof.last_seen_frame) > 60):
                    self.raw_to_canonical.pop(raw_id, None)
                    self.raw_conflict_streak.pop(raw_id, None)
                    self.hijack_excluded_ids.pop(raw_id, None)

            to_delete = [
                wid for wid, p in self.profiles.items()
                if (
                    (not p.is_active and not p.is_confirmed and (frame_idx - p.last_seen_frame) > 30)
                    or (not p.is_active and (ts - p.last_seen_ts) > self.ttl_seconds)
                )
            ]
            for wid in to_delete:
                self.profiles.pop(wid, None)

            if not self.profiles:
                self.next_id = 1

        if frame_idx % 30 == 0 and len(self.profiles) >= 2:
            self.consolidate_profiles()

    def get_registered_workers(
        self,
        role_policy: Optional[Dict[str, Dict[str, Any]]] = None,
        filter_ghosts: bool = True,
        now_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Returns active + confirmed gallery-remembered personnel enriched with Role-by-Color hierarchy."""
        policy = role_policy or DEFAULT_ROLE_HIERARCHY
        ts = float(now_ts if now_ts is not None else time.time())
        latest_frame = max([p.last_seen_frame for p in self.profiles.values()], default=0)

        workers: List[Dict[str, Any]] = []
        for wid in sorted(self.profiles.keys()):
            p = self.profiles[wid]
            if filter_ghosts and not (p.is_confirmed or (p.is_active and p.consecutive_hits >= 6)):
                continue

            cur_cls = p.current_helmet_color or "none"
            hist_col = p.last_known_helmet_color if p.last_known_helmet_color != "none" else None
            cur_role = policy.get(cur_cls, policy.get("none", {}))
            hist_role = policy.get(hist_col, cur_role) if hist_col else cur_role
            is_viol_state = (p.status == "VIOLATION" and cur_cls == "none")

            frames_since = max(0, latest_frame - p.last_seen_frame) if not p.is_active else 0
            secs_since = round(max(0.0, ts - p.last_seen_ts), 1) if not p.is_active else 0.0

            workers.append({
                "track_id": p.worker_id,
                "custom_label": p.custom_label,
                "helmet_color": cur_cls if cur_cls != "none" else None,
                "current_class": cur_cls,
                "historical_helmet_color": hist_col,
                "role_title": cur_role.get("role_title", "Unclassified"),
                "role_short_code": cur_role.get("short_code", "N/A"),
                "role_hex_color": cur_role.get("hex_color", "#64748B"),
                "historical_role_title": hist_role.get("role_title", "Unclassified"),
                "historical_role_short_code": hist_role.get("short_code", "N/A"),
                "helmet_removed": (is_viol_state and hist_col is not None),
                "status": p.status,
                "is_active": p.is_active,
                "is_confirmed": p.is_confirmed,
                "first_frame": p.first_frame,
                "last_seen_frame": p.last_seen_frame,
                "total_frames_seen": p.total_frames_seen,
                "frames_since_seen": frames_since,
                "seconds_since_seen": secs_since,
                "violations": list(p.violations),
                "trajectory_len": len(p.trajectory),
                "torso_roi": [int(v) for v in p.last_torso_roi],
                "bbox": [round(float(v), 1) for v in p.last_box],
            })
        return workers
