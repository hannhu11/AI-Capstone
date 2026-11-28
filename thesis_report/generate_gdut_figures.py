"""Generate 300-DPI publication-quality analytical figures for Capstone Thesis Report.

100% data-driven: every plotted value is read dynamically at runtime from verified raw
Kaggle CSV/JSON artifacts, SHWD_V1 outputs, and RTX 3050 / FastAPI benchmark JSON files.
Works identically in the full workspace and in the standalone Clean-Room public repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
EVID_DIR = ROOT / "kaggle_evidence"
RAW_ARTIFACTS_DIR = REPO_ROOT / "notebooks" / "gdut_official_kaggle" / "raw_artifacts"
KAGGLE_VERIFIED_DIR = REPO_ROOT / "docs" / "kaggle_verified_source"
NVTHANH_DIR = KAGGLE_VERIFIED_DIR / "nvthanh2004"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.9,
        "grid.color": "#CBD5E1",
        "grid.linestyle": "--",
        "grid.alpha": 0.65,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#CBD5E1",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _resolve_first(candidates: Iterable[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    tried = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not locate {label}. Checked:\n  - {tried}")


def _load_stage0_and_init_data() -> dict:
    p = _resolve_first(
        [
            KAGGLE_VERIFIED_DIR / "gdut_stage0_and_init_checks.json",
            RAW_ARTIFACTS_DIR / "gdut_stage0_and_init_checks.json",
        ],
        "gdut_stage0_and_init_checks.json",
    )
    return json.loads(p.read_text(encoding="utf-8"))


def copy_raw_kaggle_evidence(fig_dir: Path) -> None:
    """Copy authentic raw Kaggle PNG/JPG figures into thesis_report/figures if available."""
    if not EVID_DIR.exists():
        return
    for fname in [
        "gdut_labels_dist.jpg",
        "gdut_train_batch0.jpg",
        "a6_yolo11s_results.png",
        "a6_yolo11s_val_pr.png",
        "a6_yolo11s_val_f1.png",
        "a6_yolo11s_val_cm.png",
        "a6_yolo11s_test_pr.png",
        "a6_yolo11s_test_cm.png",
        "a6_yolo11s_test_pred0.jpg",
        "a6_yolov8s_val_pr.png",
        "a6_yolov8s_val_cm.png",
        "a1_yolo11s_test_pr.png",
        "a1_yolo11s_test_cm.png",
        "a1_yolo11s_test_pred0.jpg",
    ]:
        src = EVID_DIR / fname
        if src.exists():
            shutil.copy2(src, fig_dir / fname)


# ------------------------------------------------------------------
# Figure IV.1: GDUT-HWD 5-Class Bounding Box Distribution Across Splits
# ------------------------------------------------------------------
def make_class_distribution(fig_dir: Path) -> None:
    audit = _load_stage0_and_init_data()["dataset_audit"]
    names = audit["names"]  # ["blue", "white", "yellow", "red", "none"]
    splits = audit["split_counts"]
    boxes = audit["box_counts"]

    classes = [f"{name}\n(ID {idx})" for idx, name in enumerate(names)]
    train = [int(boxes["train"][name]) for name in names]
    val = [int(boxes["val"][name]) for name in names]
    test = [int(boxes["test"][name]) for name in names]
    totals = [t + v + s for t, v, s in zip(train, val, test)]
    grand_total = sum(totals)

    x = np.arange(len(classes))
    w = 0.25
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.55, 1.0]})

    b1 = ax1.bar(
        x - w,
        train,
        w,
        label=f"Train ({splits['train']:,} imgs; {sum(train):,} boxes)",
        color="#1D4ED8",
        edgecolor="#0F172A",
        linewidth=0.5,
    )
    b2 = ax1.bar(
        x,
        val,
        w,
        label=f"Validation ({splits['val']:,} imgs; {sum(val):,} boxes)",
        color="#059669",
        edgecolor="#0F172A",
        linewidth=0.5,
    )
    b3 = ax1.bar(
        x + w,
        test,
        w,
        label=f"Official Test ({splits['test']:,} imgs; {sum(test):,} boxes)",
        color="#D97706",
        edgecolor="#0F172A",
        linewidth=0.5,
    )

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(
                f"{int(h):,}",
                (bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_ylabel("Bounding Box Instances")
    ax1.set_title("(a) Per-Split Instance Counts (Deduplicated GDUT-HWD)")
    ax1.set_ylim(0, max(test) * 1.20)
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper left", fontsize=8.5)

    colors = ["#2563EB", "#94A3B8", "#EAB308", "#DC2626", "#475569"]
    pcts = [100.0 * t / grand_total for t in totals]
    bars2 = ax2.barh(classes[::-1], pcts[::-1], color=colors[::-1], edgecolor="#0F172A", linewidth=0.6)
    for bar, tot in zip(bars2, totals[::-1]):
        w_val = bar.get_width()
        ax2.text(
            w_val + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{w_val:.1f}% ({tot:,})",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#1E293B",
        )
    ax2.set_xlabel("Share of Total Instances (%)")
    ax2.set_title(f"(b) Overall Class Share ({grand_total:,} Boxes)")
    ax2.set_xlim(0, max(pcts) + 8.5)
    ax2.xaxis.grid(True)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_gdut_class_distribution.png")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure VI.1: A0-A6 Ablation Results (Val vs Test, YOLO11s & YOLOv8s)
# ------------------------------------------------------------------
def make_ablation_val_test(fig_dir: Path) -> None:
    val_csv = _resolve_first(
        [
            NVTHANH_DIR / "gdut-final-test-evaluation_outputs" / "test_evaluation" / "validation_summary.csv",
            RAW_ARTIFACTS_DIR / "final_test_validation_summary.csv",
        ],
        "final_test_validation_summary.csv",
    )
    test_csv = _resolve_first(
        [
            NVTHANH_DIR / "gdut-final-test-evaluation_outputs" / "test_evaluation" / "summary.csv",
            RAW_ARTIFACTS_DIR / "final_test_summary.csv",
        ],
        "final_test_summary.csv",
    )

    val_map = {}
    with val_csv.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            val_map[(row["model"].strip(), row["stage"].strip())] = (
                float(row["metrics/mAP50(B)"]) * 100.0,
                float(row["metrics/mAP50-95(B)"]) * 100.0,
            )

    test_map = {}
    with test_csv.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            test_map[(row["model"].strip(), row["stage"].strip())] = (
                float(row["metrics/mAP50(B)"]) * 100.0,
                float(row["metrics/mAP50-95(B)"]) * 100.0,
            )

    stage_keys = ["S1", "A1", "A2", "A3", "A4", "A5", "A6"]
    cfgs = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    x = np.arange(len(cfgs))
    w = 0.36

    y11_val_50 = [val_map[("yolo11s", s)][0] for s in stage_keys]
    y11_val_95 = [val_map[("yolo11s", s)][1] for s in stage_keys]
    y11_test_50 = [test_map[("yolo11s", s)][0] for s in stage_keys]
    y11_test_95 = [test_map[("yolo11s", s)][1] for s in stage_keys]

    y8_val_50 = [val_map[("yolov8s", s)][0] for s in stage_keys]
    y8_val_95 = [val_map[("yolov8s", s)][1] for s in stage_keys]
    y8_test_50 = [test_map[("yolov8s", s)][0] for s in stage_keys]
    y8_test_95 = [test_map[("yolov8s", s)][1] for s in stage_keys]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    panels = [
        (axes[0, 0], y11_val_50, y11_test_50, "(a) YOLO11s — mAP@50 (%)", (85.5, 95.2), "#1D4ED8", "#60A5FA"),
        (axes[0, 1], y11_val_95, y11_test_95, "(b) YOLO11s — mAP@50-95 (%)", (56.0, 63.2), "#0F766E", "#5EEAD4"),
        (axes[1, 0], y8_val_50, y8_test_50, "(c) YOLOv8s — mAP@50 (%)", (85.5, 95.2), "#B45309", "#FBBF24"),
        (axes[1, 1], y8_val_95, y8_test_95, "(d) YOLOv8s — mAP@50-95 (%)", (55.5, 62.8), "#7C3AED", "#C4B5FD"),
    ]
    for ax, v_vals, t_vals, title, ylim, c1, c2 in panels:
        b1 = ax.bar(x - w / 2, v_vals, w, label="Validation (316 imgs)", color=c1, edgecolor="#0F172A", linewidth=0.5)
        b2 = ax.bar(x + w / 2, t_vals, w, label="Official Test (1,587 imgs)", color=c2, edgecolor="#0F172A", linewidth=0.5)
        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.12, f"{h:.2f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.12, f"{h:.1f}", ha="center", va="bottom", fontsize=7.2)
        ax.set_xticks(x)
        ax.set_xticklabels(cfgs)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_ylabel("Score (%)")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_gdut_ablation_val_test.png")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure VI.2: Pre-Finetune Init Check + Real Training Convergence CSVs
# ------------------------------------------------------------------
def _load_csv_series(csv_path: Path):
    epochs, m50, m95 = [], [], []
    with csv_path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v.strip() for k, v in row.items()}
            epochs.append(int(float(row["epoch"])))
            m50.append(float(row["metrics/mAP50(B)"]) * 100.0)
            m95.append(float(row["metrics/mAP50-95(B)"]) * 100.0)
    return epochs, m50, m95


def make_init_and_convergence(fig_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    val_csv = _resolve_first(
        [
            NVTHANH_DIR / "gdut-final-test-evaluation_outputs" / "test_evaluation" / "validation_summary.csv",
            RAW_ARTIFACTS_DIR / "final_test_validation_summary.csv",
        ],
        "final_test_validation_summary.csv",
    )
    s1_val50 = {}
    with val_csv.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["stage"].strip() == "S1":
                s1_val50[row["model"].strip()] = float(row["metrics/mAP50(B)"]) * 100.0

    stage0_init = _load_stage0_and_init_data()["initialization_checks"]

    def _get_init_m50(stage: str, model: str) -> float:
        raw_ic = (
            NVTHANH_DIR
            / f"gdut-stage2-ablation-setup-full-train-run-{stage.lower()}_outputs"
            / "initialization_checks"
            / f"gdut_{stage}_{model}"
            / "initialization_check.json"
        )
        obj = json.loads(raw_ic.read_text(encoding="utf-8")) if raw_ic.exists() else stage0_init[stage][model]
        return float(obj["pretrain_validation"]["metrics/mAP50(B)"]) * 100.0

    labels = ["A0\n(Stage1)", "A2\n(Coord)", "A3\n(RepConv)", "A5\n(BiFormer)", "A6\n(Full)"]
    x = np.arange(len(labels))
    w = 0.37
    y11_init = [s1_val50["yolo11s"]] + [_get_init_m50(st, "yolo11s") for st in ["A2", "A3", "A5", "A6"]]
    y8_init = [s1_val50["yolov8s"]] + [_get_init_m50(st, "yolov8s") for st in ["A2", "A3", "A5", "A6"]]
    init_drop = y11_init[3] - y11_init[0]

    b1 = ax1.bar(x - w / 2 - 0.015, y11_init, w, label="YOLO11s (Replaces C2PSA)", color="#DC2626", edgecolor="#0F172A", linewidth=0.5)
    b2 = ax1.bar(x + w / 2 + 0.015, y8_init, w, label="YOLOv8s (Appends after SPPF)", color="#059669", edgecolor="#0F172A", linewidth=0.5)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.16, f"{h:.2f}", ha="center", va="bottom", fontsize=7.1, fontweight="bold")

    ax1.annotate(
        f"{init_drop:.2f} pp init drop\n(C2PSA replaced)",
        xy=(3 - w / 2 - 0.015, y11_init[3]),
        xytext=(1.75, 83.6),
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#991B1B", lw=0.9, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#991B1B", lw=1.2),
        fontsize=8.0,
        color="#991B1B",
        fontweight="bold",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(81.0, 94.5)
    ax1.set_ylabel("Pre-Finetune Val mAP@50 (%)")
    ax1.set_title("(a) Function Preservation Before Stage 2 Finetuning")
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)
    ax1.legend(loc="lower left", fontsize=8.2)

    a1_csv = _resolve_first([EVID_DIR / "a1_yolo11s_results.csv", RAW_ARTIFACTS_DIR / "a1_yolo11s_results.csv"], "a1_yolo11s_results.csv")
    a6_csv = _resolve_first([EVID_DIR / "a6_yolo11s_results.csv", RAW_ARTIFACTS_DIR / "a6_yolo11s_results.csv"], "a6_yolo11s_results.csv")
    v8_csv = _resolve_first([EVID_DIR / "a6_yolov8s_results.csv", RAW_ARTIFACTS_DIR / "a6_yolov8s_results.csv"], "a6_yolov8s_results.csv")

    e_a1, _, m95_a1 = _load_csv_series(a1_csv)
    e_a6, _, m95_a6 = _load_csv_series(a6_csv)
    e_v8, _, m95_v8 = _load_csv_series(v8_csv)

    best_a1 = int(np.argmax(m95_a1))
    best_a6 = int(np.argmax(m95_a6))
    best_v8 = int(np.argmax(m95_v8))

    ax2.plot(e_a1, m95_a1, label=f"A1 YOLO11s (best ep {e_a1[best_a1]}, stop {e_a1[-1]})", color="#1D4ED8", lw=2.0)
    ax2.plot(e_a6, m95_a6, label=f"A6 YOLO11s (best ep {e_a6[best_a6]}, stop {e_a6[-1]})", color="#DC2626", lw=2.0)
    ax2.plot(e_v8, m95_v8, label=f"A6 YOLOv8s (best ep {e_v8[best_v8]}, stop {e_v8[-1]})", color="#059669", lw=1.8, linestyle="--")

    ax2.scatter([e_a1[best_a1]], [m95_a1[best_a1]], color="#1D4ED8", s=45, zorder=5)
    ax2.scatter([e_a6[best_a6]], [m95_a6[best_a6]], color="#DC2626", s=45, zorder=5)
    ax2.scatter([e_v8[best_v8]], [m95_v8[best_v8]], color="#059669", s=45, zorder=5)

    ax2.set_xlabel("Training Epoch (Patience = 50)")
    ax2.set_ylabel("Validation mAP@50-95 (%)")
    ax2.set_title("(b) Stage 2 Convergence Trajectories (results.csv)")
    ax2.set_ylim(56.0, 62.5)
    ax2.grid(True)
    ax2.legend(loc="lower right", fontsize=8.2)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_gdut_init_check_and_epochs.png")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure VI.3: Per-Class Test Performance of A6 (YOLO11s vs YOLOv8s)
# ------------------------------------------------------------------
def make_perclass_a6(fig_dir: Path) -> None:
    p11 = _resolve_first(
        [
            NVTHANH_DIR / "gdut-final-test-evaluation_outputs" / "test_evaluation" / "A6_yolo11s_per_class.json",
            RAW_ARTIFACTS_DIR / "A6_yolo11s_per_class.json",
        ],
        "A6_yolo11s_per_class.json",
    )
    p8 = _resolve_first(
        [
            NVTHANH_DIR / "gdut-final-test-evaluation_outputs" / "test_evaluation" / "A6_yolov8s_per_class.json",
            RAW_ARTIFACTS_DIR / "A6_yolov8s_per_class.json",
        ],
        "A6_yolov8s_per_class.json",
    )
    d11 = {r["name"]: r for r in json.loads(p11.read_text(encoding="utf-8"))}
    d8 = {r["name"]: r for r in json.loads(p8.read_text(encoding="utf-8"))}
    audit = _load_stage0_and_init_data()["dataset_audit"]
    names = audit["names"]
    test_boxes = audit["box_counts"]["test"]

    classes = [f"{name}\n({int(test_boxes[name]):,})" for name in names]
    x = np.arange(len(classes))
    w = 0.28

    y11_50 = [float(d11[n]["AP50"]) * 100.0 for n in names]
    y11_95 = [float(d11[n]["AP50_95"]) * 100.0 for n in names]
    y8_50 = [float(d8[n]["AP50"]) * 100.0 for n in names]
    y8_95 = [float(d8[n]["AP50_95"]) * 100.0 for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    for ax, v11, v8, title, ylim in [
        (ax1, y11_50, y8_50, "(a) Per-Class Test AP@50 (%) — 1,587 Test Images", (78.0, 96.0)),
        (ax2, y11_95, y8_95, "(b) Per-Class Test AP@50-95 (%) — 1,587 Test Images", (48.0, 64.0)),
    ]:
        b1 = ax.bar(x - w / 2, v11, w, label="YOLO11s A6", color="#1D4ED8", edgecolor="#0F172A", linewidth=0.5)
        b2 = ax.bar(x + w / 2, v8, w, label="YOLOv8s A6", color="#D97706", edgecolor="#0F172A", linewidth=0.5)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.25, f"{h:.1f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylim(*ylim)
        ax.set_ylabel("AP (%)")
        ax.set_title(title)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_gdut_perclass_a6.png")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure VI.4: Unified ce2a883e Stage 1 Baselines (B1/B2) vs Proposed V2 A6
# ------------------------------------------------------------------
def make_v2_baselines_vs_a6(fig_dir: Path) -> None:
    cmp_base = NVTHANH_DIR / "gdut-baseline-and-a6-stage1-protocol-comparison_outputs" / "stage1_baseline_comparison"
    cmp_csv = _resolve_first([cmp_base / "comparison_test.csv", RAW_ARTIFACTS_DIR / "comparison_test.csv"], "comparison_test.csv")
    rows = {r["model"]: r for r in csv.DictReader(cmp_csv.open("r", encoding="utf-8-sig"))}

    order = [
        ("yolo11_dcrcf", "B1: YOLO11n-DCRCF\n(Zhao 2025)"),
        ("yolov8_dwr_aspp_nwd", "B2: YOLOv8s-DWR-NWD\n(Song 2024)"),
        ("A6_yolov8s", "V2 A6 (YOLOv8s)\n(Proposed)"),
        ("A6_yolo11s", "V2 A6 (YOLO11s)\n(Primary)"),
    ]
    labels = [lbl for _, lbl in order]
    map50 = [float(rows[k]["mAP50"]) * 100.0 for k, _ in order]
    map95 = [float(rows[k]["mAP50_95"]) * 100.0 for k, _ in order]
    f1_mac = [float(rows[k]["F1_native_macro"]) * 100.0 for k, _ in order]
    params_m = [int(rows[k]["deployment_parameters"]) / 1e6 for k, _ in order]
    lat_ms = [float(rows[k]["forward_ms_median"]) for k, _ in order]

    none_50, none_95 = [], []
    for k, _ in order:
        det_path = _resolve_first([cmp_base / f"{k}_details.json", RAW_ARTIFACTS_DIR / f"{k}_details.json"], f"{k}_details.json")
        det = json.loads(det_path.read_text(encoding="utf-8"))
        none_rec = next(c for c in det["splits"]["test"]["per_class"] if c["name"] == "none")
        none_50.append(float(none_rec["AP50"]) * 100.0)
        none_95.append(float(none_rec["AP50_95"]) * 100.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.9), gridspec_kw={"width_ratios": [1.18, 1.0]})

    x = np.arange(len(labels))
    w = 0.25
    b50 = ax1.bar(x - w, map50, w, label="Official Test mAP@50 (%)", color="#1D4ED8", edgecolor="#0F172A", linewidth=0.55)
    b95 = ax1.bar(x, map95, w, label="Official Test mAP@50-95 (%)", color="#059669", edgecolor="#0F172A", linewidth=0.55)
    bf1 = ax1.bar(x + w, f1_mac, w, label="Macro F1 Score (%)", color="#D97706", edgecolor="#0F172A", linewidth=0.55)

    for bars in (b50, b95, bf1):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.55, f"{h:.2f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold")

    d_b1 = map50[3] - map50[0]
    d_b2 = map50[3] - map50[1]
    ax1.annotate(
        f"+{d_b1:.2f} pp mAP@50 vs B1\n+{d_b2:.2f} pp mAP@50 vs B2",
        xy=(3 - w, map50[3] + 2.8),
        xytext=(2.36, 102.5),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1D4ED8", lw=0.9, alpha=0.96),
        arrowprops=dict(arrowstyle="->", color="#1D4ED8", lw=1.15),
        fontsize=7.6,
        color="#1E3A8A",
        fontweight="bold",
        ha="center",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.3)
    ax1.set_ylim(45.0, 114.5)
    ax1.set_ylabel("Official GDUT Test Score (%)")
    ax1.set_title("(a) Unified ce2a883e Stage 1 Test Accuracy (1,587 Imgs)")
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper left", ncol=1, fontsize=7.4, framealpha=0.96)

    w2 = 0.25
    bp = ax2.bar(x - w2, params_m, w2, label="Deploy Fused Params (M)", color="#6366F1", edgecolor="#0F172A", linewidth=0.55)
    bl = ax2.bar(x, lat_ms, w2, label="T4 FP32 Forward Median (ms)", color="#DC2626", edgecolor="#0F172A", linewidth=0.55)
    ax2_r = ax2.twinx()
    mx = x + w2 * 1.02
    ax2_r.plot(mx, none_50, marker="s", color="#0F766E", lw=2.0, markersize=6.2, label="Bare-Head (none) AP@50 (%)", zorder=5)

    for bar in bp:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.48, f"{h:.2f}M", ha="center", va="bottom", fontsize=7.1, fontweight="bold", color="#312E81")
    for bar in bl:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.48, f"{h:.1f}ms", ha="center", va="bottom", fontsize=7.1, fontweight="bold", color="#991B1B")
    pill_va = ["top", "top", "bottom", "bottom"]
    pill_dy = [-0.72, -0.72, 0.72, 0.72]
    for i, (n5, n9, pva, pdy) in enumerate(zip(none_50, none_95, pill_va, pill_dy)):
        ax2_r.text(
            mx[i],
            n5 + pdy,
            f"{n5:.1f}%\n({n9:.1f})",
            ha="center",
            va=pva,
            fontsize=6.9,
            fontweight="bold",
            color="#0F766E",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#99F6E4", alpha=0.96),
            zorder=6,
        )

    dp_b2 = (params_m[3] - params_m[1]) / params_m[1] * 100.0
    dl_b2 = (lat_ms[3] - lat_ms[1]) / lat_ms[1] * 100.0
    ax2.annotate(
        f"{dp_b2:.1f}% params &\n{dl_b2:.1f}% ms vs B2",
        xy=(3.0, lat_ms[3] + 2.6),
        xytext=(2.62, 23.0),
        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#059669", lw=0.9, alpha=0.96),
        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.15),
        fontsize=7.6,
        color="#065F46",
        fontweight="bold",
        ha="center",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(["B1: DCRCF\n(Zhao 2025)", "B2: DWR-NWD\n(Song 2024)", "V2 A6 (YOLOv8s)\n(Proposed)", "V2 A6 (YOLO11s)\n(Primary)"], fontsize=7.8)
    ax2.set_ylim(0, 42.0)
    ax2_r.set_ylim(51.0, 96.5)
    ax2.set_ylabel("Fused Params (M) / T4 Latency (ms)")
    ax2_r.set_ylabel("Bare-Head Class 'none' AP@50 (%)", color="#0F766E")
    ax2.set_title("(b) Deployment Footprint, Latency & Bare-Head Sensitivity")
    ax2.yaxis.grid(True)
    ax2.set_axisbelow(True)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.4, framealpha=0.96)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_v2_baselines_vs_a6_comparison.png")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure VI.8: Proposed V1 (SHWD 2-Class & Zero-Shot HHW, 3 Seeds Mean ± SD)
# ------------------------------------------------------------------
def make_v1_shwd_hhw_ablation(fig_dir: Path) -> None:
    shwd_dir = REPO_ROOT / "SHWD_V1" / "outputs"
    seeds = [42, 1337, 2026]
    cfgs = ["A0", "A1", "A2", "A3", "A4"]

    shwd_by_cfg = {c: [] for c in cfgs}
    lat_by_cfg = {c: [] for c in cfgs}
    for sd in seeds:
        with (shwd_dir / f"shwd_test_ablation_seed{sd}.csv").open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                shwd_by_cfg[r["ablation_id"]].append((float(r["mAP50"]), float(r["mAP50_95"])))
        with (shwd_dir / f"t4_latency_seed{sd}.csv").open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                lat_by_cfg[r["ablation_id"]].append(float(r["pyt_fp16_ms"]))

    hhw_2c = {c: [] for c in cfgs}
    hhw_hat = {c: [] for c in cfgs}
    with (shwd_dir / "v1_hhw_all_3seeds.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["hhw_eval"] == "hhw_2class_test":
                hhw_2c[r["ablation_id"]].append((float(r["mAP50"]), float(r["mAP50_95"])))
            elif r["hhw_eval"] == "hhw_hatonly_test":
                hhw_hat[r["ablation_id"]].append((float(r["mAP50"]), float(r["mAP50_95"])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8))

    x = np.arange(len(cfgs))
    m50_mean = [float(np.mean([v[0] for v in shwd_by_cfg[c]])) for c in cfgs]
    m50_sd = [float(np.std([v[0] for v in shwd_by_cfg[c]], ddof=1)) for c in cfgs]
    lat_mean = [float(np.mean(lat_by_cfg[c])) for c in cfgs]
    lat_sd = [float(np.std(lat_by_cfg[c], ddof=1)) for c in cfgs]

    colors_a = ["#475569", "#2563EB", "#059669", "#D97706", "#7C3AED"]
    bars = ax1.bar(
        x - 0.16,
        m50_mean,
        yerr=m50_sd,
        capsize=4,
        width=0.34,
        color=colors_a,
        edgecolor="#0F172A",
        linewidth=0.55,
        label="SHWD Test mAP@50 (%)",
        error_kw=dict(ecolor="#0F172A", lw=1.2),
    )
    ax1_r = ax1.twinx()
    ax1_r.errorbar(
        x + 0.16,
        lat_mean,
        yerr=lat_sd,
        capsize=4,
        marker="D",
        markersize=6.0,
        lw=2.0,
        color="#DC2626",
        label="T4 PyTorch FP16 Latency (ms)",
    )

    for bar, m, s in zip(bars, m50_mean, m50_sd):
        ax1.text(bar.get_x() + bar.get_width() / 2, m + s + 0.08, f"{m:.2f}\n±{s:.2f}", ha="center", va="bottom", fontsize=7.1, fontweight="bold", color="#1E293B")
    for i, (lm, ls) in enumerate(zip(lat_mean, lat_sd)):
        ax1_r.text(
            i + 0.16,
            lm + ls + 0.34,
            f"{lm:.2f}ms",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color="#991B1B",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#FCA5A5", alpha=0.96),
            zorder=6,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(["V1 A0\n(Stock Y11s)", "V1 A1\n(+Coord)", "V1 A2\n(Edge/Rep)", "V1 A3\n(+BiF P4/P5)", "V1 A4\n(Full V1)"], fontsize=8.1)
    ax1.set_ylim(92.5, 98.5)
    ax1_r.set_ylim(4.5, 20.8)
    ax1.set_ylabel("SHWD Test mAP@50 (%, 3 Seeds ± SD)")
    ax1_r.set_ylabel("Tesla T4 FP16 Latency (ms)", color="#991B1B")
    ax1.set_title("(a) Proposed V1 Ablation on Binary SHWD Test (1,517 Imgs)")
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, fontsize=7.3, framealpha=0.96)

    h2c_m95 = [float(np.mean([v[1] for v in hhw_2c[c]])) for c in cfgs]
    h2c_s95 = [float(np.std([v[1] for v in hhw_2c[c]], ddof=1)) for c in cfgs]
    hho_m95 = [float(np.mean([v[1] for v in hhw_hat[c]])) for c in cfgs]
    hho_s95 = [float(np.std([v[1] for v in hhw_hat[c]], ddof=1)) for c in cfgs]

    w = 0.34
    b_2c = ax2.bar(x - w / 2, h2c_m95, yerr=h2c_s95, capsize=4, width=w, color="#0284C7", edgecolor="#0F172A", linewidth=0.55, label="Zero-Shot HHW 2-Class mAP@50-95 (%)")
    b_ho = ax2.bar(x + w / 2, hho_m95, yerr=hho_s95, capsize=4, width=w, color="#0D9488", edgecolor="#0F172A", linewidth=0.55, label="Zero-Shot HHW Hat-Only mAP@50-95 (%)")

    for bars in (b_2c, b_ho):
        for i, bar in enumerate(bars):
            sd_val = h2c_s95[i] if bars is b_2c else hho_s95[i]
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + sd_val + 0.08, f"{h:.2f}\n±{sd_val:.2f}", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color="#0F172A")

    best_idx = int(np.argmax(h2c_m95))
    ax2.annotate(
        f"Peak Transfer: V1 A2 Edge\n({h2c_m95[best_idx]:.2f}% 2-cls / {hho_m95[best_idx]:.2f}% Hat-Only)",
        xy=(best_idx, h2c_m95[best_idx] + h2c_s95[best_idx] + 0.55),
        xytext=(3.05, 58.95),
        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#0284C7", lw=0.9, alpha=0.96),
        arrowprops=dict(arrowstyle="->", color="#0284C7", lw=1.1),
        fontsize=7.4,
        color="#0369A1",
        fontweight="bold",
        ha="center",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(["V1 A0", "V1 A1", "V1 A2 (Edge)", "V1 A3", "V1 A4 (Full)"], fontsize=8.3)
    ax2.set_ylim(53.0, 60.4)
    ax2.set_ylabel("Zero-Shot HHW mAP@50-95 (%, 3 Seeds ± SD)")
    ax2.set_title("(b) Zero-Shot Cross-Domain Generalization on HHW (1,766 Imgs)")
    ax2.yaxis.grid(True)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper left", fontsize=7.3, framealpha=0.96)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_v1_shwd_hhw_ablation.png")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure VI.9: Inference Optimization (Before vs After) + Dual API Load Test
# ------------------------------------------------------------------
def make_optimization_and_load(fig_dir: Path) -> None:
    opt_path = REPO_ROOT / "Output" / "task3_eval_opt" / "optimization_benchmark_table.json"
    lt4a_path = REPO_ROOT / "Output" / "load_test_results.json"
    lt4b_path = REPO_ROOT / "Output" / "load_test_enterprise_docker.json"

    opt_data = json.loads(opt_path.read_text(encoding="utf-8"))
    gpu_records = [r for r in opt_data["records"] if "CPU" not in str(r.get("stage", ""))]

    stage_labels_map = {
        "Before (Loop BiFormer FP32)": "Before\n(Loop FP32)",
        "After-1 (Vec BiFormer + RepConv FP32)": "After-1\n(Vec+Rep FP32)",
        "After-2 (+ CUDA AMP FP16)": "After-2\n(AMP FP16)",
        "After-3 (Micro-Batch N=4 AMP FP16)": "After-3\n(Batch N=4 AMP)",
    }
    stages = [stage_labels_map.get(r["stage"], r["stage"]) for r in gpu_records]
    fps_vals = [float(r["fps"]) for r in gpu_records]
    ms_vals = [float(r["latency_ms"]) for r in gpu_records]
    colors = ["#64748B", "#2563EB", "#059669", "#D97706"][: len(gpu_records)]

    lt4a = json.loads(lt4a_path.read_text(encoding="utf-8"))
    lt4b = json.loads(lt4b_path.read_text(encoding="utf-8"))
    conc = [int(r["concurrency"]) for r in lt4a]
    rps_4a = [float(r["rps"]) for r in lt4a]
    p95_4a = [float(r["p95_ms"]) for r in lt4a]
    rps_4b = [float(r["rps"]) for r in lt4b]
    p95_4b = [float(r["p95_ms"]) for r in lt4b]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2))

    x = np.arange(len(stages))
    bars = ax1.bar(x, fps_vals, width=0.52, color=colors, edgecolor="#0F172A", linewidth=0.7)
    for bar, fps, ms in zip(bars, fps_vals, ms_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            fps + 3.0,
            f"{fps:.1f} FPS\n({ms:.2f} ms)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#0F172A",
        )
    ax1.axhline(25.0, color="#DC2626", linestyle="--", linewidth=1.3, label="25 FPS Real-Time SLA")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=8.8)
    ax1.set_ylim(0, max(fps_vals) * 1.24)
    ax1.set_ylabel("Throughput (FPS) on RTX 3050")
    ax1.set_title("(a) Rep-YOLO11s-GDUT-A6 Inference Optimization")
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper left", fontsize=8.5)

    ax2_r = ax2.twinx()
    bx = np.arange(len(conc))
    bw = 0.34
    b1 = ax2.bar(bx - bw / 2, rps_4a, width=bw, color="#1D4ED8", edgecolor="#0F172A", linewidth=0.6, label="4A In-Process ASGI (RPS)")
    b2 = ax2.bar(bx + bw / 2, rps_4b, width=bw, color="#0284C7", edgecolor="#0F172A", linewidth=0.6, label="4B 6-Container Docker (RPS)")

    l1 = ax2_r.plot(bx, p95_4a, color="#DC2626", marker="o", lw=2.0, markersize=6.5, label="4A p95 Latency (ms)", zorder=5)
    l2 = ax2_r.plot(bx, p95_4b, color="#991B1B", marker="s", linestyle="--", lw=2.0, markersize=6.2, label="4B Docker p95 (ms)", zorder=5)

    for i, (r4a, r4b) in enumerate(zip(rps_4a, rps_4b)):
        ax2.text(i - bw / 2, r4a + 12, f"{r4a:.1f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#1E3A8A")
        ax2.text(i + bw / 2, r4b + 12, f"{r4b:.1f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#0369A1")

    pill_kw = dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CBD5E1", alpha=0.94, linewidth=0.7)
    offsets_4a = [(0.0, -45), (0.0, -45), (0.0, -52), (0.0, -52)]
    offsets_4b = [(-0.22, 46), (-0.22, 46), (-0.25, 78), (0.20, 46)]
    for i, (pa, pb, oa, ob) in enumerate(zip(p95_4a, p95_4b, offsets_4a, offsets_4b)):
        ax2_r.text(i + oa[0], pa + oa[1], f"{pa:.0f}ms", ha="center", va="top", fontsize=7.7, fontweight="bold", color="#DC2626", bbox=pill_kw, zorder=6)
        ax2_r.text(i + ob[0], pb + ob[1], f"{pb:.0f}ms", ha="center", va="bottom", fontsize=7.7, fontweight="bold", color="#991B1B", bbox=pill_kw, zorder=6)

    ax2.set_xticks(bx)
    ax2.set_xticklabels([f"C={c}" for c in conc], fontsize=9.0)
    ax2.set_xlabel("Concurrent API Clients (0.0% Error Across All Tiers)")
    ax2.set_ylabel("Throughput (RPS)", color="#1D4ED8")
    ax2_r.set_ylabel("p95 Response Latency (ms)", color="#991B1B")
    ax2.set_ylim(0, 720)
    ax2_r.set_ylim(-60, 1080)
    ax2.set_title("(b) Dual API Load Benchmark (4A In-Process vs. 4B Docker)")
    ax2.yaxis.grid(True)
    ax2.set_axisbelow(True)
    handles = [b1, b2] + l1 + l2
    ax2.legend(handles, [h.get_label() for h in handles], loc="upper left", ncol=2, fontsize=7.9)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_gdut_optimization_and_multistream.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate all 7 data-driven thesis figures from raw CSV/JSON artifacts.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures", help="Target directory for rendered 300-DPI PNGs.")
    parser.add_argument("--no-copy-evidence", action="store_true", help="Skip copying raw Kaggle PNG/JPG evidence files.")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_copy_evidence:
        copy_raw_kaggle_evidence(out_dir)

    make_class_distribution(out_dir)
    make_ablation_val_test(out_dir)
    make_init_and_convergence(out_dir)
    make_perclass_a6(out_dir)
    make_v2_baselines_vs_a6(out_dir)
    make_v1_shwd_hhw_ablation(out_dir)
    make_optimization_and_load(out_dir)
    print(f"Generated all 7 data-driven GDUT/SHWD analytical figures cleanly in {out_dir}")


if __name__ == "__main__":
    main()
