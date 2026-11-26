"""
scripts/sync_public_audit_repo.py
=================================
Clean-Room Public Audit Export & 7-Gate Verification Builder for `hannhu11/AI-Capstone`.
Exports strictly whitelisted GDUT-HWD 5-Class VMS source code, benchmarks, MLOps configs,
clean LaTeX thesis source, and Review 3 UI evidence into `_publish_ai_capstone/` with zero
legacy git history, zero heavy checkpoints/videos/datasets, and zero AAIML 2027 cross-contamination.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "_publish_ai_capstone"

WHITELIST_ROOT_FILES = [
    "custom_ablation_modules.py",
    "requirements_docker.txt",
    ".env.example",
    ".dockerignore",
    "Dockerfile",
    "Dockerfile.gateway",
    "docker-compose.enterprise.yml",
    "docker-compose.edge-simulation.yml",
    "deploy/nginx/nginx.conf",
    ".github/workflows/ci.yml",
]

WHITELIST_TESTS = [
    "tests/test_cctv_forensic_fixes.py",
    "tests/test_gdut_enterprise_reid_and_controls.py",
    "tests/test_persistent_tracking_focus.py",
    "tests/test_reid_anti_hijack_and_consolidation.py",
    "tests/test_stationary_worker_ppe.py",
    "tests/test_task3_ui_api.py",
    "tests/test_tracking_and_alert_engine.py",
]

WHITELIST_SCRIPTS = [
    "scripts/ablation_factory.py",
    "scripts/benchmark_gdut_vms_end2end.py",
    "scripts/benchmark_task3_optimization.py",
    "scripts/benchmark_multistream_batched_rtx3050.py",
    "scripts/load_test_api.py",
    "scripts/test_dynamic_taxonomy_gdut.py",
    "scripts/export_gdut_a6_deploy.py",
    "scripts/run_local_dashboard.py",
    "scripts/sync_public_audit_repo.py",
]

WHITELIST_OUTPUTS = [
    "Output/task3_eval_opt/gdut_vms_end2end_reid_benchmark.json",
    "Output/task3_eval_opt/gdut_a6_switch_to_deploy_benchmark.json",
    "Output/task3_eval_opt/optimization_benchmark_table.json",
    "Output/task3_eval_opt/optimization_benchmark_table.csv",
    "Output/operational_multistream/rtx3050_gdut_a6_batched_multistream_results.json",
    "Output/load_test_results.json",
    "Output/load_test_enterprise_docker.json",
]

WHITELIST_SCREENSHOTS = [
    "docs/ui_screenshots/01_1920x1080_main.png",
    "docs/ui_screenshots/02_1366x768_compact.png",
    "docs/ui_screenshots/03_runtime_sla_4tables_modal.png",
    "docs/ui_screenshots/03b_runtime_sla_loadtest.png",
    "docs/ui_screenshots/04_webcam_c960_canvas_mode.png",
    "docs/ui_screenshots/05_solo_focus_and_export.png",
    "docs/ui_screenshots/06_grafana_live_monitoring.png",
]

WHITELIST_DOCS = [
    "docs/UAT_SURVEY_TEMPLATE_AND_SUS_PROTOCOL.md",
    "docs/CHANGELOG_GDUT_VMS.md",
    "docs/research_baselines/01_KET_QUA_A1_A6.md",
    "docs/research_baselines/02_PHUONG_PHAP_VA_KIEN_TRUC.md",
    "docs/research_baselines/BASELINE_AUDIT_B1_B2_REVIEW3.md",
    "docs/research_baselines/DANH_SACH_KAGGLE_CHINH_THUC.md",
]

FORBIDDEN_EXTS = {
    ".pt", ".pth", ".onnx", ".engine", ".mp4", ".avi",
    ".zip", ".7z", ".tar", ".gz", ".db", ".sqlite",
    ".key", ".pem", ".pdf",
}

CLEAN_GITIGNORE = """# Runtime secrets & local environment
.env
*.key
*.pem
kaggle.json

# Heavy datasets, weights, media & runtime databases (use GitHub Releases for .pt / .pdf)
*.pt
*.pth
*.onnx
*.engine
*.mp4
*.avi
*.zip
*.db
*.sqlite
Dataset/
AAIML 2027/
snapshots/*.jpg
snapshots/*.png
!snapshots/.gitkeep
!docs/ui_screenshots/*.png

# Python & IDE caches
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
postgres_data/
prometheus_data/
grafana_data/
"""

CLEAN_README = """# AEGIS-CCTV Enterprise VMS — Safety Helmet & Color Classification (`GDUT-HWD 5-Class`)

Capstone Project (`FA26AI16` | Group `GFA26AI17` | FPT University) implementing **Rep-YOLO11s (`CoordConv` + 6× `RepConv` + `BiFormerBlockLite`)** with **Head-Level ByteTrack**, **Confirm-Before-Mint Non-Biometric Upper-Torso Persistent Re-ID**, a **6-Microservice Docker Enterprise Stack**, and live **Prometheus/Grafana** telemetry.

- **Task Scope:** Pure **GDUT-HWD 5-Class** (`0: blue`, `1: white`, `2: yellow`, `3: red`, `4: none`) with Role-by-Color site hierarchy (`white=PM/ENG`, `blue=SUP/MEP`, `red=HSE`, `yellow=WORKER`, `none=VIOLATION`).
- **Privacy by Design:** Zero facial recognition; persistent worker re-identification relies exclusively on below-head upper-torso geometry, 520-D appearance/HOG descriptors, and optional Multi-Layer `MobileNetV3-Small` features.

---

## 1. 6-Microservice Architecture (`docker-compose.enterprise.yml`)

```text
Browser / Chrome UI (:8080)
   ├── /                     ──▶ frontend-nginx (Nginx 1.27 Alpine reverse proxy & static UI)
   ├── /api/*                ──▶ api-gateway (FastAPI no-torch Control Plane + Circuit Breaker)
   │                              ├── PostgreSQL 16 (`incidents`, `feedback`, `role_policies`, `personnel_sessions`)
   │                              └── HTTP Pool / HALF_OPEN Probe ──▶ ai-inference (PyTorch CUDA GPU + Rep-YOLO11s + Re-ID)
   └── /video_feed, /ws/*    ──▶ ai-inference (:8001, Binary JPEG WebSocket + MJPEG)

Monitoring:
   Prometheus (:9090) scrapes `api-gateway:8000/metrics` + `ai-inference:8001/metrics` ──▶ Grafana (:3000)
```

---

## 2. Quick Start (Clean Machine Setup)

1. **Clone & Configure Environment:**
   ```bash
   git clone https://github.com/hannhu11/AI-Capstone.git
   cd AI-Capstone
   cp .env.example .env
   ```
2. **Download Deploy Checkpoint & Thesis PDF (GitHub Releases):**
   Heavy binaries (`rep_yolo11s_gdut_a6_deploy.pt` and `Capstone_Thesis_Report_FA26AI16_GFA26AI17.pdf`) are distributed via the repository's **GitHub Releases** tab rather than git blobs:
   - Place `rep_yolo11s_gdut_a6_deploy.pt` at `Output/GDUT_Bộ số 1_stage2-full-train-run-a6/deploy_weights/rep_yolo11s_gdut_a6_deploy.pt` (or set `MODEL_PATH` in `.env`).
   - Place test video clip (e.g., `construction_site_workers_1080p.mp4`) in `video_test/`.
3. **Launch Full 6-Container Enterprise Stack:**
   ```bash
   docker compose -f docker-compose.enterprise.yml up -d --build
   ```
   - **Enterprise VMS UI:** `http://localhost:8080`
   - **Swagger / OpenAPI Docs:** `http://localhost:8080/docs`
   - **Prometheus Metrics / Targets:** `http://localhost:9090`
   - **Grafana Telemetry Dashboard:** `http://localhost:3000` (credentials configured in `.env`)
4. **Weightless CI / Local Test Verification:**
   ```bash
   python -m compileall -q src tests scripts
   MODEL_PATH=ci_stub_missing.pt CUDA_VISIBLE_DEVICES=-1 python -m pytest tests -q
   ```

---

## 3. Verified Quantitative Benchmarks (RTX 3050 Laptop GPU, 1920×1080)

### A. 5-Tier End-to-End Inference & Re-ID Breakdown (`gdut_vms_end2end_reid_benchmark.json`)
| Tier | Runtime Precision / Config | Mean Latency | Throughput | p95 Latency | Peak VRAM |
|---|---|---:|---:|---:|---:|
| 1. Pure GDUT-A6 Detector | Strict FP32 (`allow_tf32=False`) | `14.55 ms` | `68.8 FPS` | `15.21 ms` | `106.1 MB` |
| 2. Pure GDUT-A6 Detector **[Default B=1]** | `TF32 FP32` (`allow_tf32=True`) | `10.79 ms` | `92.6 FPS` | `12.69 ms` | `215.0 MB` |
| 3. Pure GDUT-A6 Detector | Selective AMP FP16 (`TF32=False`) | `10.85 ms` | `92.2 FPS` | `13.76 ms` | `210.2 MB` |
| 4. End-to-End + 3-Strip Non-Biometric Re-ID | `TF32 FP32`, `use_deep=False` | `16.33 ms` | `61.2 FPS` | `19.68 ms` | `215.0 MB` |
| 5. Full VMS + Multi-Layer MobileNetV3 Re-ID | `TF32 FP32`, `use_deep=True` (`refresh=30`) | `16.00 ms` | `62.5 FPS` | `18.72 ms` | `218.6 MB` |

- **RepConv Structural Reparameterization (`gdut_a6_switch_to_deploy_benchmark.json`):** 6 layers (`[1, 3, 5, 7, 17, 20]`), `9,788,575 -> 9,487,583` parameters (`-300,992`), FP32 max diff `7.93e-4`.
- **FP32 vs AMP FP16 Video Parity (`optimization_benchmark_table.json`, 12 frames):**
  - `Batch=1`: `0` box-count delta (`12/12`), mean confidence diff `0.00036` (`10.05 ms / 99.5 FPS`).
  - `Micro-Batch N=4`: `0` box-count delta (`12/12`), mean confidence diff `0.00032` (`6.90 ms/frame / 144.9 FPS`).

### B. Dual Concurrent Load Tests (`C=10..100`, `1,400` total HTTP requests per sweep)
| Concurrency | In-Process CPU FastAPI (`load_test_results.json`) | 6-Container Docker Stack Nginx→Gateway→PostgreSQL (`load_test_enterprise_docker.json`) | Errors |
|---|---:|---:|---:|
| `C=10` (200 reqs) | `446.7 RPS` (`p95=30.56 ms`) | `260.8 RPS` (`p95=69.09 ms`) | `0.00%` |
| `C=25` (300 reqs) | `487.9 RPS` (`p95=58.05 ms`) | `256.6 RPS` (`p95=133.69 ms`) | `0.00%` |
| `C=50` (400 reqs) | `454.1 RPS` (`p95=150.70 ms`) | `238.1 RPS` (`p95=250.91 ms`) | `0.00%` |
| `C=100` (500 reqs) | `501.1 RPS` (`p95=199.25 ms`, `p99=200.31 ms`) | `225.2 RPS` (`p95=480.67 ms`, `p99=500.78 ms`) | `0.00%` |

---

## 4. Visual Evidence & Operational Limitations
- UI & Monitoring Screenshots (v2.3):
  - [Enterprise Dashboard 1920×1080](docs/ui_screenshots/01_1920x1080_main.png) | [Compact 1366×768](docs/ui_screenshots/02_1366x768_compact.png)
  - [Runtime & SLA Tables 1–3](docs/ui_screenshots/03_runtime_sla_4tables_modal.png) | [Runtime & SLA Dual Load Tests 4A/4B](docs/ui_screenshots/03b_runtime_sla_loadtest.png)
  - [C960 Ultra Binary WebSocket Webcam Mode](docs/ui_screenshots/04_webcam_c960_canvas_mode.png) | [Solo Focus Lock & CSV Export](docs/ui_screenshots/05_solo_focus_and_export.png)
  - [Live Prometheus + Grafana Telemetry (`42.4 FPS`, `18.3 ms`, `2` Gallery IDs)](docs/ui_screenshots/06_grafana_live_monitoring.png)
- **Field Status (`docs/UAT_SURVEY_TEMPLATE_AND_SUS_PROTOCOL.md`):** UAT & SUS rubric is finalized and maintained strictly at **`Pending Field Trial`** (no synthetic SUS scores). Physical USB `C960 Ultra` room-light cosine threshold calibration awaits on-site camera testing.
"""


def _copy_clean_file(src_rel: str, dest_root: Path):
    src = ROOT / src_rel
    if not src.exists():
        raise FileNotFoundError(f"Missing whitelisted source file: {src}")
    dst = dest_root / src_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_src_tree(dest_root: Path):
    src_dir = ROOT / "src"
    dst_dir = dest_root / "src"
    shutil.copytree(
        src_dir,
        dst_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.pyd"),
    )


def _copy_monitoring_tree(dest_root: Path):
    _copy_clean_file("monitoring/prometheus.yml", dest_root)
    _copy_clean_file("monitoring/grafana/dashboards/aegis_vms_overview.json", dest_root)
    prov_src = ROOT / "monitoring" / "grafana" / "provisioning"
    if prov_src.exists():
        for p in prov_src.rglob("*"):
            if p.is_file():
                rel = p.relative_to(ROOT).as_posix()
                _copy_clean_file(rel, dest_root)


def _rmtree_force(path: Path):
    def _on_err(func, p, _exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    if path.exists():
        shutil.rmtree(path, onerror=_on_err)


def _clean_target_dir(preserve_git: bool):
    if not TARGET.exists():
        TARGET.mkdir(parents=True, exist_ok=True)
        return
    for entry in list(TARGET.iterdir()):
        if preserve_git and entry.name == ".git":
            continue
        if entry.is_dir():
            _rmtree_force(entry)
        else:
            try:
                entry.unlink()
            except PermissionError:
                os.chmod(entry, stat.S_IWRITE)
                entry.unlink()


def sync_export(init_git: bool = False) -> Path:
    _clean_target_dir(preserve_git=(not init_git))

    _copy_src_tree(TARGET)
    _copy_monitoring_tree(TARGET)

    for group in (
        WHITELIST_ROOT_FILES,
        WHITELIST_TESTS,
        WHITELIST_SCRIPTS,
        WHITELIST_OUTPUTS,
        WHITELIST_SCREENSHOTS,
        WHITELIST_DOCS,
    ):
        for rel in group:
            _copy_clean_file(rel, TARGET)

    # Thesis LaTeX sources only (no PDFs, aux logs, or raster figure dumps)
    _copy_clean_file("thesis_report/main.tex", TARGET)
    _copy_clean_file("thesis_report/references.bib", TARGET)
    chapters_dir = ROOT / "thesis_report" / "chapters"
    for tex in sorted(chapters_dir.glob("*.tex")):
        _copy_clean_file(tex.relative_to(ROOT).as_posix(), TARGET)

    # Placeholder empty directories
    for empty_dir in ("snapshots", "video_test"):
        d = TARGET / empty_dir
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").write_text("", encoding="utf-8")

    # Clean .gitignore and README.md
    (TARGET / ".gitignore").write_text(CLEAN_GITIGNORE, encoding="utf-8")
    (TARGET / "README.md").write_text(CLEAN_README, encoding="utf-8")

    # Compute AUDIT_MANIFEST.json covering all exported files
    manifest_entries: List[Dict[str, object]] = []
    total_bytes = 0
    max_file_bytes = 0
    for p in sorted(TARGET.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(TARGET).as_posix()
        if rel.startswith(".git/") or rel == "docs/AUDIT_MANIFEST.json":
            continue
        data = p.read_bytes()
        sz = len(data)
        total_bytes += sz
        max_file_bytes = max(max_file_bytes, sz)
        manifest_entries.append({
            "path": rel,
            "size_bytes": sz,
            "sha256": hashlib.sha256(data).hexdigest(),
        })

    manifest = {
        "repo": "https://github.com/hannhu11/AI-Capstone",
        "scope": "AEGIS-CCTV Enterprise VMS v2.3 (GDUT-HWD 5-Class Clean-Room Export)",
        "total_files": len(manifest_entries) + 1,
        "total_size_mb": round(total_bytes / (1024 * 1024), 3),
        "max_single_file_mb": round(max_file_bytes / (1024 * 1024), 3),
        "files": manifest_entries,
    }
    manifest_path = TARGET / "docs" / "AUDIT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[SYNC] Exported {manifest['total_files']} files | "
        f"Total={manifest['total_size_mb']} MB | MaxFile={manifest['max_single_file_mb']} MB -> {TARGET.name}/"
    )
    return TARGET


def _build_secret_patterns() -> List[tuple[str, re.Pattern[str]]]:
    pat_gh = "(" + "|".join(["gh" + "p_", "github_" + "pat_", "gh" + "o_", "gh" + "u_", "gh" + "s_", "gh" + "r_"]) + r")[A-Za-z0-9_]{8,}"
    pat_kg = "KG" + r"AT_[A-Za-z0-9_]{8,}"
    pat_goog = "AI" + r"za[0-9A-Za-z_-]{35}"
    pat_oai = "sk" + r"-[A-Za-z0-9]{16,}"
    pat_aws = "AK" + r"IA[0-9A-Z]{16}"
    pat_tg = r"[0-9]{8,}:[A-Za-z0-9_-]{25,}"
    pat_pem = "BEGIN " + r"(RSA|OPENSSH|EC|PGP) PRIVATE KEY"
    pat_win1 = "C:" + r"\\" + "Users" + r"\\" + "ADMIN"
    pat_win2 = "C:" + "/Users/" + "ADMIN"
    return [
        ("github_token", re.compile(pat_gh)),
        ("kaggle_token", re.compile(pat_kg)),
        ("google_api_key", re.compile(pat_goog)),
        ("openai_key", re.compile(pat_oai)),
        ("aws_key", re.compile(pat_aws)),
        ("bot_token", re.compile(pat_tg)),
        ("private_key", re.compile(pat_pem)),
        ("personal_path_win", re.compile(re.escape(pat_win1), re.IGNORECASE)),
        ("personal_path_posix", re.compile(re.escape(pat_win2), re.IGNORECASE)),
    ]


def verify_export(init_git: bool = False):
    files = [
        p for p in TARGET.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(TARGET).parts
    ]
    total_files = len(files)
    total_bytes = sum(p.stat().st_size for p in files)
    max_bytes = max((p.stat().st_size for p in files), default=0)
    total_mb = total_bytes / (1024 * 1024)
    max_mb = max_bytes / (1024 * 1024)

    # Gate 1: Size, File-Count Budget & Binary Guard
    assert total_mb <= 15.0, f"Gate 1 failed: total_size_mb={total_mb:.2f} > 15.0"
    assert total_files <= 120, f"Gate 1 failed: total_files={total_files} > 120"
    assert max_mb <= 5.0, f"Gate 1 failed: max_single_file_mb={max_mb:.2f} > 5.0"
    for p in files:
        assert p.name != ".env", f"Gate 1 failed: .env leaked ({p})"
        assert p.suffix.lower() not in FORBIDDEN_EXTS, f"Gate 1 failed: forbidden extension {p}"
    print(f"[GATE 1 PASS] Budget ok: {total_files} files, {total_mb:.3f} MB, max={max_mb:.3f} MB")

    # Gate 2: OWASP / GitHub Push Protection Regex & Personal Path Scan
    patterns = _build_secret_patterns()
    leaks = []
    for p in files:
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        try:
            txt = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, rx in patterns:
            if rx.search(txt):
                leaks.append((p.relative_to(TARGET).as_posix(), label))
    assert not leaks, f"Gate 2 failed: security/path leaks found: {leaks}"
    print("[GATE 2 PASS] OWASP/Secret & Personal-Path Scan: 0 hits")

    # Gate 3: 3-Way Isolation Verification
    forbidden_paths = [
        "AAIML 2027",
        "paper_overleaf",
        "Dataset",
        "memory.md",
        "rtsp_pipeline.py",
        "tests/test_task_b2_multiseed.py",
        "tests/test_rtsp_bipartite_filter.py",
    ]
    for fp in forbidden_paths:
        assert not (TARGET / fp).exists(), f"Gate 3 failed: forbidden path present: {fp}"
    opt_path = TARGET / "Output" / "task3_eval_opt" / "optimization_benchmark_table.json"
    opt_data = json.loads(opt_path.read_text(encoding="utf-8"))
    chv_rows = [r for r in opt_data.get("records", []) if "CHV" in str(r.get("model", "")).upper()]
    assert not chv_rows, f"Gate 3 failed: legacy CHV records leaked into optimization JSON: {chv_rows}"
    print("[GATE 3 PASS] 3-Way Isolation verified (0 AAIML/Dataset/memory/CHV7 contamination)")

    # Clean subprocess environment simulating fresh GitHub Actions runner
    clean_env = os.environ.copy()
    clean_env["MODEL_PATH"] = "ci_stub_missing.pt"
    clean_env["CUDA_VISIBLE_DEVICES"] = "-1"

    # Gate 4: compileall inside _publish_ai_capstone/
    r_comp = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "src", "tests", "scripts"],
        cwd=str(TARGET),
        env=clean_env,
        capture_output=True,
        text=True,
    )
    assert r_comp.returncode == 0, f"Gate 4 compileall failed:\n{r_comp.stdout}\n{r_comp.stderr}"
    print("[GATE 4 PASS] Clean-Clone compileall exit 0")

    # Gate 5: Weightless CPU pytest inside _publish_ai_capstone/
    r_pytest = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=str(TARGET),
        env=clean_env,
        capture_output=True,
        text=True,
    )
    assert r_pytest.returncode == 0, f"Gate 5 Clean-Clone pytest failed:\n{r_pytest.stdout}\n{r_pytest.stderr}"
    summary_line = [ln for ln in r_pytest.stdout.splitlines() if "passed" in ln]
    print(f"[GATE 5 PASS] Clean-Clone pytest: {summary_line[-1] if summary_line else 'OK'}")

    # Gate 6: Docker Compose validation inside _publish_ai_capstone/
    for yml in ("docker-compose.enterprise.yml", "docker-compose.edge-simulation.yml"):
        r_dc = subprocess.run(
            ["docker", "compose", "-f", yml, "config", "-q"],
            cwd=str(TARGET),
            capture_output=True,
            text=True,
        )
        assert r_dc.returncode == 0, f"Gate 6 compose config failed for {yml}:\n{r_dc.stderr}"
    print("[GATE 6 PASS] Docker Compose configs validated inside Clean-Clone")

    # Gate 7: Clean Orphan Git Repository Initialization & Hygiene Check
    if init_git:
        git_dir = TARGET / ".git"
        if git_dir.exists():
            _rmtree_force(git_dir)
        cmds = [
            ["git", "init", "-b", "main"],
            ["git", "add", "-A"],
            ["git", "commit", "-m", "feat: release AEGIS-CCTV Enterprise VMS v2.3 (GDUT-HWD 5-Class, 6-Microservice Stack, Review 3 Audit Bundle)"],
        ]
        for c in cmds:
            r_g = subprocess.run(c, cwd=str(TARGET), capture_output=True, text=True)
            assert r_g.returncode == 0, f"Gate 7 git command failed ({c}): {r_g.stdout}\n{r_g.stderr}"

        log_res = subprocess.run(["git", "log", "--oneline"], cwd=str(TARGET), capture_output=True, text=True)
        st_res = subprocess.run(["git", "status", "-s"], cwd=str(TARGET), capture_output=True, text=True)
        rem_res = subprocess.run(["git", "remote", "-v"], cwd=str(TARGET), capture_output=True, text=True)
        commits = [ln for ln in log_res.stdout.splitlines() if ln.strip()]
        assert len(commits) == 1, f"Gate 7 failed: expected 1 clean orphan commit, got {commits}"
        assert st_res.stdout.strip() == "", f"Gate 7 failed: dirty status {st_res.stdout}"
        assert rem_res.stdout.strip() == "", f"Gate 7 failed: unexpected remote {rem_res.stdout}"
        print(f"[GATE 7 PASS] Orphan Git repo clean: {commits[0]} | status=clean | remote=none")


def main():
    parser = argparse.ArgumentParser(description="Clean-Room Public Export & 7-Gate Validator for AI-Capstone")
    parser.add_argument("--verify", action="store_true", help="Run all 7 audit gates on _publish_ai_capstone/")
    parser.add_argument("--init-git", action="store_true", help="Initialize clean orphan Git repo inside _publish_ai_capstone/")
    args = parser.parse_args()

    sync_export(init_git=args.init_git)
    if args.verify or args.init_git:
        verify_export(init_git=args.init_git)


if __name__ == "__main__":
    main()
