"""Automated consistency verification guard for Capstone Thesis LaTeX chapters (v3.0)."""
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS_DIR = ROOT / "thesis_report" / "chapters"
BIB_FILE = ROOT / "thesis_report" / "references.bib"

BANNED_SUBSTRINGS = [
    "146.9 FPS",
    "146.9~FPS",
    "1.89x",
    r"1.89\times",
    "doughnut",
    "Warm-Theme",
    "Warm-Sand",
    "zero person inputs",
    "0x5753434D",
    "0x57 0x53 0x43 0x4D",
    "web/{app.py, gateway_client.py, database.py",
    r"\${",
    "130 passed, 1 skipped in weightless GitHub Actions",
    "74 passed, 1 skipped",
    "74/75",
    "131/131",
    "78 passed, 1 skipped",
    "78/79",
    "135/135",
    "134 passed, 1 skipped",
    r"\nolinkurl{ruff check",
    r"\nolinkurl{python -m compileall",
    "tundng111",
    "dngtrnphmtunse183674",
    "026a2afb",
    "98-FN",
    "98 FN",
    "Pair-3407",
    "pair-3407",
    "b1_dung",
    "fig_gdut_baselines_b1_b2_audit",
    "BASELINE_AUDIT_B1_B2_REVIEW3",
    "AAIML",
    "L4021",
    "conference manuscript",
    "84.36",
    "88.19",
    "FINDING_5",
    "90.2447",
]

BANNED_REGEXES = [
    re.compile(r"\bV8F\b"),
    re.compile(r"\bV8S\b"),
    re.compile(r"\bC0\b"),
    re.compile(r"\bC3\b"),
]

VIET_DIACRITIC_RX = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
)

OFFICIAL_SLUGS = [
    "gdut-dataset-preparation",
    "gdut-structural-re-parameterize-yolo-architecture1",
    "gdut-structural-re-parameterize-yolo-architecture2",
    "gdut-stage2-ablation-setup-full-train-run-a0",
    "gdut-stage2-ablation-setup-full-train-run-a1",
    "gdut-stage2-ablation-setup-full-train-run-a2",
    "gdut-stage2-ablation-setup-full-train-run-a3",
    "gdut-stage2-ablation-setup-full-train-run-a4",
    "gdut-stage2-ablation-setup-full-train-run-a5",
    "gdut-stage2-ablation-setup-full-train-run-a6",
    "gdut-final-test-evaluation",
    "gdut-stage1-yolo11n-for-dcrcf",
    "gdut-baseline-yolov8-dwr-aspp-nwd-stage1-finetune",
    "gdut-baseline-yolo11-dcrcf-stage1-finetune",
    "gdut-baseline-and-a6-stage1-protocol-comparison",
]


def verify_thesis_consistency() -> None:
    tex_files = sorted(CHAPTERS_DIR.glob("*.tex"))
    assert tex_files, f"No .tex files found under {CHAPTERS_DIR}"

    all_text = {}
    for p in tex_files:
        txt = p.read_text(encoding="utf-8")
        all_text[p.name] = txt
        for bad in BANNED_SUBSTRINGS:
            assert bad not in txt, f"Banned stale string {bad!r} found in {p.name}"
        for rx in BANNED_REGEXES:
            m = rx.search(txt)
            assert not m, f"Banned token {m.group(0)!r} found in {p.name}"
        if p.name != "frontmatter.tex":
            m_vi = VIET_DIACRITIC_RX.search(txt)
            assert not m_vi, f"Unexpected Vietnamese character {m_vi.group(0)!r} in body file {p.name}"

    front = all_text["frontmatter.tex"]
    ch3 = all_text["ch3_sota.tex"]
    ch4 = all_text["ch4_methodology.tex"]
    ch5 = all_text["ch5_system.tex"]
    ch6 = all_text["ch6_results.tex"]
    ch9 = all_text["ch9_appendices.tex"]

    assert r"\section*{Tóm tắt}" in front, "frontmatter.tex missing Vietnamese Abstract (Tóm tắt)"
    assert "fig_baseline_b1_yolo11n_dcrcf" in ch3 and "fig_baseline_b2_yolov8s_dwr_aspp_nwd" in ch3, (
        "ch3_sota.tex must embed both baseline diagrams"
    )
    assert "Proposed V1" in ch4 and "SHWD" in ch4, "ch4_methodology.tex missing Proposed V1 architecture/protocol"

    assert "src/db/database.py" in ch5 and "operator_feedback" in ch5, "ch5_system.tex missing src/db/database.py / operator_feedback"
    assert "fig:ch5_db_erd" in ch5, "ch5_system.tex missing fig:ch5_db_erd TikZ ERD"
    assert "operator_feedback" in ch9 and "db/database.py" in ch9, "ch9_appendices.tex missing operator_feedback / db/database.py"
    assert "agnostic_nms=True" in ch5 or r"agnostic\_nms=True" in ch5, "ch5_system.tex missing agnostic_nms=True"
    assert "tests/conftest.py" in ch9, "ch9_appendices.tex missing tests/conftest.py isolation"

    app_txt = (ROOT / "src" / "web" / "app.py").read_text(encoding="utf-8")
    assert 'msg.get("bytes")' in app_txt, "src/web/app.py missing binary msg.get('bytes') handler"
    assert "0xFF 0xD8" in ch5 and "0xFF 0xD8" in ch9, "/ws/webcam docs must state raw binary JPEG (0xFF 0xD8)"

    for t4_val in ("8.0", "7.5", "8.3", "8.9", "8.5", "9.2", "7.4", "7.7", "8.4", "9.1"):
        assert t4_val in ch6, f"Missing verified T4 latency {t4_val} ms in ch6_results.tex"
    for metric_s in ("88.94", "84.85", "87.51", "9,476,783", "10,800", "fig_v2_baselines_vs_a6_comparison", "fig_v1_shwd_hhw_ablation"):
        assert metric_s in ch6, f"Missing v3.0 metric/figure {metric_s!r} in ch6_results.tex"

    for name, txt in (("frontmatter.tex", front), ("ch5_system.tex", ch5), ("ch6_results.tex", ch6), ("ch9_appendices.tex", ch9)):
        assert "82 passed, 1 skipped" in txt, f"Missing Clean-Room 82/83 CI distinction in {name}"
        assert "139" in txt, f"Missing workspace 139 pytest count in {name}"

    assert "docker-compose.enterprise.yml" in ch9 and "tab:app_enterprise_stack" in ch9

    for slug in OFFICIAL_SLUGS:
        assert slug in ch9, f"Appendix A missing official nvthanh2004 slug {slug!r}"

    bib_txt = BIB_FILE.read_text(encoding="utf-8")
    assert "10.3390/jimaging11090320" in bib_txt and "note" in bib_txt, "references.bib missing visible DOI note for B1"
    assert "10.1038/s41598-024-68446-z" in bib_txt, "references.bib missing visible DOI note for B2"
    assert "shwd_dataset" in bib_txt and "hard_hat_workers_mvd" in bib_txt, "references.bib missing SHWD/HHW entries"

    print(f"[THESIS_CONSISTENCY_PASS] Checked {len(tex_files)} chapters (v3.0): 0 stale hits, 15 slugs, ERD, V1/V2 & BibTeX verified.")


if __name__ == "__main__":
    try:
        verify_thesis_consistency()
    except AssertionError as exc:
        print(f"[THESIS_CONSISTENCY_FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
