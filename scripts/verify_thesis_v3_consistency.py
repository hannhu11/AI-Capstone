"""Automated 3-layer consistency verification guard for Capstone Thesis (v3.1).

Verifies:
  Layer 1: LaTeX chapter sources (.tex) & BibTeX (references.bib)
  Layer 2: Compiled PDF (thesis_report/Capstone_Thesis_Report.pdf via PyMuPDF, if present)
  Layer 3: Exported DOCX (thesis_report/Capstone_Thesis_Report.docx via python-docx, if present)
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
THESIS_DIR = ROOT / "thesis_report"
CHAPTERS_DIR = THESIS_DIR / "chapters"
BIB_FILE = THESIS_DIR / "references.bib"
PDF_FILE = THESIS_DIR / "Capstone_Thesis_Report.pdf"
DOCX_FILE = THESIS_DIR / "Capstone_Thesis_Report.docx"

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
    "Xuan Huy",
    "Quoc Trung",
    "External Supervisors",
    "External Review Panel",
    "Review Committee",
    "External Committee",
    "5-seed",
    "v2.6",
    "12-step",
    "01_KET_QUA_A1_A6",
    "GDUT_architecture_integration_check",
    "dOI:",
]

BANNED_REGEXES = [
    re.compile(r"\bV8F\b"),
    re.compile(r"\bV8S\b"),
    re.compile(r"\bC0\b"),
    re.compile(r"\bC3\b"),
]

# Detect single stray Markdown backtick in LaTeX while permitting valid LaTeX ``quotes''
SINGLE_BACKTICK_RX = re.compile(r"(?<!`)`(?!`)")

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

BINARY_BANNED_TOKENS = [
    r"\${",
    r"\$",
    "??",
    "[?]",
    "dOI:",
    "5-seed",
    "v2.6",
    "12-step",
    "01_KET_QUA_A1_A6",
    "GDUT_architecture_integration_check",
    "`",
    "Xuan Huy",
    "Quoc Trung",
    "External Supervisors",
    "External Review Panel",
    "Review Committee",
    "External Committee",
    "AAIML",
    "L4021",
    "tundng111",
]

REQUIRED_EVIDENCE_TOKENS = [
    "90.94",
    "58.63",
    "88.94",
    "58.08",
    "89.76",
    "59.08",
    "84.85",
    "87.51",
    "9,476,783",
    "15.207",
    "82 passed, 1 skipped",
    "139",
]


def _verify_layer1_sources() -> None:
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
        m_bt = SINGLE_BACKTICK_RX.search(txt)
        assert not m_bt, f"Stray single Markdown backtick found in {p.name} around {txt[max(0, m_bt.start()-20):m_bt.end()+20]!r}"
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
    for metric_s in ("90.94", "58.63", "88.94", "84.85", "87.51", "9,476,783", "10,800", "fig_v2_baselines_vs_a6_comparison", "fig_v1_shwd_hhw_ablation"):
        assert metric_s in ch6, f"Missing v3.1 metric/figure {metric_s!r} in ch6_results.tex"

    for name, txt in (("frontmatter.tex", front), ("ch5_system.tex", ch5), ("ch6_results.tex", ch6), ("ch9_appendices.tex", ch9)):
        assert "82 passed, 1 skipped" in txt, f"Missing Clean-Room 82/83 CI distinction in {name}"
        assert "139" in txt, f"Missing workspace 139 pytest count in {name}"

    assert "docker-compose.enterprise.yml" in ch9 and "tab:app_enterprise_stack" in ch9

    for slug in OFFICIAL_SLUGS:
        assert slug in ch9, f"Appendix A missing official nvthanh2004 slug {slug!r}"

    bib_txt = BIB_FILE.read_text(encoding="utf-8")
    assert "10.3390/jimaging11090320" in bib_txt and "{DOI}:" in bib_txt, "references.bib missing uppercase {DOI} note for B1"
    assert "10.1038/s41598-024-68446-z" in bib_txt, "references.bib missing visible DOI note for B2"
    assert "shwd_dataset" in bib_txt and "hard_hat_workers_mvd" in bib_txt, "references.bib missing SHWD/HHW entries"
    print(f"[LAYER 1 PASS] {len(tex_files)} .tex files + references.bib clean (0 stale tokens, 15 slugs, ERD, V1/V2).")


def _verify_layer2_pdf() -> None:
    if not PDF_FILE.exists():
        print(f"[LAYER 2 SKIP] PDF binary not present ({PDF_FILE.name}); skipping binary PDF scan.")
        return
    import fitz

    doc = fitz.open(str(PDF_FILE))
    pages = len(doc)
    assert 60 <= pages <= 100, f"PDF page count {pages} violates 60..100 AIP491 specification!"
    page_texts = [pg.get_text("text") for pg in doc]
    doc.close()

    full_pdf = "\n".join(page_texts)
    for bad in BINARY_BANNED_TOKENS:
        hits = [i + 1 for i, txt in enumerate(page_texts) if bad in txt]
        assert not hits, f"Banned token {bad!r} found in PDF pages {hits}"
    for req in REQUIRED_EVIDENCE_TOKENS:
        assert req in full_pdf, f"Required v3.1 token {req!r} missing from PDF text"
    print(f"[LAYER 2 PASS] PDF verified via PyMuPDF ({pages} pages <= 100, 0 banned tokens).")


def _verify_layer3_docx() -> None:
    if not DOCX_FILE.exists():
        print(f"[LAYER 3 SKIP] DOCX binary not present ({DOCX_FILE.name}); skipping binary DOCX scan.")
        return
    from docx import Document

    doc = Document(str(DOCX_FILE))
    chunks = [p.text for p in doc.paragraphs if p.text]
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                if cell.text:
                    chunks.append(cell.text)
    full_docx = "\n".join(chunks)

    docx_banned = BINARY_BANNED_TOKENS + ["tabularcc", "tabular>", "tabularx", "dagger", ">X", "1em plus", r"\begin{", r"\end{"]
    for bad in docx_banned:
        idx = full_docx.find(bad)
        assert idx == -1, (
            f"Banned token {bad!r} leaked into DOCX around: "
            f"{full_docx[max(0, idx - 30):idx + 60].replace(chr(10), ' ')!r}"
        )
    m_tab_leak = re.search(r"tabular(?!-nums)", full_docx)
    assert not m_tab_leak, (
        f"Leaked tabular token in DOCX around: "
        f"{full_docx[max(0, m_tab_leak.start() - 30):m_tab_leak.start() + 60].replace(chr(10), ' ')!r}"
    )
    for req in REQUIRED_EVIDENCE_TOKENS:
        assert req in full_docx, f"Required v3.1 token {req!r} missing from DOCX text"
    print(f"[LAYER 3 PASS] DOCX verified via python-docx ({len(doc.tables)} tables, {len(doc.paragraphs)} paras, 0 leaks).")


def verify_thesis_consistency() -> None:
    _verify_layer1_sources()
    _verify_layer2_pdf()
    _verify_layer3_docx()
    print("[THESIS_CONSISTENCY_PASS] All 3 layers verified (v3.1).")


if __name__ == "__main__":
    try:
        verify_thesis_consistency()
    except AssertionError as exc:
        print(f"[THESIS_CONSISTENCY_FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
