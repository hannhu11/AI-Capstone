"""Automated 3-layer consistency + SHWD CSV + README-JSON verification guard for Capstone Thesis (v3.2.0).

Verifies:
  Layer 1: LaTeX chapter sources (.tex) & BibTeX (references.bib)
  Layer 2: Compiled PDF (thesis_report/Capstone_Thesis_Report.pdf via PyMuPDF: <=100p, 0 prose hyphens, 0 mid-word code breaks)
  Layer 3: Exported DOCX (thesis_report/Capstone_Thesis_Report.docx via python-docx: multi-para Listings, spacing)
  Layer 4: Raw SHWD_V1/outputs/*.csv 3-seed means/SDs + README.md numerical parity against benchmark JSON artifacts
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import statistics
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
    "<2%",
    "< 2%",
    r"<2\%",
    r"< 2\%",
    "below 2%",
    r"below $2\%$",
    r"below 2\%",
    "lower variance",
    "HHW variance",
    "minimal variance",
    "giảm phương sai",
    "81%--90%",
    "81%–90%",
    r"81\%--90\%",
    r"81\%\text{--}90\%",
    "Requirementsfor",
    "Capstone_Thesis_Report_FA26AI16",
]

BANNED_REGEXES = [
    re.compile(r"\bV8F\b"),
    re.compile(r"\bV8S\b"),
    re.compile(r"\bC0\b"),
    re.compile(r"\bC3\b"),
]

# Detect duplicate section/prefix nouns (e.g., "Appendix Appendix A", "Table Table II.1")
DUP_WORD_RX = re.compile(r"\b(Appendix|Chapter|Table|Figure|Section)\s+\1\b", re.I)
# ch9_appendices.tex formats \thesection as "Appendix A", so Appendix~\ref{app:...} doubles the prefix
APP_DUP_TEX_RX = re.compile(r"\bAppendix[~ ]\\ref\{app:")

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
    "<2%",
    "< 2%",
    "below 2%",
    "lower variance",
    "HHW variance",
    "minimal variance",
    "giảm phương sai",
    "81%--90%",
    "81%–90%",
    r"81\%--90\%",
    r"81\%\text{--}90\%",
    "Requirementsfor",
    "Capstone_Thesis_Report_FA26AI16",
    "3× 3",
    "640× 640",
]

REQUIRED_EVIDENCE_TOKENS = [
    "90.94",
    "58.63",
    "88.94",
    "58.08",
    "89.76",
    "59.08",
    "72%–84%",
    "Note on T4 Timing Protocols",
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
        m_dup = DUP_WORD_RX.search(txt)
        assert not m_dup, f"Duplicate structural noun {m_dup.group(0)!r} found in {p.name}"
        m_adup = APP_DUP_TEX_RX.search(txt)
        assert not m_adup, f"Prefix-doubling {m_adup.group(0)!r} found in {p.name}; use \\ref{{app:...}}"
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
    dup_hits = [(i + 1, m.group(0)) for i, txt in enumerate(page_texts) if (m := DUP_WORD_RX.search(txt))]
    assert not dup_hits, f"Duplicate structural noun found in PDF: {dup_hits}"
    for req in REQUIRED_EVIDENCE_TOKENS:
        assert req in full_pdf, f"Required v3.2 token {req!r} missing from PDF text"

    # Check zero auto-hyphenated prose line breaks and zero mid-word filename breaks
    prose_hyphens = []
    mid_code = []
    for pno, txt in enumerate(page_texts, 1):
        lines = [ln.rstrip() for ln in txt.splitlines() if ln.strip()]
        for li in range(len(lines) - 1):
            l1, l2 = lines[li], lines[li + 1].lstrip()
            if l1.endswith("-") and len(l1) >= 2 and l1[-2].isalpha():
                w1 = re.split(r"[\s(\[/\"']+", l1)[-1]
                w2 = re.split(r"[\s)\],.;:\"/']+", l2)[0]
                if not any(s in l1.lower() or s in w1.lower() for s in ("kaggle.com", "github.com", "doi.org", "gdut-", "-protocol-")):
                    prose_hyphens.append((pno, f"{w1}|{w2}"))
            tw1 = re.split(r"\s+", l1)[-1].lower()
            tw2 = re.split(r"\s+", l2)[0].lower()
            if (
                (tw1.endswith("databa") and tw2.startswith("se"))
                or (tw1.endswith("switc") and tw2.startswith("h_"))
                or (tw1.endswith("ente") and tw2.startswith("rprise"))
                or (tw1.endswith("feedba") and tw2.startswith("ck"))
                or (tw1.endswith("template") and tw2.startswith("s/"))
            ):
                mid_code.append((pno, f"{tw1}|{tw2}"))
    assert not prose_hyphens, f"Auto-hyphenated prose line breaks found in PDF: {prose_hyphens[:10]}"
    assert not mid_code, f"Mid-word filename/code breaks found in PDF: {mid_code[:10]}"
    print(f"[LAYER 2 PASS] PDF verified via PyMuPDF ({pages} pages <= 100, 0 banned/dup, 0 prose-hyphens, 0 mid-code breaks).")


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
    m_dup_docx = DUP_WORD_RX.search(full_docx)
    assert not m_dup_docx, (
        f"Duplicate structural noun {m_dup_docx.group(0)!r} leaked into DOCX around: "
        f"{full_docx[max(0, m_dup_docx.start() - 30):m_dup_docx.start() + 60].replace(chr(10), ' ')!r}"
    )
    for req in REQUIRED_EVIDENCE_TOKENS:
        assert req in full_docx, f"Required v3.2 token {req!r} missing from DOCX text"

    listing_para_counts = []
    for tbl in doc.tables:
        if len(tbl.rows) == 1 and len(tbl.columns) == 1:
            shd_els = tbl.rows[0].cells[0]._tc.xpath(".//w:shd[@w:fill='F6F8FA']")
            if shd_els:
                listing_para_counts.append(len(tbl.rows[0].cells[0].paragraphs))
    assert len(listing_para_counts) >= 6, f"Expected >= 6 Listings in DOCX, found {len(listing_para_counts)}"
    assert all(cnt >= 5 for cnt in listing_para_counts), (
        f"Collapsed Listing paragraphs in DOCX: {listing_para_counts}"
    )
    print(
        f"[LAYER 3 PASS] DOCX verified via python-docx ({len(doc.tables)} tables, "
        f"{len(doc.paragraphs)} paras, listings_paras={listing_para_counts}, 0 leaks)."
    )


def _verify_readme_against_json() -> None:
    readme_path = ROOT / "README.md"
    if not readme_path.exists():
        print("[README_JSON SKIP] README.md not present.")
        return
    readme_txt = readme_path.read_text(encoding="utf-8")
    assert "12.37" not in readme_txt, "Stale 12.37 ms latency value still present in README.md!"

    e2e_path = ROOT / "Output" / "task3_eval_opt" / "gdut_vms_end2end_reid_benchmark.json"
    if e2e_path.exists():
        e2e_data = json.loads(e2e_path.read_text(encoding="utf-8"))
        stages = e2e_data.get("stages", [])
        assert len(stages) >= 5, f"Expected >= 5 E2E stages in {e2e_path.name}"
        expected_e2e = [f"{float(st['latency_ms']):.2f}" for st in stages[:5]]
        paired_delta = f"{float(stages[4]['paired_delta_mean_ms']):.2f}"
        for tok in expected_e2e + [paired_delta]:
            assert tok in readme_txt, f"README.md missing exact E2E JSON metric {tok!r} from {e2e_path.name}"

    opt_path = ROOT / "Output" / "task3_eval_opt" / "optimization_benchmark_table.json"
    if opt_path.exists():
        opt_data = json.loads(opt_path.read_text(encoding="utf-8"))
        recs = {r.get("stage"): r for r in opt_data.get("records", [])}
        expected_opt = [
            f"{float(recs['Before (Loop BiFormer FP32)']['latency_ms']):.2f}",
            f"{float(recs['After-1 (Vec BiFormer + RepConv FP32)']['latency_ms']):.2f}",
            f"{float(recs['After-2 (+ CUDA AMP FP16)']['latency_ms']):.2f}",
            f"{float(recs['After-3 (Micro-Batch N=4 AMP FP16)']['latency_ms']):.2f}",
            f"{float(recs['After-3 (Micro-Batch N=4 AMP FP16)']['fps']):.1f}",
        ]
        for tok in expected_opt:
            assert tok in readme_txt, f"README.md missing exact optimization JSON metric {tok!r} from {opt_path.name}"

    print("[README_JSON_PASS] README.md matches E2E Re-ID & optimization JSON artifacts (0 stale values).")


def _verify_shwd_v1_raw_csvs() -> None:
    """Computes live 3-seed means/SDs (ddof=1) from SHWD_V1/outputs/*.csv and verifies parity with ch6_results.tex."""
    out_dir = ROOT / "SHWD_V1" / "outputs"
    ch6 = (CHAPTERS_DIR / "ch6_results.tex").read_text(encoding="utf-8")
    shwd_rows = []
    for seed in (42, 1337, 2026):
        with (out_dir / f"shwd_test_ablation_seed{seed}.csv").open(encoding="utf-8") as fh:
            shwd_rows.extend(list(csv.DictReader(fh)))
    t4_rows = []
    for seed in (42, 1337, 2026):
        with (out_dir / f"t4_latency_seed{seed}.csv").open(encoding="utf-8") as fh:
            t4_rows.extend(list(csv.DictReader(fh)))
    with (out_dir / "v1_hhw_all_3seeds.csv").open(encoding="utf-8") as fh:
        hhw_rows = list(csv.DictReader(fh))

    for cfg in ("A0", "A1", "A2", "A3", "A4"):
        s_m50 = [float(r["mAP50"]) for r in shwd_rows if r["ablation_id"] == cfg]
        s_fp16 = [float(r["pyt_fp16_ms"]) for r in t4_rows if r["ablation_id"] == cfg]
        h_2c50 = [float(r["mAP50"]) for r in hhw_rows if r["ablation_id"] == cfg and r["hhw_eval"] == "hhw_2class_test"]
        h_hat50 = [float(r["hat_AP50"]) for r in hhw_rows if r["ablation_id"] == cfg and r["hhw_eval"] == "hhw_hatonly_test"]
        assert len(s_m50) == 3 and len(s_fp16) == 3 and len(h_2c50) == 3 and len(h_hat50) == 3, (
            f"Unexpected row counts for {cfg}: s={len(s_m50)}, fp16={len(s_fp16)}, h2c={len(h_2c50)}, hhat={len(h_hat50)}"
        )
        for arr in (s_m50, s_fp16, h_2c50, h_hat50):
            mean_v = f"{statistics.mean(arr):.2f}"
            assert mean_v in ch6, f"Missing live 3-seed V1 CSV mean {mean_v} for {cfg} in ch6_results.tex!"
    print("[SHWD_V1_CSV_PASS] Live 3-seed SHWD/HHW/T4 means from raw SHWD_V1/outputs/*.csv match ch6_results.tex.")


def verify_thesis_consistency() -> None:
    _verify_layer1_sources()
    _verify_layer2_pdf()
    _verify_layer3_docx()
    _verify_readme_against_json()
    _verify_shwd_v1_raw_csvs()
    print("[THESIS_CONSISTENCY_PASS] All 3 layers + SHWD_V1 raw CSVs + README JSON parity verified (v3.2.0).")


if __name__ == "__main__":
    try:
        verify_thesis_consistency()
    except AssertionError as exc:
        print(f"[THESIS_CONSISTENCY_FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
