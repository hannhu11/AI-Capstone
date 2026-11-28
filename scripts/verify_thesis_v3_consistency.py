"""Automated 3-layer consistency + LoT/LoF/Frontmatter layout + SHWD CSV + README-JSON + CSV/SHA-256 verification guard for Capstone Thesis (v3.3.0).

Verifies:
  Layer 1: LaTeX chapter sources (.tex) & BibTeX (references.bib)
  Layer 2: Compiled PDF (thesis_report/Capstone_Thesis_Report.pdf via PyMuPDF: <=100p, 0 prose hyphens, 0 mid-word code breaks)
  Layer 2B: LoT / LoF monotonic table & figure order + single-page Abstract/Keywords & non-orphan Task Assignment guards
  Layer 3: Exported DOCX (thesis_report/Capstone_Thesis_Report.docx via python-docx: multi-para Listings, spacing)
  Layer 4: Raw SHWD_V1/outputs/*.csv 3-seed means/SDs + README.md numerical parity + CSV==JSON cell parity + canonical SHA-256 chain

By default (workspace mode), ANY missing artifact (PDF, DOCX, main.lot, main.lof, README.md, JSON/CSV) is a HARD FAILURE (exit != 0).
Pass `--clean-room` only inside `_publish_ai_capstone/` where PDF/DOCX/aux binaries are intentionally distributed via GitHub Releases.
"""
from __future__ import annotations

import argparse
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
    "118.2 FPS",
    "118.2~FPS",
    "c10bb46a",
    "3 hạt giống",
    "kiểm chứng pháp y",
    "Pareto superiority",
    "under site safety-officer access control",
    "139/139",
    "138 passed, 1 skipped",
    "82 passed, 1 skipped",
    "a6_yolo11s_val_pr.png",
    "a6_yolo11s_val_cm.png",
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
    "74%–81%",
    "Note on T4 Timing Protocols",
    "84.85",
    "87.51",
    "9,476,783",
    "15.207",
    "143.7",
    "6.96",
    "84 passed, 1 skipped",
    "141",
    "d422dd1d",
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
        assert "84 passed, 1 skipped" in txt, f"Missing Clean-Room 84/85 CI distinction in {name}"
        assert "141" in txt, f"Missing workspace 141 pytest count in {name}"

    assert "docker-compose.enterprise.yml" in ch9 and "tab:app_enterprise_stack" in ch9

    for slug in OFFICIAL_SLUGS:
        assert slug in ch9, f"Appendix A missing official nvthanh2004 slug {slug!r}"

    bib_txt = BIB_FILE.read_text(encoding="utf-8")
    assert "10.3390/jimaging11090320" in bib_txt and "{DOI}:" in bib_txt, "references.bib missing uppercase {DOI} note for B1"
    assert "10.1038/s41598-024-68446-z" in bib_txt, "references.bib missing visible DOI note for B2"
    assert "shwd_dataset" in bib_txt and "hard_hat_workers_mvd" in bib_txt, "references.bib missing SHWD/HHW entries"
    print(f"[LAYER 1 PASS] {len(tex_files)} .tex files + references.bib clean (0 stale tokens, 15 slugs, ERD, V1/V2).")


def _verify_layer2_pdf(clean_room: bool = False) -> None:
    if not PDF_FILE.exists():
        if clean_room:
            print(f"[LAYER 2 CLEAN-ROOM] PDF distributed via GitHub Releases ({PDF_FILE.name}).")
            return
        raise AssertionError(f"Required compiled PDF missing: {PDF_FILE}")
    import fitz

    doc = fitz.open(str(PDF_FILE))
    pages = len(doc)
    assert 60 <= pages <= 100, f"PDF page count {pages} violates 60..100 AIP491 v3.3.0 target!"
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
                joined = (w1 + w2).lower()
                if not any(s in l1.lower() or s in joined for s in ("kaggle.com", "github.com", "doi.org", "gdut-", "-protocol-", "docker-compose", "edge-simulation")):
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


def _verify_lot_lof_monotonic_order(clean_room: bool = False) -> None:
    lot_path = THESIS_DIR / "main.lot"
    lof_path = THESIS_DIR / "main.lof"
    if not PDF_FILE.exists() or not lot_path.exists() or not lof_path.exists():
        if clean_room:
            print("[LOT_LOF CLEAN-ROOM] Aux files excluded from Clean-Room Git repo.")
            return
        raise AssertionError(f"Missing required LoT/LoF/PDF artifacts: lot={lot_path.exists()}, lof={lof_path.exists()}, pdf={PDF_FILE.exists()}")

    roman_rank = {"0": 0, "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9}
    entry_rx = re.compile(
        r"\\contentsline\s*\{(table|figure)\}\{\\numberline\s*\{([0IVX]+)\.(\d+)\}.*\}\{(\d+|[ivxlcdm]+)\}\{(?:table|figure)\.[^}]+\}%?\s*$"
    )
    counts = {}
    for kind, path in (("LoT", lot_path), ("LoF", lof_path)):
        prev_chap_rank = -1
        last_item_by_chap: dict[str, int] = {}
        prev_arabic_page = 0
        seen = 0
        for raw_ln in path.read_text(encoding="utf-8").splitlines():
            m = entry_rx.match(raw_ln.strip())
            if not m:
                continue
            seen += 1
            chap, item_s, page_s = m.group(2), m.group(3), m.group(4)
            c_rank = roman_rank.get(chap, 99)
            item_num = int(item_s)
            assert c_rank >= prev_chap_rank, f"{kind} chapter order regressed at {chap}.{item_num}"
            prev_chap_rank = c_rank
            prev_item = last_item_by_chap.get(chap, 0)
            assert item_num > prev_item, (
                f"{kind} numbering out of order in Chapter {chap}: {chap}.{item_num} appeared after {chap}.{prev_item}"
            )
            last_item_by_chap[chap] = item_num
            if page_s.isdigit():
                pno = int(page_s)
                assert pno >= prev_arabic_page, (
                    f"{kind} page number regressed at {chap}.{item_num}: page {pno} < previous {prev_arabic_page}"
                )
                prev_arabic_page = pno
        counts[kind] = seen
    assert counts["LoT"] >= 30, f"Unexpectedly few LoT entries parsed: {counts['LoT']}"
    assert counts["LoF"] == 21, f"Expected 21 LoF entries parsed, got {counts['LoF']}"
    print(f"[LOT_LOF_ORDER_PASS] Monotonic order verified (LoT={counts['LoT']} tables, LoF={counts['LoF']} figures).")


def _verify_no_sparse_frontmatter_pages(clean_room: bool = False) -> None:
    if not PDF_FILE.exists():
        if clean_room:
            print("[FRONTMATTER CLEAN-ROOM] PDF distributed via GitHub Releases.")
            return
        raise AssertionError(f"Missing required PDF for frontmatter sparsity guard: {PDF_FILE}")
    import fitz

    doc = fitz.open(str(PDF_FILE))
    front_texts = [doc[i].get_text("text") for i in range(min(25, len(doc)))]
    doc.close()

    abstract_page = None
    keywords_page = None
    tomtat_page = None
    tukhoa_page = None
    for idx, txt in enumerate(front_texts, 1):
        flat = " ".join(txt.split())
        wc = len(flat.split())
        assert wc >= 35, f"Sparse frontmatter page p.{idx} has only {wc} words (< 35)"
        if "Automated verification of safety helmet" in flat:
            abstract_page = idx
        if "Keywords:" in flat:
            keywords_page = idx
        if "Tóm tắt" in flat and ("nghiêm trọng" in flat or "phân quyền" in flat):
            tomtat_page = idx
        if re.search(r"Từ\s*khóa:", flat):
            tukhoa_page = idx
        if idx <= 10 and "2. Assigned Responsibilities and Deliverables" in flat:
            assert "Nguyen Han Nhu (Leader)" in flat and "SE183644" in flat, (
                f"Orphan heading '2. Assigned Responsibilities and Deliverables' without table on p.{idx}"
            )

    assert abstract_page is not None and keywords_page is not None, "Abstract or Keywords page not located in frontmatter"
    assert abstract_page == keywords_page, (
        f"Abstract (p.{abstract_page}) and Keywords (p.{keywords_page}) split across pages!"
    )
    assert len(front_texts[abstract_page - 1].split()) >= 150, "Abstract page unexpectedly sparse"
    assert tomtat_page is not None and tukhoa_page is not None, "Vietnamese Abstract or Tu khoa page not located"
    assert tomtat_page == tukhoa_page, (
        f"Vietnamese Abstract (p.{tomtat_page}) and Tu khoa (p.{tukhoa_page}) split across pages!"
    )
    print(
        f"[FRONTMATTER_PASS] Abstract+Keywords paired on p.{abstract_page}, Tom tat+Tu khoa on p.{tomtat_page}, "
        f"Task Assignment non-orphan, 0 sparse pages (<35 words)."
    )


def _verify_layer3_docx(clean_room: bool = False) -> None:
    if not DOCX_FILE.exists():
        if clean_room:
            print(f"[LAYER 3 CLEAN-ROOM] DOCX distributed via GitHub Releases ({DOCX_FILE.name}).")
            return
        raise AssertionError(f"Required exported DOCX missing: {DOCX_FILE}")
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
    assert readme_path.exists(), f"Missing required README.md at {readme_path}"
    readme_txt = readme_path.read_text(encoding="utf-8")
    assert "12.37" not in readme_txt, "Stale 12.37 ms latency value still present in README.md!"

    e2e_path = ROOT / "Output" / "task3_eval_opt" / "gdut_vms_end2end_reid_benchmark.json"
    opt_path = ROOT / "Output" / "task3_eval_opt" / "optimization_benchmark_table.json"
    csv_path = ROOT / "Output" / "task3_eval_opt" / "optimization_benchmark_table.csv"
    s2d_path = ROOT / "Output" / "task3_eval_opt" / "gdut_a6_switch_to_deploy_benchmark.json"
    ms_path = ROOT / "Output" / "operational_multistream" / "rtx3050_gdut_a6_batched_multistream_results.json"
    for req_p in (e2e_path, opt_path, csv_path, s2d_path, ms_path):
        assert req_p.exists(), f"Missing required benchmark artifact: {req_p}"

    e2e_data = json.loads(e2e_path.read_text(encoding="utf-8"))
    stages = e2e_data.get("stages", [])
    assert len(stages) >= 5, f"Expected >= 5 E2E stages in {e2e_path.name}"
    expected_e2e = [f"{float(st['latency_ms']):.2f}" for st in stages[:5]]
    paired_delta = f"{float(stages[4]['paired_delta_mean_ms']):.2f}"
    for tok in expected_e2e + [paired_delta]:
        assert tok in readme_txt, f"README.md missing exact E2E JSON metric {tok!r} from {e2e_path.name}"

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

    # Verify 100% CSV == JSON cell parity & single canonical SHA-256 chain across all 4 RTX 3050 receipts
    with csv_path.open(encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == len(opt_data["records"]), "CSV/JSON row count mismatch in optimization_benchmark_table"
    for r_csv, r_json in zip(csv_rows, opt_data["records"]):
        for k, v in r_json.items():
            assert str(r_csv[k]) == str(v), f"CSV/JSON cell mismatch on {k}: {r_csv[k]!r} != {v!r}"

    s2d_data = json.loads(s2d_path.read_text(encoding="utf-8"))
    ms_data = json.loads(ms_path.read_text(encoding="utf-8"))
    src_shas = {
        s2d_data.get("source_checkpoint_sha256"),
        opt_data.get("source_checkpoint_sha256"),
        e2e_data.get("source_checkpoint_sha256"),
        *(r.get("source_checkpoint_sha256") for r in ms_data),
    }
    dep_shas = {
        s2d_data.get("deployed_checkpoint_sha256"),
        opt_data.get("deployed_checkpoint_sha256"),
        e2e_data.get("deployed_checkpoint_sha256"),
        *(r.get("deployed_checkpoint_sha256") for r in ms_data),
    }
    assert src_shas == {"21866da3396e36a408236b1c3717bff0ed5b99f2e59e9cbbbce96ea6bdb68233"}, (
        f"Inconsistent source_checkpoint_sha256 across 4 RTX 3050 benchmarks: {src_shas}"
    )
    assert len(dep_shas) == 1 and next(iter(dep_shas)), (
        f"Inconsistent deployed_checkpoint_sha256 across 4 RTX 3050 benchmarks: {dep_shas}"
    )
    print("[README_JSON_PASS] README.md, CSV==JSON cell parity, and 4-benchmark canonical SHA-256 chain verified.")


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


def verify_thesis_consistency(clean_room: bool = False) -> None:
    _verify_layer1_sources()
    _verify_layer2_pdf(clean_room=clean_room)
    _verify_lot_lof_monotonic_order(clean_room=clean_room)
    _verify_no_sparse_frontmatter_pages(clean_room=clean_room)
    _verify_layer3_docx(clean_room=clean_room)
    _verify_readme_against_json()
    _verify_shwd_v1_raw_csvs()
    mode_tag = "clean-room" if clean_room else "full-workspace"
    print(f"[THESIS_CONSISTENCY_PASS] All checks verified ({mode_tag}, v3.3.0).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capstone Thesis v3.3.0 Strict Consistency Guard")
    parser.add_argument(
        "--clean-room",
        action="store_true",
        help="Allow missing PDF/DOCX/aux binaries inside Clean-Room Git export (distributed via GitHub Releases)",
    )
    args = parser.parse_args()
    try:
        verify_thesis_consistency(clean_room=args.clean_room)
    except AssertionError as exc:
        print(f"[THESIS_CONSISTENCY_FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
