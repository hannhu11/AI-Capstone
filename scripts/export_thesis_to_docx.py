"""
scripts/export_thesis_to_docx.py
================================
Converts the compiled v3.2.1 LuaLaTeX Capstone Thesis (`thesis_report/main.tex` + `chapters/*.tex`
+ `main.aux` + `main.bbl` + 300-DPI figures/TikZ renders) into a clean, reflowable,
Google-Docs-compatible `.docx` document mirroring 100% of the 98-page PDF content without TextBoxes.

Outputs (bit-for-bit identical):
  - thesis_report/Capstone_Thesis_Report.docx
  - ~/Downloads/Capstone_Thesis_Report.docx
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import docx
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Inches, Pt, RGBColor
import fitz  # PyMuPDF
from PIL import Image, ImageChops, ImageOps

ROOT = Path(__file__).resolve().parent.parent
THESIS_DIR = ROOT / "thesis_report"
CHAPTERS_DIR = THESIS_DIR / "chapters"
FIGURES_DIR = THESIS_DIR / "figures"
DOCX_OUT_REPO = THESIS_DIR / "Capstone_Thesis_Report.docx"
DOCX_OUT_DOWNLOADS = Path.home() / "Downloads" / "Capstone_Thesis_Report.docx"

COLOR_FPT_BLUE = RGBColor(15, 76, 129)
COLOR_FPT_ORANGE = RGBColor(234, 88, 12)
COLOR_DARK_TEXT = RGBColor(20, 24, 31)
COLOR_MUTED = RGBColor(85, 95, 110)
COLOR_LINK = RGBColor(15, 76, 129)
COLOR_CODE = RGBColor(25, 32, 44)

HEADER_HEX = "E8EEF5"
BORDER_HEX = "CBD5E1"
CODE_BG_HEX = "F6F8FA"
ALT_ROW_HEX = "F8FAFC"

ROMAN_CHAPTERS = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
    9: "IX",
}


# ---------------------------------------------------------------------------
# XML Helpers for Word & Google Docs Fidelity
# ---------------------------------------------------------------------------
def _set_cell_shading(cell, fill_hex: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tc_pr.append(shd)


def _set_cell_margins(cell, top_dxa: int = 75, bottom_dxa: int = 75, left_dxa: int = 105, right_dxa: int = 105) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement("w:tcMar")
    for side, val in (("top", top_dxa), ("bottom", bottom_dxa), ("left", left_dxa), ("right", right_dxa)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        tc_mar.append(el)
    tc_pr.append(tc_mar)


def _set_table_borders(table, border_hex: str = BORDER_HEX, sz: str = "4", none: bool = False) -> None:
    tbl_pr = table._tbl.tblPr
    val = "none" if none else "single"
    attrs = f'w:val="{val}"' if none else f'w:val="{val}" w:sz="{sz}" w:space="0" w:color="{border_hex}"'
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f"<w:top {attrs}/><w:left {attrs}/><w:bottom {attrs}/>"
        f"<w:right {attrs}/><w:insideH {attrs}/><w:insideV {attrs}/>"
        "</w:tblBorders>"
    )
    tbl_pr.append(borders)


def _set_row_cant_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))


def _set_row_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:tblHeader"))


def _set_cell_width_dxa(cell, w_dxa: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    for old in tc_pr.findall(qn("w:tcW")):
        tc_pr.remove(old)
    tc_w = OxmlElement("w:tcW")
    tc_w.set(qn("w:w"), str(max(240, int(w_dxa))))
    tc_w.set(qn("w:type"), "dxa")
    tc_pr.append(tc_w)


def _apply_fixed_table_widths(table, widths_cm: List[float]) -> List[int]:
    """Enforces fixed table layout, tblW, and tblGrid column widths in dxa for Word & Google Docs."""
    col_dxas = [max(320, int(round(w * 566.929))) for w in widths_cm]
    total_dxa = sum(col_dxas)
    tbl = table._tbl
    tbl_pr = tbl.tblPr

    for tag in ("w:tblW", "w:tblLayout"):
        for old in tbl_pr.findall(qn(tag)):
            tbl_pr.remove(old)

    tbl_w = OxmlElement("w:tblW")
    tbl_w.set(qn("w:w"), str(total_dxa))
    tbl_w.set(qn("w:type"), "dxa")
    # Place tblW near start (right after tblStyle if present) for strict OOXML schema ordering
    tbl_style = tbl_pr.find(qn("w:tblStyle"))
    if tbl_style is not None:
        tbl_pr.insert(list(tbl_pr).index(tbl_style) + 1, tbl_w)
    else:
        tbl_pr.insert(0, tbl_w)

    tbl_layout = OxmlElement("w:tblLayout")
    tbl_layout.set(qn("w:type"), "fixed")
    tbl_pr.append(tbl_layout)

    for old_grid in tbl.findall(qn("w:tblGrid")):
        tbl.remove(old_grid)
    tbl_grid = OxmlElement("w:tblGrid")
    for w_dxa in col_dxas:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(w_dxa))
        tbl_grid.append(grid_col)
    tbl_pr_idx = list(tbl).index(tbl_pr)
    tbl.insert(tbl_pr_idx + 1, tbl_grid)
    return col_dxas


def _set_paragraph_bottom_border(paragraph, color_hex: str = "0F4C81", sz: str = "12") -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="{sz}" w:space="4" w:color="{color_hex}"/></w:pBdr>'
    )
    p_pr.append(p_bdr)


def _add_page_number_field(run) -> None:
    fld_begin = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls("w"))
    instr = parse_xml(r'<w:instrText %s xml:space="preserve"> PAGE </w:instrText>' % nsdecls("w"))
    fld_sep = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls("w"))
    fld_end = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls("w"))
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(fld_end)


# ---------------------------------------------------------------------------
# Auxiliary & Vector Asset Pre-Rendering
# ---------------------------------------------------------------------------
def _extract_brace_arg(text: str, open_idx: int) -> Tuple[str, int]:
    """Given index at '{', returns (content_inside, index_after_matching_'}')."""
    if open_idx >= len(text) or text[open_idx] != "{":
        return "", open_idx
    depth = 0
    i = open_idx
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in ("{", "}"):
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], i + 1
        i += 1
    return text[open_idx + 1 :], len(text)


def _extract_bracket_arg(text: str, open_idx: int) -> Tuple[str, int]:
    if open_idx >= len(text) or text[open_idx] != "[":
        return "", open_idx
    depth = 0
    i = open_idx
    while i < len(text):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], i + 1
        i += 1
    return text[open_idx + 1 :], len(text)


def _extract_caption_arg(text: str, cmd_start: int) -> Tuple[str, int]:
    """Extracts optional [short] and mandatory {full} from \\caption starting at cmd_start."""
    pos = cmd_start + len(r"\caption")
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos < len(text) and text[pos] == "[":
        _, pos = _extract_bracket_arg(text, pos)
        while pos < len(text) and text[pos].isspace():
            pos += 1
    if pos < len(text) and text[pos] == "{":
        return _extract_brace_arg(text, pos)
    return "", pos


def parse_aux_file(aux_path: Path) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, str]]:
    """Returns (labels[key] -> (num, page), bibcites[key] -> num)."""
    txt = aux_path.read_text(encoding="utf-8", errors="replace")
    labels: Dict[str, Tuple[str, str]] = {}
    bibcites: Dict[str, str] = {}

    pos = 0
    while True:
        idx = txt.find(r"\newlabel{", pos)
        if idx == -1:
            break
        key, p1 = _extract_brace_arg(txt, idx + len(r"\newlabel"))
        p1 = p1
        if p1 < len(txt) and txt[p1] == "{":
            outer, p2 = _extract_brace_arg(txt, p1)
            if outer.startswith("{"):
                num, q1 = _extract_brace_arg(outer, 0)
                page, _ = _extract_brace_arg(outer, q1) if q1 < len(outer) and outer[q1] == "{" else ("", q1)
                labels[key] = (num.strip(), page.strip())
            pos = p2
        else:
            pos = idx + 9

    for m in re.finditer(r"\\bibcite\{([^}]+)\}\{([^}]+)\}", txt):
        bibcites[m.group(1).strip()] = m.group(2).strip()
    return labels, bibcites


def _parse_toc_entries(path: Path, kind: str) -> List[Tuple[str, str, str, str]]:
    """Parses .toc / .lot / .lof into [(level, number, title, page), ...]."""
    if not path.exists():
        return []
    entries: List[Tuple[str, str, str, str]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw.startswith(r"\contentsline"):
            continue
        idx = len(r"\contentsline")
        while idx < len(raw) and raw[idx].isspace():
            idx += 1
        c_type, idx = _extract_brace_arg(raw, idx)
        title_raw, idx = _extract_brace_arg(raw, idx)
        page_raw, idx = _extract_brace_arg(raw, idx)
        num_str = ""
        m_num = re.match(r"^\s*\\numberline\s*", title_raw)
        if m_num:
            num_str, end_n = _extract_brace_arg(title_raw, m_num.end())
            title_clean = title_raw[end_n:].strip()
        else:
            title_clean = title_raw.strip()
        title_clean = re.sub(r"\\ignorespaces\s*", "", title_clean)
        title_clean = re.sub(r"\\hspace\s*\{[^}]*\}", "", title_clean).strip()
        title_clean = re.sub(r"\\protect\s*", "", title_clean)
        title_clean = re.sub(r"(\\[a-zA-Z@]+)\s+\{", r"\1{", title_clean)
        entries.append((c_type.strip(), num_str.strip(), title_clean, page_raw.strip()))
    return entries


def _render_pdf_to_png(pdf_path: Path, out_png: Path, dpi: int = 300) -> Path:
    doc = fitz.open(str(pdf_path))
    try:
        pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_png))
        return out_png
    finally:
        doc.close()


def _crop_whitespace(img_path: Path, pad_px: int = 18) -> Path:
    """Tightly trims white outer margins from a high-DPI rasterized diagram while adding uniform padding."""
    im = Image.open(img_path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        im = im.crop(bbox)
    im = ImageOps.expand(im, border=pad_px, fill=(255, 255, 255))
    im.save(str(img_path))
    return img_path


def _render_tikz_to_png(tikz_code: str, out_png: Path, work_dir: Path, name: str, dpi: int = 300) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    # Render using exact packages/classes verified in thesis_report/main.tex, then tight-crop at 300 DPI
    tex_src = rf"""\documentclass[12pt,a4paper]{{report}}
\usepackage{{fontspec}}
\setmainfont{{Times New Roman}}
\setsansfont{{Arial}}
\setmonofont{{Consolas}}[Scale=0.84]
\usepackage[margin=1.2cm]{{geometry}}
\usepackage{{tikz}}
\usepackage[table]{{xcolor}}
\usepackage{{hyperref}}
\urlstyle{{same}}
\usetikzlibrary{{arrows.meta,positioning,shapes.geometric,fit,calc,backgrounds}}
\definecolor{{fptblue}}{{RGB}}{{15, 76, 129}}
\definecolor{{fptorange}}{{RGB}}{{234, 88, 12}}
\definecolor{{slatehdr}}{{RGB}}{{241, 245, 249}}
\pagestyle{{empty}}
\begin{{document}}
\noindent
{tikz_code.strip()}
\end{{document}}
"""
    tex_file = work_dir / f"{name}.tex"
    tex_file.write_text(tex_src, encoding="utf-8")
    r = subprocess.run(
        ["lualatex", "-interaction=nonstopmode", tex_file.name],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    pdf_out = work_dir / f"{name}.pdf"
    if r.returncode != 0 or not pdf_out.exists():
        raise RuntimeError(f"TikZ render compile failed for {name}:\n{r.stdout[-1500:]}")
    _render_pdf_to_png(pdf_out, out_png, dpi=dpi)
    return _crop_whitespace(out_png, pad_px=18)


def prepare_vector_assets(temp_dir: Path) -> Dict[str, Path]:
    """Pre-renders PDF/TikZ schematics into 300-DPI PNGs for inline Word/Google Docs embedding."""
    rendered: Dict[str, Path] = {}
    arch_pdf = FIGURES_DIR / "GDUT_A6_YOLO11s_Overall_Simplified.pdf"
    rendered["figures/GDUT_A6_YOLO11s_Overall_Simplified.pdf"] = _render_pdf_to_png(
        arch_pdf, temp_dir / "fig_iv4_arch_300dpi.png", dpi=300
    )

    ch5_txt = (CHAPTERS_DIR / "ch5_system.tex").read_text(encoding="utf-8")
    tikz_figs = re.findall(
        r"\\begin\{figure\}.*?(\\resizebox.*?\\end\{tikzpicture\}\}?).*?\\label\{(fig:[^}]+)\}",
        ch5_txt,
        flags=re.S,
    )
    for tikz_block, label_name in tikz_figs:
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", label_name)
        rendered[label_name] = _render_tikz_to_png(
            tikz_block, temp_dir / f"{safe_name}_300dpi.png", temp_dir, safe_name, dpi=300
        )
    return rendered


# ---------------------------------------------------------------------------
# Math & Inline LaTeX-to-Word Run Renderer
# ---------------------------------------------------------------------------
MATH_SYMBOLS = {
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\Delta": "Δ",
    r"\epsilon": "ε",
    r"\varepsilon": "ε",
    r"\sigma": "σ",
    r"\Sigma": "Σ",
    r"\tau": "τ",
    r"\rho": "ρ",
    r"\mu": "μ",
    r"\pi": "π",
    r"\times": "×",
    r"\pm": "±",
    r"\le": "≤",
    r"\leq": "≤",
    r"\ge": "≥",
    r"\geq": "≥",
    r"\approx": "≈",
    r"\neq": "≠",
    r"\cdot": "·",
    r"\odot": "⊙",
    r"\in": "∈",
    r"\notin": "∉",
    r"\cap": "∩",
    r"\cup": "∪",
    r"\land": "∧",
    r"\lor": "∨",
    r"\to": "→",
    r"\rightarrow": "→",
    r"\leftarrow": "←",
    r"\leftrightarrow": "↔",
    r"\mapsto": "↦",
    r"\sum": "∑",
    r"\int": "∫",
    r"\infty": "∞",
    r"\forall": "∀",
    r"\exists": "∃",
    r"\partial": "∂",
    r"\nabla": "∇",
    r"\top": "ᵀ",
    r"\dagger": "†",
    r"\ddagger": "‡",
    r"\mathbb{R}": "ℝ",
    r"\mathbb{I}": "𝕀",
    r"\mathcal{L}": "L",
    r"\mathcal{N}": "N",
    r"\mathbf{X}": "𝐗",
    r"\mathbf{Y}": "𝐘",
    r"\mathbf{W}": "𝐖",
    r"\mathbf{b}": "𝐛",
    r"\mathbf{Q}": "𝐐",
    r"\mathbf{K}": "𝐊",
    r"\mathbf{V}": "𝐕",
    r"\mathbf{I}": "𝐈",
    r"\mathbf{O}": "𝐎",
    r"\mathbf{A}": "𝐀",
    r"\tilde{\mathbf{X}}": "X̃",
    r"\quad": "  ",
    r"\qquad": "    ",
    r"\,": " ",
    r"\!": "",
    r"\;": " ",
    r"\left": "",
    r"\right": "",
    r"\nonumber": "",
}


def _emit_math_runs(
    para,
    math_src: str,
    font_name: str = "Times New Roman",
    font_size: Pt = Pt(12),
    color: Optional[RGBColor] = None,
    bold: bool = False,
    sup: bool = False,
    sub: bool = False,
) -> None:
    """Converts TeX math expression recursively into clean Unicode + Word superscript/subscript runs."""
    s = math_src.strip()
    # Normalize common compound macros first
    s = s.replace(r"\mathbb{R}", "ℝ").replace(r"\mathbb{I}", "𝕀")
    s = s.replace(r"\tilde{\mathbf{X}}", "X̃")
    s = re.sub(r"\\mathcal\{([A-Za-z])\}", lambda m: {"L": "L", "N": "N"}.get(m.group(1), m.group(1)), s)

    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            if i + 1 < len(s) and s[i - 1 : i] not in ("(", "["):
                r = para.add_run(" ")
                _style_run(r, font_name, font_size, color=color, bold=bold, sup=sup, sub=sub)
            while i < len(s) and s[i].isspace():
                i += 1
            continue

        if ch in ("^", "_"):
            is_sup = (ch == "^")
            i += 1
            while i < len(s) and s[i].isspace():
                i += 1
            if i < len(s) and s[i] == "{":
                arg, i = _extract_brace_arg(s, i)
            elif i < len(s):
                if s[i] == "\\":
                    m_cmd = re.match(r"\\[A-Za-z]+", s[i:])
                    if m_cmd:
                        arg = m_cmd.group(0)
                        i += len(arg)
                    else:
                        arg = s[i]
                        i += 1
                else:
                    arg = s[i]
                    i += 1
            else:
                arg = ""
            _emit_math_runs(
                para,
                arg,
                font_name=font_name,
                font_size=font_size,
                color=color,
                bold=bold,
                sup=True if is_sup else sup,
                sub=True if not is_sup else sub,
            )
            continue

        if ch == "\\":
            if s.startswith(r"\frac", i):
                p1 = i + len(r"\frac")
                while p1 < len(s) and s[p1].isspace():
                    p1 += 1
                num, p2 = _extract_brace_arg(s, p1)
                while p2 < len(s) and s[p2].isspace():
                    p2 += 1
                den, i = _extract_brace_arg(s, p2)
                use_paren_n = any(op in num for op in ("+", "-", " "))
                use_paren_d = any(op in den for op in ("+", "-", " "))
                if use_paren_n:
                    _style_run(para.add_run("("), font_name, font_size, color=color, bold=bold, sup=sup, sub=sub)
                _emit_math_runs(para, num, font_name, font_size, color=color, bold=bold, sup=sup, sub=sub)
                if use_paren_n:
                    _style_run(para.add_run(")"), font_name, font_size, color=color, bold=bold, sup=sup, sub=sub)
                _style_run(para.add_run(" / "), font_name, font_size, color=color, bold=bold, sup=sup, sub=sub)
                if use_paren_d:
                    _style_run(para.add_run("("), font_name, font_size, color=color, bold=bold, sup=sup, sub=sub)
                _emit_math_runs(para, den, font_name, font_size, color=color, bold=bold, sup=sup, sub=sub)
                if use_paren_d:
                    _style_run(para.add_run(")"), font_name, font_size, color=color, bold=bold, sup=sup, sub=sub)
                continue

            if s.startswith(r"\sqrt", i):
                p1 = i + len(r"\sqrt")
                while p1 < len(s) and s[p1].isspace():
                    p1 += 1
                arg, i = _extract_brace_arg(s, p1)
                _style_run(para.add_run("√("), font_name, font_size, color=color, bold=bold, sup=sup, sub=sub)
                _emit_math_runs(para, arg, font_name, font_size, color=color, bold=bold, sup=sup, sub=sub)
                _style_run(para.add_run(")"), font_name, font_size, color=color, bold=bold, sup=sup, sub=sub)
                continue

            for wrap_cmd in (r"\text", r"\mathrm", r"\mathbf", r"\bm", r"\textit", r"\texttt"):
                if s.startswith(wrap_cmd + "{", i):
                    arg, i = _extract_brace_arg(s, i + len(wrap_cmd))
                    b_arg = bold or (wrap_cmd in (r"\mathbf", r"\bm"))
                    it_arg = (wrap_cmd == r"\textit")
                    clean_arg = arg.replace(r"\text{--}", "–").replace("--", "–").replace(r"\_", "_")
                    if clean_arg == "--":
                        clean_arg = "–"
                    r = para.add_run(clean_arg)
                    _style_run(
                        r,
                        "Consolas" if wrap_cmd == r"\texttt" else font_name,
                        font_size,
                        color=color,
                        bold=b_arg,
                        italic=it_arg,
                        sup=sup,
                        sub=sub,
                    )
                    break
            else:
                if s.startswith(r"\label{", i):
                    _, i = _extract_brace_arg(s, i + len(r"\label"))
                    continue
                if s.startswith(r"\\", i):
                    i += 2
                    continue
                if s[i : i + 2] in (r"\,", r"\!", r"\;", r"\ "):
                    sym = MATH_SYMBOLS.get(s[i : i + 2], " ")
                    if sym:
                        _style_run(para.add_run(sym), font_name, font_size, color=color, bold=bold, sup=sup, sub=sub)
                    i += 2
                    continue
                m_cmd = re.match(r"\\[A-Za-z]+", s[i:])
                if m_cmd:
                    tok = m_cmd.group(0)
                    i += len(tok)
                    if tok == r"\times":
                        k = i
                        while k < len(s) and s[k].isspace():
                            k += 1
                        is_binary = k < len(s) and (s[k].isalnum() or s[k] in ("(", "\\", "{"))
                        if is_binary:
                            prev_space = bool(para.runs and para.runs[-1].text.endswith(" "))
                            repl = "× " if prev_space else " × "
                            i = k
                        else:
                            repl = "×"
                    else:
                        repl = MATH_SYMBOLS.get(tok)
                        if repl is None:
                            repl = tok[1:]  # e.g. \max, \min, \log, \exp, \softmax, \arctan
                    if repl:
                        _style_run(para.add_run(repl), font_name, font_size, color=color, bold=bold, sup=sup, sub=sub)
                    continue
                if i + 1 < len(s) and s[i + 1] in ("_", "%", "&", "$", "#", "{", "}"):
                    _style_run(para.add_run(s[i + 1]), font_name, font_size, color=color, bold=bold, sup=sup, sub=sub)
                    i += 2
                    continue
                i += 1
            continue

        if ch in ("{", "}"):
            i += 1
            continue

        if ch == "'":
            _style_run(para.add_run("′"), font_name, font_size, color=color, bold=bold, sup=sup, sub=sub)
            i += 1
            continue

        _style_run(para.add_run(ch), font_name, font_size, color=color, bold=bold, sup=sup, sub=sub)
        i += 1


def _style_run(
    run,
    font_name: str,
    font_size: Pt,
    color: Optional[RGBColor] = None,
    bold: bool = False,
    italic: bool = False,
    sup: bool = False,
    sub: bool = False,
) -> None:
    run.font.name = font_name
    run.font.size = font_size
    if color is not None:
        run.font.color.rgb = color
    if bold:
        run.bold = True
    if italic:
        run.italic = True
    if sup:
        run.font.superscript = True
    if sub:
        run.font.subscript = True


def _pre_resolve_refs_and_cites(
    text: str,
    labels_map: Dict[str, Tuple[str, str]],
    bib_map: Dict[str, str],
) -> str:
    def _rep_ref(m):
        k = m.group(1).strip()
        return labels_map.get(k, (f"[{k}]", ""))[0]

    def _rep_eqref(m):
        k = m.group(1).strip()
        return f"({labels_map.get(k, (f'{k}', ''))[0]})"

    def _rep_cite(m):
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        nums = [bib_map.get(k, k) for k in keys]
        return "[" + ", ".join(nums) + "]"

    text = re.sub(r"\\eqref\{([^}]+)\}", _rep_eqref, text)
    text = re.sub(r"\\ref\{([^}]+)\}", _rep_ref, text)
    text = re.sub(r"\\cite\{([^}]+)\}", _rep_cite, text)
    return text


def _unescape_plain_latex(s: str) -> str:
    s = s.replace(r"\%", "%").replace(r"\_", "_").replace(r"\&", "&").replace(r"\#", "#").replace(r"\$", "$")
    s = s.replace("~", " ").replace(r"\,", " ").replace(r"\ ", " ")
    s = s.replace("---", "—").replace("--", "–")
    s = s.replace("``", "“").replace("''", "”")
    return s


def emit_inline_runs(
    para,
    text: str,
    labels_map: Dict[str, Tuple[str, str]],
    bib_map: Dict[str, str],
    base_font: str = "Times New Roman",
    base_size: Pt = Pt(12),
    base_color: Optional[RGBColor] = COLOR_DARK_TEXT,
    bold: bool = False,
    italic: bool = False,
) -> None:
    """Recursively emits formatted docx runs from LaTeX text without leaving raw macros."""
    txt = _pre_resolve_refs_and_cites(text, labels_map, bib_map)

    i = 0
    buf: List[str] = []

    def _flush():
        if buf:
            chunk = _unescape_plain_latex("".join(buf))
            if chunk:
                r = para.add_run(chunk)
                _style_run(r, base_font, base_size, color=base_color, bold=bold, italic=italic)
            buf.clear()

    while i < len(txt):
        ch = txt[i]
        # Display/inline math $...$ (but not escaped \$)
        if ch == "$" and (i == 0 or txt[i - 1] != "\\"):
            _flush()
            end_m = i + 1
            while end_m < len(txt):
                if txt[end_m] == "$" and txt[end_m - 1] != "\\":
                    break
                end_m += 1
            math_inner = txt[i + 1 : end_m]
            _emit_math_runs(para, math_inner, font_name=base_font, font_size=base_size, color=base_color, bold=bold)
            i = end_m + 1
            continue

        if ch == "\\":
            if txt.startswith(r"\newline", i):
                _flush()
                para.add_run().add_break()
                i += len(r"\newline")
                continue
            if txt.startswith(r"\\", i):
                _flush()
                # Ignore optional [0.5cm]
                j = i + 2
                if j < len(txt) and txt[j] == "[":
                    _, j = _extract_bracket_arg(txt, j)
                para.add_run().add_break()
                i = j
                continue
            if i + 1 < len(txt) and txt[i + 1] in ("%", "_", "&", "$", "#", "{", "}"):
                buf.append(txt[i + 1])
                i += 2
                continue
            if txt[i : i + 2] in (r"\,", r"\ ", r"\;"):
                buf.append(" ")
                i += 2
                continue

            matched = False
            for cmd, st_b, st_i, f_override, sz_delta, col_ov in (
                (r"\textbf", True, False, None, 0.0, None),
                (r"\textit", False, True, None, 0.0, None),
                (r"\emph", False, True, None, 0.0, None),
                (r"\texttt", False, False, "Consolas", -1.0, COLOR_CODE),
                (r"\nolinkurl", False, False, "Consolas", -1.0, COLOR_CODE),
                (r"\hexhash", False, False, "Consolas", -1.0, COLOR_CODE),
                (r"\url", False, False, "Consolas", -1.0, COLOR_LINK),
                (r"\text", False, False, None, 0.0, None),
                (r"\mathrm", False, False, None, 0.0, None),
            ):
                if txt.startswith(cmd + "{", i):
                    _flush()
                    arg, i = _extract_brace_arg(txt, i + len(cmd))
                    if cmd in (r"\nolinkurl", r"\url", r"\hexhash"):
                        r_url = para.add_run(arg.replace(r"\_", "_"))
                        _style_run(
                            r_url,
                            f_override or base_font,
                            Pt(max(8.0, float(base_size.pt) + sz_delta)),
                            color=col_ov or base_color,
                            bold=bold or st_b,
                            italic=italic or st_i,
                        )
                    else:
                        emit_inline_runs(
                            para,
                            arg,
                            labels_map,
                            bib_map,
                            base_font=f_override or base_font,
                            base_size=Pt(max(8.0, float(base_size.pt) + sz_delta)),
                            base_color=col_ov or base_color,
                            bold=bold or st_b,
                            italic=italic or st_i,
                        )
                    matched = True
                    break
            if matched:
                continue

            if txt.startswith(r"\textcolor{", i):
                _flush()
                cname, j = _extract_brace_arg(txt, i + len(r"\textcolor"))
                inner, i = _extract_brace_arg(txt, j)
                c_rgb = COLOR_FPT_BLUE if "fptblue" in cname or "blue" in cname else base_color
                emit_inline_runs(para, inner, labels_map, bib_map, base_font, base_size, c_rgb, bold, italic)
                continue

            if txt.startswith(r"\color{", i):
                _flush()
                _, i = _extract_brace_arg(txt, i + len(r"\color"))
                continue

            if txt.startswith(r"\fontsize{", i):
                _flush()
                _, j = _extract_brace_arg(txt, i + len(r"\fontsize"))
                _, j = _extract_brace_arg(txt, j)
                if txt.startswith(r"\selectfont", j):
                    j += len(r"\selectfont")
                i = j
                continue

            if txt.startswith(r"\selectfont", i) or txt.startswith(r"\bfseries", i) or txt.startswith(r"\scriptsize", i) or txt.startswith(r"\footnotesize", i):
                _flush()
                m_c = re.match(r"\\[a-zA-Z]+", txt[i:])
                i += len(m_c.group(0)) if m_c else 1
                continue

            if txt.startswith(r"\rule{", i) or txt.startswith(r"\vspace", i) or txt.startswith(r"\hspace", i):
                _flush()
                m_skip = re.match(r"\\[a-zA-Z]+\*?(?:\{[^}]*\})*", txt[i:])
                i += len(m_skip.group(0)) if m_skip else 1
                if i < len(txt) and txt[i] == "{":
                    _, i = _extract_brace_arg(txt, i)
                continue

            if txt.startswith(r"\hfill", i):
                _flush()
                buf.append("   |   ")
                i += len(r"\hfill")
                continue

            if txt.startswith(r"\`", i) or txt.startswith(r"\'", i):
                buf.append(txt[i + 2 : i + 3])
                i += 3
                continue

            # Ignore non-printing structural macros cleanly
            m_other = re.match(r"\\([a-zA-Z@]+)", txt[i:])
            if m_other:
                cmd_name = "\\" + m_other.group(1)
                if cmd_name in MATH_SYMBOLS:
                    _flush()
                    _style_run(para.add_run(MATH_SYMBOLS[cmd_name]), base_font, base_size, base_color, bold, italic)
                    i += len(cmd_name)
                    continue
                i += len(cmd_name)
                continue

        if ch == "{":
            _flush()
            arg, i = _extract_brace_arg(txt, i)
            emit_inline_runs(para, arg, labels_map, bib_map, base_font, base_size, base_color, bold, italic)
            continue

        if ch == "}":
            i += 1
            continue

        buf.append(ch)
        i += 1

    _flush()


# ---------------------------------------------------------------------------
# Environment Parsers (Tables, Figures, Listings, Equations, Lists)
# ---------------------------------------------------------------------------
def _split_top_level_rows(body: str) -> List[str]:
    """Splits LaTeX table body on row separators `\\\\` outside brace groups."""
    rows: List[str] = []
    cur: List[str] = []
    depth = 0
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body) and body[i + 1] in ("{", "}"):
            cur.append(ch + body[i + 1])
            i += 2
            continue
        if ch == "{":
            depth += 1
            cur.append(ch)
            i += 1
            continue
        if ch == "}":
            depth = max(0, depth - 1)
            cur.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < len(body) and body[i + 1] == "\\" and depth == 0:
            j = i + 2
            if j < len(body) and body[j] == "[":
                _, j = _extract_bracket_arg(body, j)
            row_txt = "".join(cur).strip()
            if row_txt:
                rows.append(row_txt)
            cur.clear()
            i = j
            continue
        cur.append(ch)
        i += 1
    rem = "".join(cur).strip()
    if rem:
        rows.append(rem)
    return rows


def _split_top_level_cells(row_str: str) -> List[str]:
    cells: List[str] = []
    cur: List[str] = []
    depth = 0
    i = 0
    while i < len(row_str):
        ch = row_str[i]
        if ch == "\\" and i + 1 < len(row_str) and row_str[i + 1] in ("&", "{", "}"):
            cur.append(ch + row_str[i + 1])
            i += 2
            continue
        if ch == "{":
            depth += 1
            cur.append(ch)
            i += 1
            continue
        if ch == "}":
            depth = max(0, depth - 1)
            cur.append(ch)
            i += 1
            continue
        if ch == "&" and depth == 0:
            cells.append("".join(cur).strip())
            cur.clear()
            i += 1
            continue
        cur.append(ch)
        i += 1
    cells.append("".join(cur).strip())
    return cells


def _clean_table_row_directives(row_str: str) -> Tuple[str, bool]:
    """Strips hline/cline/rowcolor directives; returns (cleaned_row, is_shaded)."""
    shaded = r"\rowcolor" in row_str
    s = row_str
    s = re.sub(r"\\hline\b", "", s)
    s = re.sub(r"\\(toprule|midrule|bottomrule)\b", "", s)
    s = re.sub(r"\\cline\{[^}]*\}", "", s)
    s = re.sub(r"\\rowcolor\{[^}]*\}", "", s)
    s = re.sub(r"\\end(firsthead|head|foot|lastfoot)\b", "", s)
    return s.strip(), shaded


def _strip_colspec_modifiers(colspec: str) -> str:
    """Removes LaTeX `>{...}` and `<{...}` column decorators (e.g. `>{\\bfseries}`) cleanly."""
    out: List[str] = []
    i = 0
    while i < len(colspec):
        if colspec[i] in (">", "<") and i + 1 < len(colspec) and colspec[i + 1] == "{":
            _, i = _extract_brace_arg(colspec, i + 1)
            continue
        out.append(colspec[i])
        i += 1
    return "".join(out)


def _parse_colspec_tokens(colspec: str) -> List[Tuple[Optional[float], str]]:
    """Returns list of (explicit_cm_or_None, align_char) per table column."""
    clean = _strip_colspec_modifiers(colspec)
    res: List[Tuple[Optional[float], str]] = []
    for m in re.finditer(r"p\{([\d.]+)cm\}|([lcrX])", clean):
        if m.group(1):
            res.append((float(m.group(1)), "l"))
        else:
            res.append((None, m.group(2)))
    return res


def _cell_plain_preview(cell_latex: str) -> str:
    """Produces rough plain text for a table cell to measure unbreakable token widths."""
    s = re.sub(r"\\(?:cite|ref|eqref|autoref)\{[^}]*\}", " [99] ", cell_latex)
    s = re.sub(r"\\(?:textbf|textit|emph|texttt|text|mathrm|nolinkurl|hexhash|url)\{([^}]*)\}", r"\1", s)
    s = s.replace(r"\_", "_").replace(r"\%", "%").replace(r"\&", "&")
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}$]", " ", s)
    return " ".join(s.split())


def _compute_col_widths_cm(
    colspec: str,
    num_cols: int,
    parsed_rows: Optional[List[Tuple[bool, List[Tuple[int, str, str]]]]] = None,
    total_cm: float = 16.0,
) -> List[float]:
    parsed_spec = _parse_colspec_tokens(colspec)
    if len(parsed_spec) != num_cols:
        parsed_spec = [(None, "l")] * num_cols

    # Measure both overall column length and longest unbreakable word (split on whitespace, slashes, hyphens)
    is_dense = num_cols >= 8
    char_cm = 0.142 if is_dense else (0.155 if num_cols >= 6 else 0.168)
    margin_cm = (80.0 if is_dense else 130.0) / 566.929
    min_words_len = [4.0] * num_cols
    est_lens = [5.0] * num_cols
    if parsed_rows:
        for r_i, (_shaded, row_items) in enumerate(parsed_rows[:25]):
            c_idx = 0
            for span, _as, ctxt in row_items:
                if c_idx >= num_cols:
                    break
                if span == 1:
                    plain = _cell_plain_preview(ctxt)
                    # Header tokens (r_i == 0) are bold; wide glyphs (@, %, A-Z, m, w) need ~1.22x width
                    weight = 1.08 if r_i == 0 else 1.0
                    tokens = [t for seg in plain.split() for t in re.split(r"[-/]", seg) if t]
                    longest_tok = max((len(t) for t in tokens), default=4)
                    tok_eff = max(
                        (sum(1.22 if (ch.isupper() or ch in "@%mw") else 1.0 for ch in t[:20]) for t in tokens),
                        default=4.0,
                    ) * weight
                    if tok_eff > min_words_len[c_idx]:
                        min_words_len[c_idx] = tok_eff
                    eff_len = max(min(len(plain), 28), longest_tok + 2)
                    if eff_len > est_lens[c_idx]:
                        est_lens[c_idx] = float(eff_len)
                c_idx += span

    min_widths = [
        round(max(0.95 if is_dense else (1.15 if num_cols >= 6 else 1.55), min_words_len[i] * char_cm + margin_cm + 0.18), 3)
        for i in range(num_cols)
    ]

    desired: List[float] = []
    for idx, (exp_cm, ch_tok) in enumerate(parsed_spec):
        if exp_cm is not None:
            w = max(exp_cm, min_widths[idx])
        elif ch_tok == "X":
            w = max(3.6, min(7.2, 0.24 * est_lens[idx]), min_widths[idx])
        elif ch_tok == "l":
            w = max(2.2, min(4.8, 0.22 * est_lens[idx]), min_widths[idx])
        else:  # c or r
            w = max(1.05 if is_dense else 1.20, min(2.6, 0.15 * est_lens[idx]), min_widths[idx])
        desired.append(w)

    # Give X columns any remaining budget before slack compression
    x_indices = [i for i, (_e, ch) in enumerate(parsed_spec) if ch == "X"]
    non_x_sum = sum(desired[i] for i in range(num_cols) if i not in x_indices)
    if x_indices and non_x_sum < total_cm:
        per_x = (total_cm - non_x_sum) / len(x_indices)
        for xi in x_indices:
            desired[xi] = max(min_widths[xi], per_x)

    min_sum = sum(min_widths)
    des_sum = sum(desired) or 1.0
    if min_sum >= total_cm:
        scale = total_cm / min_sum
        return [round(w * scale, 3) for w in min_widths]

    if des_sum <= total_cm:
        # Expand X columns (or l columns if no X) to fill full total_cm cleanly
        extra = total_cm - des_sum
        expand_idxs = x_indices or [i for i, (_e, ch) in enumerate(parsed_spec) if ch == "l"] or list(range(num_cols))
        for ei in expand_idxs:
            desired[ei] += extra / len(expand_idxs)
        return [round(w, 3) for w in desired]

    # Compress only slack above min_widths so no unbreakable token gets crushed below min_widths[i]
    slacks = [max(0.0, desired[i] - min_widths[i]) for i in range(num_cols)]
    slack_sum = sum(slacks) or 1.0
    avail_slack = total_cm - min_sum
    return [round(min_widths[i] + avail_slack * (slacks[i] / slack_sum), 3) for i in range(num_cols)]


def _render_latex_table_into_doc(
    doc: Document,
    block_text: str,
    is_longtable: bool,
    labels_map: Dict[str, Tuple[str, str]],
    bib_map: Dict[str, str],
) -> None:
    cap_txt = ""
    lbl_key = ""
    m_lbl = re.search(r"\\label\{([^}]+)\}", block_text)
    if m_lbl:
        lbl_key = m_lbl.group(1).strip()
    m_cap = re.search(r"\\caption(?=\s*(?:\[|\{))", block_text)
    if m_cap:
        cap_txt, _ = _extract_caption_arg(block_text, m_cap.start())

    trailing_note = ""
    if is_longtable:
        start_c = block_text.find("{", block_text.find(r"\begin{longtable}") + len(r"\begin{longtable}"))
        colspec, body_pos = _extract_brace_arg(block_text, start_c)
        end_pos = block_text.rfind(r"\end{longtable}")
        table_body = block_text[body_pos:end_pos]
        # Strip everything from \endfirsthead..\endhead and \endfoot to keep 1 clean header + data rows
        if r"\endfirsthead" in table_body and r"\endfoot" in table_body:
            part_hdr = table_body.split(r"\endfirsthead", 1)[0]
            part_dat = table_body.split(r"\endfoot", 1)[1]
            table_body = part_hdr + "\n" + part_dat
        elif r"\endhead" in table_body:
            table_body = table_body.split(r"\endhead", 1)[1]
        # Brace-balanced removal of \caption{...}\label{...}\\ inside longtable top (handles nested \nolinkurl{} / \texttt{})
        while True:
            m_c = re.search(r"\\caption(?=\s*(?:\[|\{))", table_body)
            if not m_c:
                break
            _, after_c = _extract_caption_arg(table_body, m_c.start())
            rest = table_body[after_c:].lstrip()
            if rest.startswith(r"\label{"):
                _, after_l = _extract_brace_arg(rest, len(r"\label"))
                rest = rest[after_l:].lstrip()
            if rest.startswith(r"\\"):
                rest = rest[2:].lstrip()
            table_body = table_body[: m_c.start()] + rest
    else:
        m_tab = re.search(r"\\begin\{(tabularx|tabular)\}", block_text)
        if not m_tab:
            return
        env_name = m_tab.group(1)
        pos = m_tab.end()
        if env_name == "tabularx":
            _, pos = _extract_brace_arg(block_text, block_text.find("{", pos))
        colspec, pos = _extract_brace_arg(block_text, block_text.find("{", pos))
        end_pos = block_text.rfind(rf"\end{{{env_name}}}")
        table_body = block_text[pos:end_pos]
        after_tab_pos = end_pos + len(rf"\end{{{env_name}}}")
        end_tbl_pos = block_text.rfind(r"\end{table}")
        if end_tbl_pos > after_tab_pos:
            raw_note = block_text[after_tab_pos:end_tbl_pos].strip()
            raw_note = re.sub(r"^\s*\}+\s*", "", raw_note)
            raw_note = re.sub(r"\\vspace\*?\{[^}]*\}", "", raw_note)
            raw_note = re.sub(r"\\(par|noindent|footnotesize|scriptsize|small|normalsize|raggedright|centering)\b", "", raw_note)
            trailing_note = raw_note.strip()

    raw_rows = _split_top_level_rows(table_body)
    parsed_rows: List[Tuple[bool, List[Tuple[int, str, str]]]] = []
    max_cols = 0
    for rr in raw_rows:
        cleaned, is_shaded = _clean_table_row_directives(rr)
        if not cleaned:
            continue
        cells_raw = _split_top_level_cells(cleaned)
        if not cells_raw or all(not c for c in cells_raw):
            continue
        row_parsed: List[Tuple[int, str, str]] = []
        col_count_row = 0
        for cell_s in cells_raw:
            m_mc = re.match(r"\\multicolumn\{(\d+)\}\{([^}]*)\}\{(.*)\}$", cell_s, flags=re.S)
            if m_mc:
                span = max(1, int(m_mc.group(1)))
                align_spec = m_mc.group(2)
                inner = m_mc.group(3).strip()
                row_parsed.append((span, align_spec, inner))
                col_count_row += span
            else:
                row_parsed.append((1, "", cell_s))
                col_count_row += 1
        max_cols = max(max_cols, col_count_row)
        parsed_rows.append((is_shaded, row_parsed))

    if not parsed_rows or max_cols == 0:
        return

    # Emit caption above table (matching academic PDF convention)
    if cap_txt:
        t_num = labels_map.get(lbl_key, ("", ""))[0] if lbl_key else ""
        p_cap = doc.add_paragraph()
        p_cap.paragraph_format.space_before = Pt(8)
        p_cap.paragraph_format.space_after = Pt(3)
        p_cap.paragraph_format.keep_with_next = True
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        prefix = f"Table {t_num}: " if t_num else "Table: "
        r_pre = p_cap.add_run(prefix)
        _style_run(r_pre, "Times New Roman", Pt(10.5), COLOR_FPT_BLUE, bold=True)
        emit_inline_runs(p_cap, cap_txt, labels_map, bib_map, "Times New Roman", Pt(10.5), COLOR_DARK_TEXT)

    is_borderless = "| " not in colspec and "|" not in colspec
    tbl = doc.add_table(rows=len(parsed_rows), cols=max_cols)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    _set_table_borders(tbl, none=is_borderless)

    widths_cm = _compute_col_widths_cm(colspec, max_cols, parsed_rows=parsed_rows, total_cm=16.0)
    col_dxas = _apply_fixed_table_widths(tbl, widths_cm)
    col_align_tokens = [ch for (_w, ch) in _parse_colspec_tokens(colspec)]
    # Also respect >{\centering...}p{...} from raw colspec cleanly
    raw_col_parts = re.findall(r"((?:[><]\{[^}]*\})*(?:p\{[\d.]+cm\}|[lcrX]))", colspec)
    if len(raw_col_parts) == max_cols:
        col_align_tokens = [
            "c" if (r"\centering" in part or part.endswith("c")) else ("r" if part.endswith("r") else "l")
            for part in raw_col_parts
        ]

    pad_v = 45 if max_cols >= 6 else 70
    pad_h = 40 if max_cols >= 8 else (55 if max_cols >= 6 else 80)

    for r_idx, (is_shaded, row_items) in enumerate(parsed_rows):
        row = tbl.rows[r_idx]
        _set_row_cant_split(row)
        is_header_row = is_shaded or (not is_borderless and r_idx == 0)
        if is_header_row:
            _set_row_header(row)

        c_cursor = 0
        for span, align_spec, cell_txt in row_items:
            if c_cursor >= max_cols:
                break
            cell = row.cells[c_cursor]
            if span > 1 and (c_cursor + span - 1) < max_cols:
                cell.merge(row.cells[c_cursor + span - 1])
            cell.width = Cm(sum(widths_cm[c_cursor : c_cursor + span]))
            _set_cell_width_dxa(cell, sum(col_dxas[c_cursor : c_cursor + span]))
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            _set_cell_margins(cell, top_dxa=pad_v, bottom_dxa=pad_v, left_dxa=pad_h, right_dxa=pad_h)
            if is_shaded and not is_borderless:
                _set_cell_shading(cell, HEADER_HEX)

            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.08
            col_tok = align_spec or (col_align_tokens[c_cursor] if c_cursor < len(col_align_tokens) else "l")
            if is_shaded or "c" in col_tok:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif "r" in col_tok:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            if cell_txt:
                f_sz = Pt(8.0) if max_cols >= 8 else (Pt(8.8) if max_cols >= 6 else Pt(9.5))
                emit_inline_runs(
                    p,
                    cell_txt,
                    labels_map,
                    bib_map,
                    base_font="Times New Roman",
                    base_size=f_sz,
                    base_color=COLOR_DARK_TEXT,
                    bold=False,
                )
            c_cursor += span

    if trailing_note:
        p_note = doc.add_paragraph()
        p_note.paragraph_format.space_before = Pt(3)
        p_note.paragraph_format.space_after = Pt(4)
        p_note.paragraph_format.line_spacing = 1.12
        emit_inline_runs(p_note, trailing_note, labels_map, bib_map, "Times New Roman", Pt(9.5), COLOR_MUTED)

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(0)
    p_sp.paragraph_format.space_after = Pt(4)


def _render_latex_figure_into_doc(
    doc: Document,
    block_text: str,
    labels_map: Dict[str, Tuple[str, str]],
    bib_map: Dict[str, str],
    vector_assets: Dict[str, Path],
) -> None:
    m_lbl = re.search(r"\\label\{([^}]+)\}", block_text)
    lbl_key = m_lbl.group(1).strip() if m_lbl else ""
    fig_num = labels_map.get(lbl_key, ("", ""))[0] if lbl_key else ""

    # Subfigures vs main caption
    subfigs = list(re.finditer(r"\\begin\{subfigure\}.*?\\end\{subfigure\}", block_text, flags=re.S))
    cleaned_for_main_cap = re.sub(r"\\begin\{subfigure\}.*?\\end\{subfigure\}", "", block_text, flags=re.S)
    m_cap = re.search(r"\\caption(?=\s*(?:\[|\{))", cleaned_for_main_cap)
    main_cap = _extract_caption_arg(cleaned_for_main_cap, m_cap.start())[0] if m_cap else ""

    def _resolve_img(rel_path: str) -> Optional[Path]:
        if rel_path in vector_assets and vector_assets[rel_path].exists():
            return vector_assets[rel_path]
        cand = THESIS_DIR / rel_path
        if cand.exists():
            return cand
        return None

    if lbl_key in vector_assets and vector_assets[lbl_key].exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(3)
        p_img.paragraph_format.keep_with_next = True
        p_img.add_run().add_picture(str(vector_assets[lbl_key]), width=Cm(15.2))
    elif subfigs:
        # Arrange subfigures cleanly in 2-column borderless reflowable table
        ncols = 2
        nrows = (len(subfigs) + 1) // 2
        grid = doc.add_table(rows=nrows, cols=ncols)
        grid.alignment = WD_TABLE_ALIGNMENT.CENTER
        grid.autofit = False
        _set_table_borders(grid, none=True)
        for idx_sf, m_sf in enumerate(subfigs):
            r_i, c_i = divmod(idx_sf, ncols)
            row = grid.rows[r_i]
            _set_row_cant_split(row)
            cell = row.cells[c_i]
            cell.width = Cm(7.9)
            _set_cell_margins(cell, 40, 40, 40, 40)
            sf_txt = m_sf.group(0)
            m_inc = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", sf_txt)
            m_sc = re.search(r"\\caption(?=\s*(?:\[|\{))", sf_txt)
            sc_txt = _extract_caption_arg(sf_txt, m_sc.start())[0] if m_sc else ""
            p_c = cell.paragraphs[0]
            p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_c.paragraph_format.space_after = Pt(2)
            if m_inc:
                img_p = _resolve_img(m_inc.group(1).strip())
                if img_p:
                    p_c.add_run().add_picture(str(img_p), width=Cm(7.4))
            if sc_txt:
                p_sub = cell.add_paragraph()
                p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_sub.paragraph_format.space_after = Pt(4)
                sub_lbl = chr(ord("a") + idx_sf)
                r_s = p_sub.add_run(f"({sub_lbl}) ")
                _style_run(r_s, "Times New Roman", Pt(9.5), COLOR_DARK_TEXT, bold=True)
                emit_inline_runs(p_sub, sc_txt, labels_map, bib_map, "Times New Roman", Pt(9.5), COLOR_DARK_TEXT)
    else:
        m_inc = re.search(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}", block_text)
        if m_inc:
            opt_w = m_inc.group(1) or ""
            img_p = _resolve_img(m_inc.group(2).strip())
            if img_p:
                w_cm = 15.0
                m_frac = re.search(r"width=([\d.]+)\\textwidth", opt_w)
                if m_frac:
                    w_cm = min(15.4, max(6.0, float(m_frac.group(1)) * 15.6))
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(6)
                p_img.paragraph_format.space_after = Pt(3)
                p_img.paragraph_format.keep_with_next = True
                p_img.add_run().add_picture(str(img_p), width=Cm(w_cm))

    if main_cap:
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        p_cap.paragraph_format.space_after = Pt(8)
        prefix = f"Figure {fig_num}: " if fig_num else "Figure: "
        r_pre = p_cap.add_run(prefix)
        _style_run(r_pre, "Times New Roman", Pt(10.0), COLOR_FPT_BLUE, bold=True)
        emit_inline_runs(p_cap, main_cap, labels_map, bib_map, "Times New Roman", Pt(10.0), COLOR_DARK_TEXT)


def _render_lstlisting_into_doc(
    doc: Document,
    block_text: str,
    labels_map: Dict[str, Tuple[str, str]],
    bib_map: Dict[str, str],
) -> None:
    m_start = re.match(r"\\begin\{lstlisting\}(?:\[(.*?)\])?", block_text, flags=re.S)
    opts = m_start.group(1) if (m_start and m_start.group(1)) else ""
    code_body = block_text[m_start.end() : block_text.rfind(r"\end{lstlisting}")].strip("\r\n")

    lbl_key = ""
    cap_txt = ""
    m_lbl = re.search(r"label=\{([^}]+)\}", opts)
    if m_lbl:
        lbl_key = m_lbl.group(1).strip()
    m_cap = re.search(r"caption=\{", opts)
    if m_cap:
        cap_txt, _ = _extract_brace_arg(opts, m_cap.end() - 1)

    lst_num = labels_map.get(lbl_key, ("", ""))[0] if lbl_key else ""
    if cap_txt:
        p_cap = doc.add_paragraph()
        p_cap.paragraph_format.space_before = Pt(6)
        p_cap.paragraph_format.space_after = Pt(2)
        p_cap.paragraph_format.keep_with_next = True
        prefix = f"Listing {lst_num}: " if lst_num else "Listing: "
        r_p = p_cap.add_run(prefix)
        _style_run(r_p, "Times New Roman", Pt(10.0), COLOR_FPT_BLUE, bold=True)
        emit_inline_runs(p_cap, cap_txt, labels_map, bib_map, "Times New Roman", Pt(10.0), COLOR_DARK_TEXT)

    box = doc.add_table(rows=1, cols=1)
    box.alignment = WD_TABLE_ALIGNMENT.CENTER
    box.autofit = False
    _set_table_borders(box, border_hex=BORDER_HEX, sz="4")
    row = box.rows[0]
    _set_row_cant_split(row)
    cell = row.cells[0]
    cell.width = Cm(15.8)
    _set_cell_shading(cell, CODE_BG_HEX)
    _set_cell_margins(cell, 80, 80, 120, 100)
    code_lines = code_body.splitlines() or [""]
    for idx_ln, raw_ln in enumerate(code_lines):
        p = cell.paragraphs[0] if idx_ln == 0 else cell.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.08
        lead = len(raw_ln) - len(raw_ln.lstrip(" "))
        safe_ln = ("\u00A0" * lead + raw_ln[lead:]) if raw_ln else "\u00A0"
        r = p.add_run(safe_ln)
        _style_run(r, "Consolas", Pt(8.5), COLOR_CODE)

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(4)


def _render_display_equation(
    doc: Document,
    env_name: str,
    block_text: str,
    labels_map: Dict[str, Tuple[str, str]],
) -> None:
    inner = block_text[len(rf"\begin{{{env_name}}}") : block_text.rfind(rf"\end{{{env_name}}}")].strip()
    lines = [inner] if env_name.startswith("equation") else [ln.strip() for ln in inner.split(r"\\") if ln.strip()]

    for eq_line in lines:
        m_lbl = re.search(r"\\label\{([^}]+)\}", eq_line)
        eq_num = ""
        if m_lbl:
            eq_num = labels_map.get(m_lbl.group(1).strip(), ("", ""))[0]
        clean_eq = re.sub(r"\\label\{[^}]+\}", "", eq_line).replace(r"\nonumber", "").replace("&=", " = ").replace("&", "  ").strip()

        tbl = doc.add_table(rows=1, cols=2)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        _set_table_borders(tbl, none=True)
        row = tbl.rows[0]
        _set_row_cant_split(row)
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Cm(14.0)
        c1.width = Cm(1.9)
        _set_cell_shading(c0, ALT_ROW_HEX)
        _set_cell_shading(c1, ALT_ROW_HEX)
        _set_cell_margins(c0, 55, 55, 80, 80)
        _set_cell_margins(c1, 55, 55, 40, 80)
        c1.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

        p0 = c0.paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p0.paragraph_format.space_before = Pt(0)
        p0.paragraph_format.space_after = Pt(0)
        _emit_math_runs(p0, clean_eq, font_name="Times New Roman", font_size=Pt(11.0), color=COLOR_DARK_TEXT)

        p1 = c1.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p1.paragraph_format.space_before = Pt(0)
        p1.paragraph_format.space_after = Pt(0)
        if eq_num:
            _style_run(p1.add_run(f"({eq_num})"), "Times New Roman", Pt(10.5), COLOR_FPT_BLUE, bold=True)

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(4)


def _split_list_items(list_body: str) -> List[str]:
    """Splits `\item ...` at the top brace/env depth, keeping nested sub-lists intact inside each item."""
    items: List[str] = []
    cur: List[str] = []
    env_depth = 0
    lines = list_body.splitlines()
    for ln in lines:
        s = ln.strip()
        if re.match(r"\\begin\{(enumerate|itemize)\}", s):
            env_depth += 1
        elif re.match(r"\\end\{(enumerate|itemize)\}", s):
            env_depth = max(0, env_depth - 1)
        if env_depth == 0 and s.startswith(r"\item"):
            if cur:
                items.append("\n".join(cur).strip())
                cur.clear()
            item_rest = s[len(r"\item") :].strip()
            cur.append(item_rest)
        else:
            cur.append(ln)
    if cur:
        items.append("\n".join(cur).strip())
    return [it for it in items if it]


def _render_list_block(
    doc: Document,
    env_name: str,
    body_inner: str,
    labels_map: Dict[str, Tuple[str, str]],
    bib_map: Dict[str, str],
    nest_level: int = 0,
) -> None:
    items = _split_list_items(body_inner)
    for idx, item_txt in enumerate(items, 1):
        # Check for nested enumerate/itemize inside item
        m_sub = re.search(r"\\begin\{(enumerate|itemize)\}(?:\[[^\]]*\])?(.*)\\end\{\1\}", item_txt, flags=re.S)
        main_part = item_txt[: m_sub.start()].strip() if m_sub else item_txt
        bullet = f"{idx}." if env_name == "enumerate" else ("◦" if nest_level > 0 else "•")

        p = doc.add_paragraph()
        left_cm = 0.75 + nest_level * 0.65
        p.paragraph_format.left_indent = Cm(left_cm)
        p.paragraph_format.first_line_indent = Cm(-0.45)
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        r_b = p.add_run(f"{bullet} ")
        _style_run(r_b, "Times New Roman", Pt(12), COLOR_FPT_BLUE if env_name == "enumerate" else COLOR_DARK_TEXT, bold=True)

        # Extract any display equation/align environments nested inside the list item
        eq_pat = re.compile(r"(\\begin\{(equation|align)\*?\}.*?\\end\{\2\*?\})", flags=re.S)
        eq_parts = list(eq_pat.finditer(main_part))
        if not eq_parts:
            emit_inline_runs(p, main_part, labels_map, bib_map, "Times New Roman", Pt(12), COLOR_DARK_TEXT)
        else:
            cursor = 0
            first_seg = True
            for m_eq in eq_parts:
                before = main_part[cursor : m_eq.start()].strip()
                if before:
                    if first_seg:
                        emit_inline_runs(p, before, labels_map, bib_map, "Times New Roman", Pt(12), COLOR_DARK_TEXT)
                    else:
                        p_cont = doc.add_paragraph()
                        p_cont.paragraph_format.left_indent = Cm(left_cm)
                        p_cont.paragraph_format.space_before = Pt(2)
                        p_cont.paragraph_format.space_after = Pt(3)
                        p_cont.paragraph_format.line_spacing = 1.15
                        emit_inline_runs(p_cont, before, labels_map, bib_map, "Times New Roman", Pt(12), COLOR_DARK_TEXT)
                    first_seg = False
                eq_block = m_eq.group(1)
                eq_env_full = eq_block[len(r"\begin{") : eq_block.find("}")]
                _render_display_equation(doc, eq_env_full, eq_block, labels_map)
                first_seg = False
                cursor = m_eq.end()
            tail_txt = main_part[cursor:].strip()
            if tail_txt:
                p_tail = doc.add_paragraph()
                p_tail.paragraph_format.left_indent = Cm(left_cm)
                p_tail.paragraph_format.space_before = Pt(2)
                p_tail.paragraph_format.space_after = Pt(3)
                p_tail.paragraph_format.line_spacing = 1.15
                emit_inline_runs(p_tail, tail_txt, labels_map, bib_map, "Times New Roman", Pt(12), COLOR_DARK_TEXT)

        if m_sub:
            _render_list_block(doc, m_sub.group(1), m_sub.group(2), labels_map, bib_map, nest_level=nest_level + 1)


# ---------------------------------------------------------------------------
# Sequential Environment-Aware Chapter Parser
# ---------------------------------------------------------------------------
ENV_BLOCK_RE = re.compile(
    r"\\begin\{(table|longtable|tabular|tabularx|figure|lstlisting|equation|align|enumerate|itemize|center|titlepage)\}"
)


def _find_matching_end(text: str, start_idx: int, env_name: str) -> int:
    begin_pat = rf"\begin{{{env_name}}}"
    end_pat = rf"\end{{{env_name}}}"
    depth = 0
    i = start_idx
    while i < len(text):
        next_b = text.find(begin_pat, i)
        next_e = text.find(end_pat, i)
        if next_e == -1:
            return len(text)
        if next_b != -1 and next_b < next_e:
            depth += 1
            i = next_b + len(begin_pat)
        else:
            depth -= 1
            i = next_e + len(end_pat)
            if depth <= 0:
                return i
    return len(text)


def _render_text_paragraphs(
    doc: Document,
    chunk: str,
    labels_map: Dict[str, Tuple[str, str]],
    bib_map: Dict[str, str],
    chapter_state: Dict[str, Any],
) -> None:
    """Splits non-environment prose by blank lines and headings, emitting native Headings and body paragraphs."""
    # Strip comments line-by-line cleanly while keeping escaped \%
    clean_lines: List[str] = []
    for raw_ln in chunk.splitlines():
        st = raw_ln.lstrip()
        if st.startswith("%"):
            continue
        # Strip inline unescaped % comment
        ln = re.sub(r"(?<!\\)%.*$", "", raw_ln)
        clean_lines.append(ln)
    text = "\n".join(clean_lines)

    blocks = re.split(r"\n\s*\n", text)
    for blk in blocks:
        b = blk.strip()
        if not b:
            continue

        # Handle standalone heading commands anywhere inside block lines
        lines = b.splitlines()
        prose_acc: List[str] = []

        def _flush_prose():
            joined = " ".join(prose_acc).strip()
            prose_acc.clear()
            if not joined:
                return
            # Strip non-printing preamble commands
            joined_s = re.sub(r"\\label\{[^}]*\}", "", joined)
            joined_s = re.sub(r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", joined_s)
            joined_s = re.sub(r"\\addtocontents\{[^}]*\}\{.*?\}(?=\s*$|\s*\\)", "", joined_s)
            joined_s = re.sub(r"\\(?:thispagestyle|setstretch|vspace\*?|hspace\*?)\{[^}]*\}", "", joined_s)
            joined_s = re.sub(r"\\(?:setlength|renewcommand|addtolength)\{[^}]*\}\{[^}]*\}", "", joined_s)
            joined_s = re.sub(r"\\(clearpage|newpage|FloatBarrier|vfill|noindent|begingroup|endgroup|scriptsize|footnotesize|tiny|small|normalsize)\b", "", joined_s)
            joined_s = re.sub(r"\\setcounter\{[^}]*\}\{[^}]*\}", "", joined_s)
            joined_s = joined_s.strip()
            if not joined_s:
                return
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.line_spacing = 1.18
            has_long_unbroken = any(len(tok) >= 24 for tok in re.split(r"\s+", joined_s))
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if has_long_unbroken else WD_ALIGN_PARAGRAPH.JUSTIFY
            emit_inline_runs(p, joined_s, labels_map, bib_map, "Times New Roman", Pt(12), COLOR_DARK_TEXT)

        for ln in lines:
            st = ln.strip()
            if not st:
                continue
            if re.match(r"^\\(clearpage|newpage)\b", st):
                _flush_prose()
                doc.add_page_break()
                continue
            if st.startswith(r"\chapter"):
                _flush_prose()
                m_ch = re.match(r"\\chapter\*?\{", st)
                if m_ch:
                    title, _ = _extract_brace_arg(st, m_ch.end() - 1)
                    if chapter_state.get("seen_first_chapter"):
                        doc.add_page_break()
                    chapter_state["seen_first_chapter"] = True
                    chapter_state["ch"] = chapter_state.get("ch", 0) + 1
                    chapter_state["sec"] = 0
                    chapter_state["sub"] = 0
                    chapter_state["subsub"] = 0
                    c_idx = chapter_state["ch"]
                    roman = ROMAN_CHAPTERS.get(c_idx, str(c_idx))
                    h = doc.add_heading(f"{roman}. {_unescape_plain_latex(title)}", level=1)
                    h.paragraph_format.space_before = Pt(4)
                    h.paragraph_format.space_after = Pt(10)
                    for r in h.runs:
                        _style_run(r, "Times New Roman", Pt(16), COLOR_FPT_BLUE, bold=True)
                    _set_paragraph_bottom_border(h, "0F4C81", "8")
                    if title.strip().lower() == "appendices":
                        chapter_state["is_appendix"] = True
                continue
            if st.startswith(r"\section"):
                _flush_prose()
                is_star = st.startswith(r"\section*")
                m_s = re.match(r"\\section\*?\{", st)
                if m_s:
                    title, _ = _extract_brace_arg(st, m_s.end() - 1)
                    title_c = _unescape_plain_latex(re.sub(r"\\&", "&", title))
                    if is_star:
                        if chapter_state.get("need_page_before_front"):
                            doc.add_page_break()
                        chapter_state["need_page_before_front"] = True
                        h = doc.add_heading(title_c, level=1)
                        h.paragraph_format.space_before = Pt(4)
                        h.paragraph_format.space_after = Pt(8)
                        for r in h.runs:
                            _style_run(r, "Times New Roman", Pt(15), COLOR_FPT_BLUE, bold=True)
                        _set_paragraph_bottom_border(h, "0F4C81", "6")
                    else:
                        chapter_state["sec"] = chapter_state.get("sec", 0) + 1
                        chapter_state["sub"] = 0
                        chapter_state["subsub"] = 0
                        if chapter_state.get("is_appendix"):
                            app_ltr = chr(ord("A") + chapter_state["sec"] - 1)
                            prefix = f"Appendix {app_ltr}. "
                        else:
                            prefix = f"{chapter_state['sec']}. "
                        h = doc.add_heading(prefix + title_c, level=2)
                        h.paragraph_format.space_before = Pt(10)
                        h.paragraph_format.space_after = Pt(4)
                        for r in h.runs:
                            _style_run(r, "Times New Roman", Pt(13.5), COLOR_DARK_TEXT, bold=True)
                continue
            if st.startswith(r"\subsection"):
                _flush_prose()
                m_s = re.match(r"\\subsection\*?\{", st)
                if m_s:
                    title, _ = _extract_brace_arg(st, m_s.end() - 1)
                    chapter_state["sub"] = chapter_state.get("sub", 0) + 1
                    chapter_state["subsub"] = 0
                    num_s = f"{chapter_state['sec']}.{chapter_state['sub']} "
                    h = doc.add_heading("", level=3)
                    h.paragraph_format.space_before = Pt(8)
                    h.paragraph_format.space_after = Pt(3)
                    _style_run(h.add_run(num_s), "Times New Roman", Pt(12.5), COLOR_DARK_TEXT, bold=True)
                    emit_inline_runs(h, title, labels_map, bib_map, "Times New Roman", Pt(12.5), COLOR_DARK_TEXT, bold=True)
                continue
            if st.startswith(r"\subsubsection"):
                _flush_prose()
                m_s = re.match(r"\\subsubsection\*?\{", st)
                if m_s:
                    title, _ = _extract_brace_arg(st, m_s.end() - 1)
                    chapter_state["subsub"] = chapter_state.get("subsub", 0) + 1
                    num_s = f"{chapter_state['sec']}.{chapter_state['sub']}.{chapter_state['subsub']} "
                    h = doc.add_heading("", level=4)
                    h.paragraph_format.space_before = Pt(6)
                    h.paragraph_format.space_after = Pt(2)
                    _style_run(h.add_run(num_s), "Times New Roman", Pt(12), COLOR_DARK_TEXT, bold=True, italic=True)
                    emit_inline_runs(h, title, labels_map, bib_map, "Times New Roman", Pt(12), COLOR_DARK_TEXT, bold=True, italic=True)
                continue
            if st in (r"\listoftables", r"\listoffigures", r"\tableofcontents"):
                _flush_prose()
                kind = "lot" if "tables" in st else ("lof" if "figures" in st else "toc")
                _render_toc_like_list(doc, kind, labels_map, bib_map)
                continue
            prose_acc.append(st)

        _flush_prose()


def _render_toc_like_list(
    doc: Document,
    kind: str,
    labels_map: Dict[str, Tuple[str, str]],
    bib_map: Dict[str, str],
) -> None:
    heading_title = {"lot": "List of Tables", "lof": "List of Figures", "toc": "Table of Contents"}[kind]
    doc.add_page_break()
    h = doc.add_heading(heading_title, level=1)
    for r in h.runs:
        _style_run(r, "Times New Roman", Pt(15), COLOR_FPT_BLUE, bold=True)
    _set_paragraph_bottom_border(h, "0F4C81", "6")

    entries = _parse_toc_entries(THESIS_DIR / f"main.{kind}", kind)
    if not entries:
        return

    tbl = doc.add_table(rows=len(entries), cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    _set_table_borders(tbl, none=True)
    for r_i, (c_type, num_str, title_raw, page_str) in enumerate(entries):
        row = tbl.rows[r_i]
        _set_row_cant_split(row)
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Cm(14.5)
        c1.width = Cm(1.4)
        _set_cell_margins(c0, 22, 22, 20, 40)
        _set_cell_margins(c1, 22, 22, 20, 20)
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_before = Pt(0)
        p0.paragraph_format.space_after = Pt(0)
        is_top = (kind == "toc" and c_type == "chapter")
        indent_cm = 0.0 if (is_top or kind != "toc") else (0.5 if c_type == "section" else 1.0)
        p0.paragraph_format.left_indent = Cm(indent_cm)
        label_txt = f"{num_str}  " if num_str else ""
        if label_txt:
            _style_run(p0.add_run(label_txt), "Times New Roman", Pt(10.5), COLOR_FPT_BLUE if (is_top or kind != "toc") else COLOR_DARK_TEXT, bold=True)
        emit_inline_runs(
            p0,
            title_raw,
            labels_map,
            bib_map,
            "Times New Roman",
            Pt(10.5 if (is_top or kind != "toc") else 10.0),
            COLOR_DARK_TEXT,
            bold=is_top,
        )
        p1 = c1.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p1.paragraph_format.space_before = Pt(0)
        p1.paragraph_format.space_after = Pt(0)
        _style_run(p1.add_run(page_str), "Times New Roman", Pt(10.0), COLOR_MUTED, bold=is_top)


def _render_frontmatter_titlepages(
    doc: Document,
    front_txt: str,
    labels_map: Dict[str, Tuple[str, str]],
    bib_map: Dict[str, str],
) -> str:
    """Renders Cover Page, Title Page, signatures, and returns remainder of frontmatter.tex."""
    logo_path = FIGURES_DIR / "fpt_template_image1.png"

    # Extract first two titlepage blocks cleanly
    pos = 0
    tp_blocks: List[str] = []
    for _ in range(2):
        idx = front_txt.find(r"\begin{titlepage}", pos)
        if idx == -1:
            break
        end_idx = _find_matching_end(front_txt, idx, "titlepage")
        tp_blocks.append(front_txt[idx:end_idx])
        pos = end_idx

    # 1. Cover Page
    if len(tp_blocks) >= 1:
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_after = Pt(12)
        if logo_path.exists():
            p_logo.add_run().add_picture(str(logo_path), width=Cm(5.2))

        for line_txt, sz, col in (
            ("MINISTRY OF EDUCATION AND TRAINING", Pt(13), COLOR_DARK_TEXT),
            ("FPT UNIVERSITY", Pt(15), COLOR_FPT_BLUE),
            ("CAPSTONE PROJECT DOCUMENT", Pt(14), COLOR_DARK_TEXT),
        ):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(8 if sz != Pt(14) else 16)
            _style_run(p.add_run(line_txt), "Times New Roman", sz, col, bold=True)

        p_title = doc.add_paragraph()
        p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_title.paragraph_format.space_before = Pt(8)
        p_title.paragraph_format.space_after = Pt(14)
        _set_paragraph_bottom_border(p_title, "0F4C81", "12")
        p_pr = p_title._p.get_or_add_pPr()
        p_pr.append(parse_xml(f'<w:pBdr {nsdecls("w")}><w:top w:val="single" w:sz="12" w:space="6" w:color="0F4C81"/><w:bottom w:val="single" w:sz="12" w:space="6" w:color="0F4C81"/></w:pBdr>'))
        _style_run(
            p_title.add_run(
                "STRUCTURAL RE-PARAMETERIZATION, SPATIAL COORDINATE ENCODING, AND MULTI-DOMAIN ROBUSTNESS FOR REAL-TIME SAFETY HELMET AND COLOR CLASSIFICATION IN CONSTRUCTION SURVEILLANCE"
            ),
            "Times New Roman",
            Pt(15),
            COLOR_FPT_BLUE,
            bold=True,
        )

        p_grp = doc.add_paragraph()
        p_grp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_grp.paragraph_format.space_after = Pt(12)
        _style_run(p_grp.add_run("Group Name: GFA26AI17 (Group NTD)"), "Times New Roman", Pt(13), COLOR_DARK_TEXT, bold=True)

        m_tbl = re.search(r"\\begin\{table\}.*?\\end\{table\}", tp_blocks[0], flags=re.S)
        if m_tbl:
            _render_latex_table_into_doc(doc, m_tbl.group(0), False, labels_map, bib_map)

        p_loc = doc.add_paragraph()
        p_loc.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_loc.paragraph_format.space_before = Pt(18)
        _style_run(p_loc.add_run("— Ho Chi Minh City, November 2026 —"), "Times New Roman", Pt(12), COLOR_DARK_TEXT, bold=True)
        doc.add_page_break()

    # 2. Secondary Title Page
    if len(tp_blocks) >= 2:
        for line_txt, sz, col in (
            ("MINISTRY OF EDUCATION AND TRAINING", Pt(13), COLOR_DARK_TEXT),
            ("FPT UNIVERSITY — HO CHI MINH CITY CAMPUS", Pt(15), COLOR_FPT_BLUE),
            ("Department of Artificial Intelligence", Pt(12), COLOR_DARK_TEXT),
        ):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(6)
            _style_run(p.add_run(line_txt), "Times New Roman", sz, col, bold=(sz != Pt(12)))

        if logo_path.exists():
            p_l2 = doc.add_paragraph()
            p_l2.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_l2.paragraph_format.space_before = Pt(10)
            p_l2.paragraph_format.space_after = Pt(14)
            p_l2.add_run().add_picture(str(logo_path), width=Cm(4.4))

        p_t2 = doc.add_paragraph()
        p_t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_t2.paragraph_format.space_after = Pt(12)
        _style_run(
            p_t2.add_run(
                "STRUCTURAL RE-PARAMETERIZATION, SPATIAL COORDINATE ENCODING, AND MULTI-DOMAIN ROBUSTNESS FOR REAL-TIME SAFETY HELMET AND COLOR CLASSIFICATION IN CONSTRUCTION SURVEILLANCE"
            ),
            "Times New Roman",
            Pt(14.5),
            COLOR_DARK_TEXT,
            bold=True,
        )
        p_sub1 = doc.add_paragraph()
        p_sub1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_sub1.paragraph_format.space_after = Pt(0)
        _style_run(
            p_sub1.add_run("A Capstone Project Report Submitted in Partial Fulfillment of the Requirements"),
            "Times New Roman",
            Pt(11.5),
            COLOR_DARK_TEXT,
            italic=True,
        )
        p_sub2 = doc.add_paragraph()
        p_sub2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_sub2.paragraph_format.space_before = Pt(0)
        p_sub2.paragraph_format.space_after = Pt(18)
        _style_run(
            p_sub2.add_run("for the Degree of Bachelor of Information Technology (Artificial Intelligence)"),
            "Times New Roman",
            Pt(11.5),
            COLOR_DARK_TEXT,
            italic=True,
        )

        m_tx = re.search(r"\\begin\{tabularx\}.*?\\end\{tabularx\}", tp_blocks[1], flags=re.S)
        if m_tx:
            _render_latex_table_into_doc(doc, m_tx.group(0), False, labels_map, bib_map)

        p_loc2 = doc.add_paragraph()
        p_loc2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_loc2.paragraph_format.space_before = Pt(18)
        _style_run(p_loc2.add_run("Ho Chi Minh City, November 2026"), "Times New Roman", Pt(12), COLOR_DARK_TEXT, bold=True)
        doc.add_page_break()

    rem = front_txt[pos:]
    # Convert the Task Assignment two-minipage signature block into a clean borderless tabularx
    sig_pair = (
        r"\begin{tabular}{cc}\textbf{Supervisor Approval}\newline\newline"
        r".....................................................\newline"
        r"\textbf{M.Sc. Ha Anh Vu} & "
        r"\textbf{Team Representative}\newline\newline"
        r".....................................................\newline"
        r"\textbf{Nguyen Han Nhu (SE183644)}\end{tabular}"
    )
    rem = re.sub(
        r"\\begin\{minipage\}.*?\\end\{minipage\}\s*\\hfill\s*\\begin\{minipage\}.*?\\end\{minipage\}",
        lambda _m: sig_pair,
        rem,
        flags=re.S,
    )
    return rem


def _render_chapter_stream(
    doc: Document,
    tex_src: str,
    labels_map: Dict[str, Tuple[str, str]],
    bib_map: Dict[str, str],
    vector_assets: Dict[str, Path],
    chapter_state: Dict[str, Any],
) -> None:
    pos = 0
    while pos < len(tex_src):
        m = ENV_BLOCK_RE.search(tex_src, pos)
        if not m:
            _render_text_paragraphs(doc, tex_src[pos:], labels_map, bib_map, chapter_state)
            break
        if m.start() > pos:
            _render_text_paragraphs(doc, tex_src[pos : m.start()], labels_map, bib_map, chapter_state)

        env_name = m.group(1)
        end_idx = _find_matching_end(tex_src, m.start(), env_name)
        block = tex_src[m.start() : end_idx]

        if env_name in ("table", "longtable", "tabular", "tabularx"):
            _render_latex_table_into_doc(doc, block, env_name == "longtable", labels_map, bib_map)
        elif env_name == "figure":
            _render_latex_figure_into_doc(doc, block, labels_map, bib_map, vector_assets)
        elif env_name == "lstlisting":
            _render_lstlisting_into_doc(doc, block, labels_map, bib_map)
        elif env_name in ("equation", "align"):
            _render_display_equation(doc, env_name, block, labels_map)
        elif env_name in ("enumerate", "itemize"):
            m_hdr = re.match(rf"\\begin\{{{env_name}\}}(?:\[[^\]]*\])?", block)
            inner = block[m_hdr.end() : block.rfind(rf"\end{{{env_name}}}")]
            _render_list_block(doc, env_name, inner, labels_map, bib_map, nest_level=0)
        elif env_name == "center":
            inner = block[len(r"\begin{center}") : block.rfind(r"\end{center}")]
            _render_text_paragraphs(doc, inner, labels_map, bib_map, chapter_state)
        elif env_name == "titlepage":
            pass
        pos = end_idx


def _render_references_chapter(
    doc: Document,
    bbl_path: Path,
    labels_map: Dict[str, Tuple[str, str]],
    bib_map: Dict[str, str],
) -> None:
    doc.add_page_break()
    h = doc.add_heading("VIII. References", level=1)
    for r in h.runs:
        _style_run(r, "Times New Roman", Pt(16), COLOR_FPT_BLUE, bold=True)
    _set_paragraph_bottom_border(h, "0F4C81", "8")

    raw_bbl = bbl_path.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"\\bibitem\{([^}]+)\}", raw_bbl)
    # parts = [header, key1, body1, key2, body2, ...]
    for idx in range(1, len(parts), 2):
        key = parts[idx].strip()
        body = parts[idx + 1]
        body = body.split(r"\end{thebibliography}")[0]
        body = body.replace(r"\newblock", " ")
        body = re.sub(r"\\BIBentrySTDinterwordspacing|\\BIBentryALTinterwordspacing", "", body)
        body = re.sub(r"\\hskip\s+.*?\\relax", " ", body)
        body = re.sub(r"\b1em\s+plus\s+0\.5em\s+minus\s+0\.4em\b", " ", body)
        body = " ".join(body.split())
        num = bib_map.get(key, str((idx + 1) // 2))

        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.85)
        p.paragraph_format.first_line_indent = Cm(-0.85)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        _style_run(p.add_run(f"[{num}]  "), "Times New Roman", Pt(11), COLOR_FPT_BLUE, bold=True)
        emit_inline_runs(p, body, labels_map, bib_map, "Times New Roman", Pt(11), COLOR_DARK_TEXT)


def build_docx(output_path: Path) -> Path:
    labels_map, bib_map = parse_aux_file(THESIS_DIR / "main.aux")
    assert labels_map and bib_map, "Missing main.aux labels/bibcites; run LuaLaTeX compile first."

    temp_assets = THESIS_DIR / "_docx_assets"
    try:
        vector_assets = prepare_vector_assets(temp_assets)
        doc = Document()

        # Page setup: A4 margins matching academic Word & Google Docs cleanly
        sec = doc.sections[0]
        sec.page_width = Cm(21.0)
        sec.page_height = Cm(29.7)
        sec.top_margin = Cm(2.3)
        sec.bottom_margin = Cm(2.3)
        sec.left_margin = Cm(2.8)
        sec.right_margin = Cm(2.0)
        sec.different_first_page_header_footer = True

        # Configure default Normal style cleanly
        normal_style = doc.styles["Normal"]
        normal_style.font.name = "Times New Roman"
        normal_style.font.size = Pt(12)
        normal_style.font.color.rgb = COLOR_DARK_TEXT

        # Running Header & Page Number Footer
        hdr_p = sec.header.paragraphs[0]
        hdr_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        _style_run(
            hdr_p.add_run("Capstone Project FA26AI16 — Group GFA26AI17    |    Safety Helmet & Color Detection (GDUT-HWD)"),
            "Times New Roman",
            Pt(9),
            COLOR_MUTED,
            italic=True,
        )
        ftr_p = sec.footer.paragraphs[0]
        ftr_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _style_run(ftr_p.add_run("Page "), "Times New Roman", Pt(9.5), COLOR_MUTED)
        r_pg = ftr_p.add_run()
        _style_run(r_pg, "Times New Roman", Pt(9.5), COLOR_MUTED, bold=True)
        _add_page_number_field(r_pg)

        chapter_state: Dict[str, Any] = {
            "ch": 0,
            "sec": 0,
            "sub": 0,
            "subsub": 0,
            "seen_first_chapter": False,
            "need_page_before_front": False,
            "is_appendix": False,
        }

        front_raw = (CHAPTERS_DIR / "frontmatter.tex").read_text(encoding="utf-8")
        front_rem = _render_frontmatter_titlepages(doc, front_raw, labels_map, bib_map)
        _render_chapter_stream(doc, front_rem, labels_map, bib_map, vector_assets, chapter_state)

        for ch_name in (
            "ch1_intro.tex",
            "ch2_pmp.tex",
            "ch3_sota.tex",
            "ch4_methodology.tex",
            "ch5_system.tex",
            "ch6_results.tex",
            "ch7_conclusion.tex",
        ):
            txt = (CHAPTERS_DIR / ch_name).read_text(encoding="utf-8")
            _render_chapter_stream(doc, txt, labels_map, bib_map, vector_assets, chapter_state)

        # VIII. References directly from IEEE main.bbl
        _render_references_chapter(doc, THESIS_DIR / "main.bbl", labels_map, bib_map)

        # IX. Appendices
        chapter_state["ch"] = 8  # increments to IX on \chapter{Appendices}
        ch9_txt = (CHAPTERS_DIR / "ch9_appendices.tex").read_text(encoding="utf-8")
        _render_chapter_stream(doc, ch9_txt, labels_map, bib_map, vector_assets, chapter_state)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        return output_path
    finally:
        if temp_assets.exists():
            shutil.rmtree(temp_assets, ignore_errors=True)


# ---------------------------------------------------------------------------
# Strict Automated QA Verification Gate on Exported .docx
# ---------------------------------------------------------------------------
BANNED_DOCX_TOKENS = (
    r"\begin{",
    r"\end{",
    r"\ref{",
    r"\eqref{",
    r"\cite{",
    r"\textbf{",
    r"\textit{",
    r"\emph{",
    r"\texttt{",
    r"\nolinkurl{",
    r"\hexhash{",
    r"\includegraphics",
    r"\section{",
    r"\chapter{",
    r"\caption{",
    r"\label{",
    r"\setstretch{",
    "??",
    "[?]",
    r"\${",
    "tabularcc",
    "tabular>",
    "tabularx",
    "dagger",
    ">X",
    "1em plus",
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
    "tundng111",
    "026a2afb",
    "98-FN",
    "Pair-3407",
    "V8F",
    "V8S",
    "b1_dung",
    "AAIML",
    "L4021",
    "<2%",
    "< 2%",
    "below 2%",
    "lower variance",
    "HHW variance",
    "minimal variance",
    "giảm phương sai",
    "81%–90%",
    "81%--90%",
    "Requirementsfor",
    "3× 3",
    "640× 640",
    "forensically",
    "uncompromising",
    "decisively",
    "cryptographically verifiable",
    "industrial-grade",
    "unvarnished",
    "candidly",
    "transparently",
    "delivers tangible value",
    "exemplifies",
    "Production-Ready",
    "rigorously",
    "20.897",
    "19.733",
    "13.239",
    "12.539",
    "79.75 FPS",
    "a6_yolo11s_val_pred0",
    "Enterprise VMS",
)

REQUIRED_DOCX_METRICS = (
    "90.94%",
    "58.63%",
    "89.76%",
    "59.08%",
    "88.94%",
    "58.08%",
    "84.85%",
    "87.51%",
    "74%–81%",
    "Timing Note:",
    "9,476,783",
    "15.207",
    "20.206",
    "19.177",
    "9.754",
    "2.88 ms/frame",
    "FIGURE_PROVENANCE.md",
    "75/75",
    "SHWD_V1",
    "operator_feedback",
    "Tóm tắt",
    "143.7 FPS",
    "65.7 FPS",
    "501.1",
    "260.8",
    "141",
    "84 passed, 1 skipped",
    "ce2a883e",
    "d422dd1d",
    "agnostic_nms",
    "tests/conftest.py",
)


def verify_docx(docx_path: Path) -> Dict[str, Any]:
    assert docx_path.exists(), f"Missing DOCX: {docx_path}"

    # 1. Inspect embedded media inside .docx zip archive
    with zipfile.ZipFile(docx_path, "r") as zf:
        media_files = [n for n in zf.namelist() if n.startswith("word/media/")]
        for mf in media_files:
            sz = zf.getinfo(mf).file_size
            assert sz > 0, f"Zero-byte image inside DOCX: {mf}"
        assert len(media_files) >= 17, f"Expected >= 17 embedded images (Logo + 16 figs), found {len(media_files)}"

    # 2. Scan all paragraphs and table cells for leaked LaTeX tokens & required content
    doc = Document(str(docx_path))
    table_cnt = len(doc.tables)
    assert table_cnt >= 25, f"Expected >= 25 native tables, found {table_cnt}"

    chunks: List[str] = [p.text for p in doc.paragraphs if p.text]
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                if cell.text:
                    chunks.append(cell.text)
    full_text = "\n".join(chunks)

    leaks: List[Tuple[str, str]] = []
    for bad in BANNED_DOCX_TOKENS:
        idx = full_text.find(bad)
        if idx != -1:
            ctx = full_text[max(0, idx - 35) : idx + 65].replace("\n", " ")
            leaks.append((bad, ctx))
    m_tab_leak = re.search(r"tabular(?!-nums)", full_text)
    if m_tab_leak:
        idx = m_tab_leak.start()
        leaks.append(("tabular*", full_text[max(0, idx - 35) : idx + 65].replace("\n", " ")))
    m_dup = re.search(r"\b(Appendix|Chapter|Table|Figure|Section)\s+\1\b", full_text, flags=re.I)
    if m_dup:
        idx = m_dup.start()
        leaks.append((m_dup.group(0), full_text[max(0, idx - 35) : idx + 65].replace("\n", " ")))
    assert not leaks, f"Leaked LaTeX/unresolved/duplicate tokens inside DOCX: {leaks}"

    missing = [req for req in REQUIRED_DOCX_METRICS if req not in full_text]
    assert not missing, f"Missing required v3.2 metrics/hashes inside DOCX: {missing}"

    # Verify every shaded 1x1 Listing block has >= 5 distinct <w:p> paragraphs (not collapsed into 1 para)
    listing_para_counts: List[int] = []
    for tbl in doc.tables:
        if len(tbl.rows) == 1 and len(tbl.columns) == 1:
            shd_els = tbl.rows[0].cells[0]._tc.xpath(f".//w:shd[@w:fill='{CODE_BG_HEX}']")
            if shd_els:
                listing_para_counts.append(len(tbl.rows[0].cells[0].paragraphs))
    assert len(listing_para_counts) >= 6, f"Expected >= 6 Listing tables in DOCX, found {len(listing_para_counts)}"
    assert all(cnt >= 5 for cnt in listing_para_counts), (
        f"Collapsed Listing paragraphs detected in DOCX: {listing_para_counts}"
    )

    sha = hashlib.sha256(docx_path.read_bytes()).hexdigest()
    info = {
        "path": str(docx_path),
        "size_bytes": docx_path.stat().st_size,
        "sha256": sha,
        "embedded_images": len(media_files),
        "native_tables": table_cnt,
        "listing_para_counts": listing_para_counts,
        "paragraphs": len(doc.paragraphs),
    }
    print(
        f"[DOCX_AUDIT_PASS] images={info['embedded_images']} | tables={info['native_tables']} | "
        f"listings_paras={listing_para_counts} | paras={info['paragraphs']} | size={info['size_bytes']:,} B | sha256={sha}"
    )
    return info


def main():
    parser = argparse.ArgumentParser(description="Export v3.2.1 LaTeX Capstone Thesis to Google-Docs-ready .docx")
    parser.add_argument("--verify", action="store_true", help="Run strict automated QA assertions on exported .docx")
    args = parser.parse_args()

    out_path = build_docx(DOCX_OUT_REPO)
    DOCX_OUT_DOWNLOADS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, DOCX_OUT_DOWNLOADS)
    h1 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    h2 = hashlib.sha256(DOCX_OUT_DOWNLOADS.read_bytes()).hexdigest()
    assert h1 == h2, "DOCX sync checksum mismatch between repo and Downloads!"
    print(f"[DOCX_SYNC_OK] {out_path.name} == {DOCX_OUT_DOWNLOADS}")

    if args.verify:
        verify_docx(out_path)


if __name__ == "__main__":
    main()
