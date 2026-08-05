"""
formulas.py — renders the formulas/formulas.json file written by
scientific-pdf-pipeline's Exporter into plain text suitable for an LLM
prompt. Pure file I/O, no network calls, so it stays unit-testable
alongside reader.py and tables.py.

formulas.json, per the pipeline's Exporter, is a single JSON list of:
    {
        "id":         str,    # e.g. "formula_0001"
        "latex":      str,    # LaTeX source if recognized, else raw/OCR text
        "is_latex":   bool,   # True if confidence >= 0.95 (valid LaTeX parse)
        "context":    str,    # surrounding sentence(s) the formula appeared in
        "page":       int,
        "confidence": float,
    }

Equations rarely carry the kind of categorical/numeric values most domain
schemas extract (those live in prose and tables), but some domains genuinely
need them -- e.g. an explicit rate law, an activation-energy expression, or
a thermodynamic relation cited as the basis for a reported value. This is
why formulas are OFF by default (extraction.include_formulas: false) and,
unlike tables, are not assumed to be worth the added prompt tokens for every
domain -- turn it on per-domain/per-run as needed.

Low-confidence (is_latex=False) entries are still included by default but
labelled as such, so the model is not misled into treating noisy OCR text
as exact notation; _try_parse downstream doesn't care either way since this
is plain context, not something validated against the extraction schema.
"""

from __future__ import annotations

import json
from pathlib import Path


def render_formulas_text(paper_dir: Path, max_chars: int) -> str:
    """Concatenate every formula in paper_dir/formulas/formulas.json into a
    single text block, capped at max_chars. Returns '' if the file is
    missing, unreadable, or empty."""
    formulas_path = paper_dir / "formulas" / "formulas.json"
    if not formulas_path.is_file():
        return ""

    try:
        records = json.loads(formulas_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""

    if not isinstance(records, list) or not records:
        return ""

    blocks: list[str] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        formula_id = rec.get("id", "?")
        page = rec.get("page", "?")
        latex = (rec.get("latex") or "").strip()
        if not latex:
            continue
        is_latex = rec.get("is_latex", False)
        confidence = rec.get("confidence")
        context = (rec.get("context") or "").strip()

        quality = "LaTeX" if is_latex else "low-confidence OCR/text"
        conf_str = f", confidence {confidence:.2f}" if isinstance(confidence, (int, float)) else ""
        header = f"Formula {formula_id} (page {page}, {quality}{conf_str}): {latex}"
        block = f"{header}\n  Context: {context}" if context else header
        blocks.append(block)

    if not blocks:
        return ""

    text = "\n\n".join(blocks)
    return text[:max_chars]
