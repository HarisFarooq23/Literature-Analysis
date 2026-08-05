"""
reader.py — the ONLY module that knows the on-disk shape of
scientific-pdf-pipeline's output. Everything downstream gets a clean
PaperRecord object and never touches a file path again.

Contract (as written by scientific-pdf-pipeline's Paper.save() / Exporter):

    <pipeline_output_root>/
        master_catalogue.csv                       (corpus-level summary; informational only)
        <FirstAuthor_Journal_Year_hash8>/
            metadata.csv                            (single data row, the source of truth per paper)
            cleaned_full_text.txt
            checkpoint.json
            sections/<safe_section_name>.txt        (present only if sections were found)
            tables/<table_id>.json                  (present only if tables were found)
            figures/figures_index.json + *.png      (present only if figures were found)
            formulas/formulas.json                  (present only if formulas were found)

We discover papers by walking subdirectories of pipeline_output_root and
reading each metadata.csv directly — NOT by parsing master_catalogue.csv,
which does not carry the stable paper id/hash needed to join back to a
folder. metadata.csv is authoritative.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger

from .formulas import render_formulas_text
from .tables import render_tables_text

_SKIP_STATUSES = {"failed", "pending", "skipped_duplicate"}


@dataclass
class PaperRecord:
    paper_id: str             # hash_blake3 — stable across runs, the canonical join key
    title: str
    first_author: str
    journal: str
    publication_year: Optional[int]
    doi: str
    material_class: str
    is_review: bool
    output_dir: Path
    abstract: str
    text_for_extraction: str  # assembled, truncated text actually sent to the LLM


def discover_paper_dirs(pipeline_output_root: Path) -> list[Path]:
    """Return every subdirectory of pipeline_output_root that has a metadata.csv."""
    if not pipeline_output_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {pipeline_output_root}")
    return sorted(
        p.parent for p in pipeline_output_root.glob("*/metadata.csv")
    )


def _read_metadata_row(metadata_path: Path) -> dict:
    with open(metadata_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty metadata.csv: {metadata_path}")
    return rows[0]


def _select_relevant_text(
    paper_dir: Path,
    keywords: list[str],
    max_chars: int,
    include_tables: bool,
    max_table_chars: int,
    include_formulas: bool = False,
    max_formula_chars: int = 3000,
) -> str:
    """
    Assemble the text actually sent to the LLM: matched sections (or the
    full-text fallback) plus, if enabled, every table and/or formula the
    pipeline extracted for this paper. Catalysis papers routinely report the
    exact catalyst/condition/property combinations in tables rather than
    prose, so tables are included by default and budgeted separately from
    section text so a few large tables can't silently crowd out everything
    else. Formulas are off by default -- most domain schemas extract
    numeric/categorical values that live in prose and tables, not equations
    -- but some domains genuinely need them (e.g. a cited rate law or
    thermodynamic relation), so it's a per-run toggle, budgeted separately
    from both section text and tables for the same reason.

    Section names in this corpus are NOT canonicalized (they're whatever
    heading the source PDF used, lowercased and filesystem-sanitized), so we
    match section *filenames* against the keyword list by substring rather
    than assuming exact names like 'experimental.txt' exist. Sections are
    concatenated in keyword-priority order until max_chars is reached.
    Falls back to cleaned_full_text.txt if no sections directory exists or
    nothing matches.
    """
    sections_dir = paper_dir / "sections"
    chunks: list[str] = []

    if sections_dir.is_dir():
        section_files = {p.stem.lower(): p for p in sections_dir.glob("*.txt")}
        used: set[str] = set()
        for kw in keywords:
            for stem, path in section_files.items():
                if stem in used:
                    continue
                if kw.lower() in stem:
                    chunks.append(path.read_text(encoding="utf-8", errors="replace"))
                    used.add(stem)
            if sum(len(c) for c in chunks) >= max_chars:
                break

    if not chunks:
        full_text_path = paper_dir / "cleaned_full_text.txt"
        if full_text_path.exists():
            chunks.append(full_text_path.read_text(encoding="utf-8", errors="replace"))
        else:
            logger.warning(f"No sections/ and no cleaned_full_text.txt in {paper_dir}")

    text = "\n\n".join(chunks)[:max_chars]

    if include_tables:
        tables_text = render_tables_text(paper_dir, max_table_chars)
        if tables_text:
            text = f"{text}\n\n=== TABLES ===\n{tables_text}"

    if include_formulas:
        formulas_text = render_formulas_text(paper_dir, max_formula_chars)
        if formulas_text:
            text = f"{text}\n\n=== FORMULAS ===\n{formulas_text}"

    return text


def iter_papers(
    pipeline_output_root: Path,
    relevant_section_keywords: list[str],
    max_input_chars: int,
    include_tables: bool = True,
    max_table_chars: int = 6000,
    include_formulas: bool = False,
    max_formula_chars: int = 3000,
    already_done: set[str] | None = None,
) -> Iterator[PaperRecord]:
    """
    Yield one PaperRecord per eligible paper.

    Skips: papers whose pipeline processing_status indicates failure/incompleteness,
    papers flagged as duplicates by the pipeline's dedup stage, and (if
    already_done is given, e.g. from a checkpoint) papers already extracted
    in a prior run of THIS package.
    """
    already_done = already_done or set()
    paper_dirs = discover_paper_dirs(pipeline_output_root)
    logger.info(f"Discovered {len(paper_dirs)} paper directories under {pipeline_output_root}")

    n_skipped_status = n_skipped_dup = n_skipped_done = n_skipped_bad_meta = 0

    for paper_dir in paper_dirs:
        try:
            row = _read_metadata_row(paper_dir / "metadata.csv")
        except Exception as exc:
            logger.error(f"Unreadable metadata.csv in {paper_dir}: {exc}")
            n_skipped_bad_meta += 1
            continue

        paper_id = row.get("hash_blake3") or row.get("hash_sha256") or row.get("id", "")
        if not paper_id:
            logger.error(f"No stable id (hash_blake3/hash_sha256/id) in {paper_dir}, skipping")
            n_skipped_bad_meta += 1
            continue

        if row.get("processing_status", "") in _SKIP_STATUSES:
            n_skipped_status += 1
            continue

        if str(row.get("duplicate_status", "")).strip().lower() in {"true", "1"}:
            n_skipped_dup += 1
            continue

        if paper_id in already_done:
            n_skipped_done += 1
            continue

        try:
            publication_year = int(row["publication_year"]) if row.get("publication_year") else None
        except ValueError:
            publication_year = None

        text = _select_relevant_text(
            paper_dir, relevant_section_keywords, max_input_chars,
            include_tables, max_table_chars,
            include_formulas, max_formula_chars,
        )
        if not text.strip():
            logger.warning(f"No extractable text for {paper_dir.name}, skipping")
            continue

        yield PaperRecord(
            paper_id=paper_id,
            title=row.get("title", ""),
            first_author=row.get("first_author", ""),
            journal=row.get("journal", ""),
            publication_year=publication_year,
            doi=row.get("doi", ""),
            material_class=row.get("material_class", "unknown"),
            is_review=str(row.get("is_review", "")).strip().lower() in {"true", "1"},
            output_dir=paper_dir,
            abstract=row.get("abstract", ""),
            text_for_extraction=text,
        )

    logger.info(
        f"Eligible for extraction: skipped {n_skipped_status} (status), "
        f"{n_skipped_dup} (duplicate), {n_skipped_done} (already done), "
        f"{n_skipped_bad_meta} (bad metadata)"
    )
