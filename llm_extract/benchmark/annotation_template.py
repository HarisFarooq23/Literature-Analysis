"""
benchmark/annotation_template.py — generates a fill-in-the-blank gold-
standard annotation file for a domain, so a human annotator produces
ground truth in EXACTLY the shape the pipeline's own output has (paper-
scope fields + a list of experiment-scope records). This is what makes
score.py's comparison meaningful: gold and predicted are the same schema,
not two formats that need a translation layer that could itself introduce
error.

Design note on why this reuses build_pydantic_model instead of a bespoke
"annotation schema": if the domain YAML changes (a field is added/removed/
renamed), the annotation template and the scorer both pick up the change
automatically, with no separate schema to keep in sync by hand. This is the
same principle the rest of this package already follows (one YAML is the
single source of truth for a domain) applied to the benchmarking tooling.

Usage (see cli.py for the wired-up command):
    llm-extract make-annotation-template \
        --config config.yaml \
        --n-papers 30 \
        --out gold_standard_template.jsonl \
        [--seed 42] [--stratify-by material_class]

The output is a JSONL file, one paper per line, structurally identical to
extracted_records.jsonl but with every extractable value set to null and a
"gold_annotator" / "gold_annotated_at" pair of placeholder fields for the
human to fill in as they go. Annotators edit this file directly (in a text
editor, a notebook, or a small spreadsheet round-trip via pandas) --
whichever fields they can determine from reading the actual paper, they
fill in; anything genuinely not stated in the paper, they leave as null,
exactly like the LLM extraction itself is instructed to do. This symmetry
(both gold and predicted use null the same way) is what makes "field X was
correctly extracted" and "field X was correctly left null because the paper
doesn't state it" both scoreable, rather than only scoring the cases where
a value exists.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..domain_registry import SchemaBundle, get_schema

# Fields that don't count when deciding whether a gold experiment row was
# actually filled in by an annotator vs. left as an unused template slot.
# "evidence" is a nested dict the template pre-populates with an all-None
# structure (not a scalar None), so it would otherwise make every blank
# row look "non-empty" to a naive `any(v is not None for v in row.values())`
# check.
_EXCLUDED_FROM_GOLD_COMPARISON = {"evidence", "experiment_confidence"}
from ..reader import _SKIP_STATUSES, discover_paper_dirs


@dataclass
class PaperStub:
    """Just enough metadata to build an annotation template row -- unlike
    reader.PaperRecord, this deliberately does NOT assemble
    text_for_extraction (sections + tables + formulas), since selecting
    papers for annotation never needs the extraction text: a human
    annotator reads the actual PDF, not the pipeline's truncated/section-
    filtered text. Building the full text for every eligible paper just to
    sample a handful of paper_ids would be wasted I/O on a large corpus."""
    paper_id: str
    doi: str
    title: str
    first_author: str
    journal: str
    publication_year: Optional[int]


def _read_metadata_row(metadata_path: Path) -> dict:
    with open(metadata_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty metadata.csv: {metadata_path}")
    return rows[0]


def _list_eligible_paper_stubs(
    pipeline_output_root: Path,
    already_annotated_ids: Optional[set[str]] = None,
) -> list[PaperStub]:
    """Same eligibility rules as reader.iter_papers (skip failed/incomplete
    processing_status, skip duplicates, skip already-annotated paper_ids),
    but metadata-only -- no section/table/formula text assembly."""
    already_annotated_ids = already_annotated_ids or set()
    stubs = []
    for paper_dir in discover_paper_dirs(pipeline_output_root):
        try:
            row = _read_metadata_row(paper_dir / "metadata.csv")
        except Exception:
            continue

        paper_id = row.get("hash_blake3") or row.get("hash_sha256") or row.get("id", "")
        if not paper_id:
            continue
        if row.get("processing_status", "") in _SKIP_STATUSES:
            continue
        if str(row.get("duplicate_status", "")).strip().lower() in {"true", "1"}:
            continue
        if paper_id in already_annotated_ids:
            continue

        try:
            publication_year = int(row["publication_year"]) if row.get("publication_year") else None
        except ValueError:
            publication_year = None

        stubs.append(PaperStub(
            paper_id=paper_id,
            doi=row.get("doi", ""),
            title=row.get("title", ""),
            first_author=row.get("first_author", ""),
            journal=row.get("journal", ""),
            publication_year=publication_year,
        ))
    return stubs


def _blank_experiment_dict(schema: SchemaBundle) -> dict:
    """One experiment record with every field null except the fixed
    metadata fields every experiment carries (evidence, confidence) which
    are meaningless for a human annotator and are dropped from the
    template entirely -- a human annotating ground truth doesn't have a
    'confidence' in their own reading, and evidence_text is populated by
    the annotator directly as free text pointing at where they found the
    value, which is more useful to them than an empty placeholder."""
    d = {f.name: None for f in schema.spec.experiment_fields()}
    d["evidence"] = {"source_location": None, "evidence_text": None}
    return d


def select_papers_for_annotation(
    pipeline_output_root: Path,
    n_papers: int,
    seed: int = 42,
    already_annotated_ids: Optional[set[str]] = None,
) -> list[PaperStub]:
    """Randomly sample n_papers eligible papers for annotation, excluding
    any paper_id already present in an existing gold-standard file (so
    running this again to grow the benchmark doesn't re-sample papers
    someone already annotated). Uses a fixed seed by default so the sample
    is reproducible and reportable in a methods section ("30 papers were
    randomly sampled, seed=42, from N eligible papers").
    """
    all_stubs = _list_eligible_paper_stubs(pipeline_output_root, already_annotated_ids)
    if len(all_stubs) < n_papers:
        raise ValueError(
            f"Requested {n_papers} papers for annotation but only "
            f"{len(all_stubs)} eligible papers exist under {pipeline_output_root} "
            "(after excluding already-annotated ones)."
        )
    rng = random.Random(seed)
    return rng.sample(all_stubs, n_papers)


def generate_annotation_template(
    schema: SchemaBundle,
    papers: list[PaperStub],
    n_experiment_slots: int = 3,
) -> list[dict]:
    """Build the fill-in template records. n_experiment_slots is a starting
    number of blank experiment rows per paper (annotators add/delete rows
    as needed once they've actually read the paper and know how many
    distinct catalysts/conditions it reports -- this is just a convenience
    starting point, not a constraint enforced anywhere downstream).
    """
    templates = []
    for paper in papers:
        record = {
            "paper_id": paper.paper_id,
            "doi": paper.doi,
            "title": paper.title,
            "first_author": paper.first_author,
            "journal": paper.journal,
            "publication_year": paper.publication_year,
            "gold_annotator": None,
            "gold_annotated_at": None,
            "gold_notes": None,
            "fields": {f.name: None for f in schema.spec.paper_fields()},
            "experiments": [_blank_experiment_dict(schema) for _ in range(n_experiment_slots)],
        }
        if schema.extract_claims:
            record["claims"] = []
        templates.append(record)
    return templates


def write_annotation_template(
    out_path: Path,
    schema_name: str,
    domains_dir: Path,
    pipeline_output_root: Path,
    n_papers: int,
    seed: int = 42,
    n_experiment_slots: int = 3,
    already_annotated_ids: Optional[set[str]] = None,
) -> Path:
    schema = get_schema(schema_name, domains_dir)
    papers = select_papers_for_annotation(
        pipeline_output_root, n_papers, seed, already_annotated_ids,
    )
    templates = generate_annotation_template(schema, papers, n_experiment_slots)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in templates:
            f.write(json.dumps(record, default=str) + "\n")
    return out_path
