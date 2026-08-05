"""
normalize.py — turns extracted_records.jsonl (one JSON object per paper,
including failures and provenance) into a single clean, flat DataFrame
ready for the downstream ML/optimization package.

Two record shapes are handled, matching pipeline.py's output:

  flat domain   : one row per paper (unchanged from earlier versions)
  multi_experiment domain : one row per experiment, with paper-scope
                  provenance columns duplicated across all experiment rows
                  for that paper. The unit of analysis for ML is the
                  experiment, not the paper — this is what the critique
                  requested, and it's correct.

Design choices that matter for correctness, not just style:
  - Failed extractions are excluded from the clean dataset but kept in their
    own DataFrame so nothing is silently dropped without a trace.
  - Binary "meets spec" targets are pandas nullable booleans (BooleanDtype),
    never plain bool — "not reported in the literature" must stay
    distinguishable from "reported and fails the spec" all the way through
    to the ML stage. This is enforced once, generically, in
    domain_spec.build_binary_targets_generic — not re-implemented per domain.
  - Canonical sibling columns (active_metal_canonical, support_material_canonical)
    are added by canonicalize.apply_canonicalization in the multi_experiment
    path so the ML dataset has a clean join key without losing the original
    text the LLM actually extracted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .canonicalize import apply_canonicalization
from .domain_registry import SchemaBundle, get_schema

_PAPER_PROVENANCE_COLS = [
    "paper_id", "doi", "title", "first_author", "journal",
    "publication_year", "material_class", "is_review",
    "extracted_at", "agreement_status",
]


def load_records(records_path: Path) -> list[dict]:
    records = []
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_success_failure(records: list[dict]) -> tuple[list[dict], list[dict]]:
    successes = [r for r in records if r.get("fields") is not None]
    failures = [r for r in records if r.get("fields") is None]
    return successes, failures


def to_dataframe_flat(successes: list[dict]) -> pd.DataFrame:
    """One row per paper — for flat (non-multi_experiment) domains."""
    rows = []
    for r in successes:
        row = {
            "paper_id": r["paper_id"],
            "doi": r["doi"],
            "title": r["title"],
            "first_author": r["first_author"],
            "journal": r["journal"],
            "publication_year": r["publication_year"],
            "material_class": r["material_class"],
            "is_review": r["is_review"],
            "extracted_at": r["extracted_at"],
            "agreement_status": r["agreement_status"],
            "primary_confidence": r.get("primary_confidence"),
            "cross_check_confidence": r.get("cross_check_confidence"),
            "n_unresolved_fields": len(r.get("unresolved_fields", [])),
            **r["fields"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def to_dataframe_multi_experiment(successes: list[dict], ml_core_fields: list[str] | None = None) -> pd.DataFrame:
    """One row per experiment — for multi_experiment domains.
    Paper-scope provenance columns are duplicated across every experiment
    row for that paper so every row is self-contained for ML.
    Canonical sibling columns for active_metal and support_material are
    added by apply_canonicalization so the ML dataset has a clean,
    normalized join key without losing the LLM's original extracted text.

    Also adds two consumer-facing convenience columns:
      - experiment_id: f"{paper_id}_{experiment_idx}", unique within a
        single extracted_records.jsonl / normalize() call, and stable
        across repeated `build-dataset` runs against THAT SAME jsonl
        (build-dataset itself is a pure function of the file on disk).
        NOT guaranteed stable across a full re-extraction (`llm-extract
        run` again from raw papers): exp_idx position is assigned by
        resolve.py's tier-bucketed matching (identity-key matches, then
        collision-group matches, then position-fallback -- see
        match_experiments), not simply "whatever order the primary model
        listed them in," so a model output change, a canonicalization
        fix, or a schema change can shift which tier an experiment lands
        in and therefore its index. A content-derived hash (e.g. over
        active_metal+support+temperature) was considered and rejected as
        the fix for this: those fields are exactly what
        resolve.py's identity-key collision groups already exist because
        they are NOT sufficiently unique on their own (two experiments
        differing only in loading/pressure/promoter would collide), and
        canonicalization fixes (see canonicalize.py's changelog) change
        those fields' resolved values over time anyway -- a
        content-derived ID would drift on every such improvement, which
        is worse than positional drift, not better. If you need an
        identity that survives a full re-extraction (e.g. to upsert into
        a long-lived vector index across pipeline versions), key on
        evidence_location instead (tied to the paper's own fixed
        structure, e.g. "Table 3, row 2") plus paper_id, and treat that as
        a best-effort re-linking step, not an assumed-stable primary key.
      - ml_core_features_complete: True only if every field named in
        ml_core_fields (see DomainSpec.ml_core_fields) is non-null for
        this row. None (via pandas' nullable boolean) if ml_core_fields is
        empty/not declared for this domain, so a domain that hasn't
        opted in doesn't get a misleading all-False column.
    """
    ml_core_fields = ml_core_fields or []
    rows = []
    for r in successes:
        paper_provenance = {col: r.get(col) for col in _PAPER_PROVENANCE_COLS}
        paper_provenance["n_experiments"] = len(r.get("experiments", []))
        paper_provenance["n_claims"] = len(r.get("claims", []))

        # Paper-scope fields (e.g. feedstock_type, doi, etc.)
        paper_fields = r.get("fields", {})

        experiments = r.get("experiments", [])
        for exp_idx, exp in enumerate(experiments):
            # Flatten evidence span into sibling columns
            evidence = exp.pop("evidence", {}) or {}
            exp_meta = {
                "experiment_id": f"{r['paper_id']}_{exp_idx}",
                "experiment_idx": exp_idx,
                "experiment_match_status": exp.pop("experiment_match_status", None),
                "match_method": exp.pop("match_method", None),
                "n_unresolved_experiment_fields": len(exp.pop("unresolved_fields", [])),
                "evidence_location": evidence.get("source_location"),
                "evidence_text": evidence.get("evidence_text"),
            }

            row = {
                **paper_provenance,
                **paper_fields,
                **exp_meta,
                **exp,
            }
            row = apply_canonicalization(row)

            if ml_core_fields:
                row["ml_core_features_complete"] = all(
                    row.get(f) is not None for f in ml_core_fields
                )

            rows.append(row)

    df = pd.DataFrame(rows)
    if ml_core_fields and "ml_core_features_complete" in df.columns:
        df["ml_core_features_complete"] = df["ml_core_features_complete"].astype("boolean")
    return df


def normalize(records_path: Path, schema_name: str, domains_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Returns a dict with DataFrames:
      'clean'    — successfully extracted records, with binary targets added
                   (one row per experiment for multi_experiment domains,
                    one row per paper for flat domains)
      'failures' — papers where extraction failed outright (both models)
      'unresolved_field_flags' — one row per (paper, field) that needed
                                  escalation but still has no majority value
      'claims'   — (multi_experiment only) all extracted causal/comparative
                   claims, one row per claim, linked by paper_id
    """
    records = load_records(records_path)
    successes, failures = split_success_failure(records)

    if not successes:
        raise ValueError(
            f"No successful extractions in {records_path} — nothing to normalize. "
            "Check the Ollama server is reachable and the prompt/schema match."
        )

    schema: SchemaBundle = get_schema(schema_name, domains_dir)

    if schema.multi_experiment:
        df = to_dataframe_multi_experiment(successes, ml_core_fields=schema.spec.ml_core_fields)
    else:
        df = to_dataframe_flat(successes)

    df = schema.build_binary_targets(df)

    unresolved_rows = []
    if schema.multi_experiment:
        # Unresolved fields live at experiment level in multi_experiment mode
        for r in successes:
            for exp in r.get("experiments", []):
                for field in exp.get("unresolved_fields", []):
                    unresolved_rows.append({"paper_id": r["paper_id"], "field": field})
    else:
        for r in successes:
            for field in r.get("unresolved_fields", []):
                unresolved_rows.append({"paper_id": r["paper_id"], "field": field})

    claims_rows = []
    if schema.multi_experiment:
        for r in successes:
            for claim in r.get("claims", []):
                claims_rows.append({"paper_id": r["paper_id"], **claim})

    return {
        "clean": df,
        "failures": pd.DataFrame(failures) if failures else pd.DataFrame(),
        "unresolved_field_flags": pd.DataFrame(unresolved_rows),
        "claims": pd.DataFrame(claims_rows),
    }
