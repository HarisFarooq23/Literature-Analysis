"""
writer.py — emits the final, versioned output contract for downstream
ML/optimization packages. Everything a consumer needs lives in one
directory:

    extracted_dataset.parquet     — typed, the primary artifact
    extracted_dataset.csv         — same data, human-inspectable mirror
    extracted_dataset.schema.json — JSON Schema describing every column
    data_dictionary.md            — human-readable column/unit reference
    manifest.json                 — run provenance: models, timestamps, counts
    failed_extractions.csv        — papers neither model could extract
    unresolved_field_flags.csv    — (paper, field) pairs flagged as conflicting
                                     with no majority even after escalation

A downstream package should depend on nothing but this directory. Nothing
in this module is domain-specific — it documents whatever domain.yaml says,
generically, for any number of catalysis topics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import __version__
from .config import AppConfig
from .domain_registry import SchemaBundle, get_schema


def _multi_experiment_meta_props(schema: SchemaBundle) -> dict:
    """Provenance/meta columns that to_dataframe_multi_experiment adds on
    top of the LLM-extracted fields themselves. These existed in every
    multi_experiment dataset already (experiment_idx, evidence_text, etc.)
    but were never documented in the schema/data-dictionary output --
    fixed here alongside adding experiment_id and ml_core_features_complete,
    rather than documenting only the two new ones and leaving the
    pre-existing gap in place.
    """
    props = {
        "experiment_id": {
            "type": "string",
            "description": "f'{paper_id}_{experiment_idx}' -- unique within this dataset build. Stable across repeated build-dataset runs against the same extracted_records.jsonl; NOT guaranteed stable across a full re-extraction (llm-extract run again), since experiment order is assigned by tier-bucketed cross-model matching, not fixed paper structure. For an ID that survives re-extraction, prefer (paper_id, evidence_location).",
        },
        "experiment_idx": {"type": "integer", "description": "Position of this experiment within its paper's experiments list (0-indexed)."},
        "n_experiments": {"type": "integer", "description": "Total experiments extracted from this row's paper (duplicated across all its experiment rows)."},
        "n_claims": {"type": "integer", "description": "Total causal/comparative claims extracted from this row's paper."},
        "experiment_match_status": {
            "type": ["string", "null"],
            "description": "full_agreement | resolved_after_escalation | unresolved_conflict | single_model_only -- cross-model agreement status for THIS experiment specifically (distinct from the paper-level agreement_status).",
        },
        "match_method": {
            "type": ["string", "null"],
            "description": "identity_key | position -- how this experiment was paired across primary/cross_check/tie_breaker. 'position' means no unique catalyst-identity match was found; treat with extra scrutiny.",
        },
        "n_unresolved_experiment_fields": {"type": "integer", "description": "Count of fields within this experiment that no model majority could resolve."},
        "evidence_location": {"type": ["string", "null"], "description": "Where in the source paper this experiment's data came from, e.g. 'Table 3, page 8'."},
        "evidence_text": {"type": ["string", "null"], "description": "Verbatim supporting sentence/table row for this experiment, as quoted by the extracting model."},
    }
    if schema.spec.ml_core_fields:
        props["ml_core_features_complete"] = {
            "type": ["boolean", "null"],
            "description": (
                "True only if every field in "
                f"{schema.spec.ml_core_fields} is non-null for this "
                "experiment -- a ready-made filter for a shallow-ML "
                "structure-property-performance dataset. Null is not "
                "possible in practice (always True or False) but kept "
                "nullable for consistency with every other boolean "
                "column's not-reported-vs-false distinction."
            ),
        }
    return props


def _multi_experiment_meta_dictionary_rows(schema: SchemaBundle) -> list[str]:
    rows = [
        "| `experiment_id` | Key: `{paper_id}_{experiment_idx}`. Stable within one dataset build (repeated `build-dataset` runs against the same extracted_records.jsonl); NOT guaranteed stable across a full re-extraction, since experiment order comes from cross-model matching, not fixed paper structure. For an ID surviving re-extraction, join on `(paper_id, evidence_location)` instead. | - |",
        "| `experiment_idx` | Position of this experiment within its paper (0-indexed) | - |",
        "| `n_experiments` / `n_claims` | Total experiments/claims extracted from this row's paper | count |",
        "| `experiment_match_status` | Cross-model agreement for THIS experiment specifically | categorical |",
        "| `match_method` | `identity_key` \\\\| `position` -- how this experiment was paired across models; `position` warrants extra scrutiny | categorical |",
        "| `n_unresolved_experiment_fields` | Fields within this experiment with no resolved majority value | count |",
        "| `evidence_location` / `evidence_text` | Where in the paper this experiment's data came from, and the verbatim supporting text | - |",
    ]
    if schema.spec.ml_core_fields:
        rows.append(
            "| `ml_core_features_complete` | True iff every one of "
            f"{', '.join(f'`{f}`' for f in schema.spec.ml_core_fields)} "
            "is non-null for this row -- filter on this for a "
            "ready-to-model shallow-ML subset | boolean |"
        )
    return rows


def _build_json_schema(schema: SchemaBundle) -> dict:
    extraction_schema = schema.ExtractionFields.model_json_schema()

    provenance_props = {
        "paper_id": {"type": "string", "description": "Stable blake3 hash from scientific-pdf-pipeline; primary key."},
        "doi": {"type": "string"},
        "title": {"type": "string"},
        "first_author": {"type": "string"},
        "journal": {"type": "string"},
        "publication_year": {"type": ["integer", "null"]},
        "material_class": {"type": "string"},
        "is_review": {"type": "boolean"},
        "extracted_at": {"type": "string", "format": "date-time"},
        "agreement_status": {
            "type": "string",
            "enum": ["full_agreement", "resolved_after_escalation", "unresolved_conflict"],
        },
        "primary_confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "cross_check_confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "n_unresolved_fields": {"type": "integer"},
    }
    if schema.multi_experiment:
        provenance_props = {**provenance_props, **_multi_experiment_meta_props(schema)}

    binary_target_props = {
        bt.name: {
            "type": ["boolean", "null"],
            "description": bt.description or f"{bt.field} {bt.op} {bt.value}. "
                            "Null means the underlying value was not reported, not that the spec failed.",
        }
        for bt in schema.spec.binary_targets
    }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"llm_extract clean dataset — domain: {schema.domain_name}",
        "type": "object",
        "properties": {**provenance_props, **extraction_schema.get("properties", {}), **binary_target_props},
    }


def _build_data_dictionary(schema: SchemaBundle) -> str:
    lines = [
        f"# Data Dictionary — llm_extract clean dataset (domain: {schema.domain_name})",
        "",
        schema.spec.description or "",
        "",
        "| Column | Description | Unit |",
        "|---|---|---|",
        "| `paper_id` | Stable blake3 hash, primary key, joins back to scientific-pdf-pipeline output | - |",
        "| `doi` / `title` / `first_author` / `journal` / `publication_year` | Bibliographic provenance | - |",
        "| `agreement_status` | `full_agreement` \\| `resolved_after_escalation` \\| `unresolved_conflict` | categorical |",
        "| `primary_confidence` / `cross_check_confidence` | Each model's self-reported confidence | 0-1 |",
        "| `n_unresolved_fields` | Count of fields no model majority could resolve for this paper | count |",
    ]
    if schema.multi_experiment:
        lines.extend(_multi_experiment_meta_dictionary_rows(schema))
    for col, label, unit in schema.FIELD_DOCS:
        lines.append(f"| `{col}` | {label} | {unit} |")
    for bt in schema.spec.binary_targets:
        desc = bt.description or f"True if {bt.field} {bt.op} {bt.value}"
        lines.append(f"| `{bt.name}` | {desc} (nullable: null = not reported, not 'fails spec') | boolean |")
    lines.append("")
    lines.append(
        "**Nullability note:** every extracted field and binary target uses null/NA to mean "
        "'not reported in the source paper.' Treat null and False as distinct categories in "
        "any downstream model -- collapsing them reintroduces the exact bias the literature "
        "already has (most papers under-report properties they did not optimize for)."
    )
    return "\n".join(lines)


def write_outputs(
    config: AppConfig,
    clean_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    unresolved_df: pd.DataFrame,
) -> Path:
    out_dir = config.paths.extraction_output_root
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = get_schema(config.extraction.schema_name, config.paths.domains_dir)

    clean_df.to_parquet(out_dir / "extracted_dataset.parquet", index=False)
    clean_df.to_csv(out_dir / "extracted_dataset.csv", index=False)

    if not failures_df.empty:
        failures_df.to_csv(out_dir / "failed_extractions.csv", index=False)
    if not unresolved_df.empty:
        unresolved_df.to_csv(out_dir / "unresolved_field_flags.csv", index=False)

    (out_dir / "extracted_dataset.schema.json").write_text(
        json.dumps(_build_json_schema(schema), indent=2), encoding="utf-8"
    )
    (out_dir / "data_dictionary.md").write_text(
        _build_data_dictionary(schema), encoding="utf-8"
    )

    manifest = {
        "llm_extract_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain": schema.domain_name,
        "models": {
            "primary": config.models.primary,
            "cross_check": config.models.cross_check,
            "tie_breaker": config.models.tie_breaker,
            "vision": config.models.vision,
        },
        "figure_vision_enabled": config.extraction.enable_figure_vision,
        "tables_included": config.extraction.include_tables,
        "n_papers_clean": int(len(clean_df)),
        "n_papers_failed": int(len(failures_df)),
        "n_unresolved_field_flags": int(len(unresolved_df)),
        "agreement_status_counts": clean_df["agreement_status"].value_counts().to_dict()
            if "agreement_status" in clean_df.columns else {},
        "source_pipeline_output_root": str(config.paths.pipeline_output_root),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return out_dir
