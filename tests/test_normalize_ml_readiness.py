"""
test_normalize_ml_readiness.py — coverage for the two consumer-facing
columns added to to_dataframe_multi_experiment: experiment_id (stable key
for RAG/vector-index use) and ml_core_features_complete (shallow-ML
dataset-readiness filter driven by DomainSpec.ml_core_fields).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from llm_extract.normalize import normalize, to_dataframe_multi_experiment

_DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains"


def _paper_record(paper_id: str, experiments: list[dict], **overrides) -> dict:
    base = {
        "paper_id": paper_id, "doi": "10.1/x", "title": "t", "first_author": "A",
        "journal": "J", "publication_year": 2024, "material_class": "saf_hefa",
        "is_review": False, "extracted_at": "2026-01-01T00:00:00+00:00",
        "agreement_status": "full_agreement",
        "primary_confidence": None, "cross_check_confidence": None,
        "unresolved_fields": [],
        "fields": {"feedstock_type": "waste_cooking_oil"},
        "experiments": experiments,
        "claims": [],
    }
    base.update(overrides)
    return base


def _experiment(**overrides) -> dict:
    base = {
        "active_metal": "NiMo", "support_type": "metal_oxide",
        "metal_loading_wt_pct": 10.0, "reaction_temperature_C": 350.0,
        "reaction_pressure_bar": 30.0, "oil_conversion_pct": 95.0,
        "evidence": {"source_location": "Table 2", "evidence_text": "NiMo/Al2O3 gave 95% conversion."},
        "experiment_confidence": 0.9,
        "unresolved_fields": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# experiment_id
# ---------------------------------------------------------------------------

def test_experiment_id_is_paper_id_underscore_index():
    df = to_dataframe_multi_experiment([
        _paper_record("paperA", [_experiment(), _experiment()]),
    ])
    assert list(df["experiment_id"]) == ["paperA_0", "paperA_1"]


def test_experiment_id_unique_across_multiple_papers():
    df = to_dataframe_multi_experiment([
        _paper_record("paperA", [_experiment()]),
        _paper_record("paperB", [_experiment(), _experiment()]),
    ])
    assert df["experiment_id"].is_unique
    assert set(df["experiment_id"]) == {"paperA_0", "paperB_0", "paperB_1"}


# ---------------------------------------------------------------------------
# ml_core_features_complete
# ---------------------------------------------------------------------------

def test_no_completeness_column_when_ml_core_fields_empty():
    df = to_dataframe_multi_experiment(
        [_paper_record("paperA", [_experiment()])], ml_core_fields=[],
    )
    assert "ml_core_features_complete" not in df.columns


def test_complete_row_flagged_true():
    df = to_dataframe_multi_experiment(
        [_paper_record("paperA", [_experiment()])],
        ml_core_fields=["active_metal", "support_type", "reaction_temperature_C", "oil_conversion_pct"],
    )
    assert df.loc[0, "ml_core_features_complete"] == True


def test_incomplete_row_flagged_false_not_dropped():
    """A row missing one core field must be flagged False, not silently
    excluded -- the consumer decides whether to filter it out."""
    df = to_dataframe_multi_experiment(
        [_paper_record("paperA", [_experiment(oil_conversion_pct=None)])],
        ml_core_fields=["active_metal", "support_type", "reaction_temperature_C", "oil_conversion_pct"],
    )
    assert len(df) == 1
    assert df.loc[0, "ml_core_features_complete"] == False


def test_completeness_column_is_nullable_boolean_dtype():
    df = to_dataframe_multi_experiment(
        [_paper_record("paperA", [_experiment()])],
        ml_core_fields=["active_metal"],
    )
    assert str(df["ml_core_features_complete"].dtype) == "boolean"


def test_mixed_complete_and_incomplete_rows_across_papers():
    df = to_dataframe_multi_experiment(
        [
            _paper_record("paperA", [_experiment()]),
            _paper_record("paperB", [_experiment(active_metal=None)]),
        ],
        ml_core_fields=["active_metal", "support_type", "reaction_temperature_C", "oil_conversion_pct"],
    )
    flags = dict(zip(df["experiment_id"], df["ml_core_features_complete"]))
    assert flags == {"paperA_0": True, "paperB_0": False}


# ---------------------------------------------------------------------------
# End-to-end against the real shipped saf_hefa domain (which declares
# ml_core_fields) -- regression check that the YAML's ml_core_fields list
# stays in sync with actual field names as the domain evolves.
# ---------------------------------------------------------------------------

def test_end_to_end_against_real_saf_hefa_domain(tmp_path: Path):
    records_path = tmp_path / "extracted_records.jsonl"
    rows = [
        _paper_record("paperA", [
            _experiment(),
            _experiment(active_metal="Pt", oil_conversion_pct=None),  # incomplete
        ]),
    ]
    with open(records_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    tables = normalize(records_path, schema_name="saf_hefa", domains_dir=_DOMAINS_DIR)
    clean = tables["clean"]
    assert "experiment_id" in clean.columns
    assert "ml_core_features_complete" in clean.columns
    assert list(clean["experiment_id"]) == ["paperA_0", "paperA_1"]
    assert clean.loc[0, "ml_core_features_complete"] == True
    assert clean.loc[1, "ml_core_features_complete"] == False
