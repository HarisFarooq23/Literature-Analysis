"""
test_pipeline_multi_experiment.py — end-to-end integration test for the
multi_experiment pipeline path without a live Ollama server.

Mocks OllamaExtractor.extract so the test runs in CI with no GPU/network
requirement, but exercises every real code path in pipeline._run_multi_experiment_paper:
  - multi-catalyst extraction (3 experiments per paper)
  - experiment-list matching and resolution
  - claim merging
  - evidence span propagation
  - JSONL output structure
  - normalize.py flattening to one-row-per-experiment DataFrame
  - canonicalize.py sibling columns on the flat output
"""

from __future__ import annotations

import json
import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel

from llm_extract.canonicalize import apply_canonicalization
from llm_extract.domain_spec import (
    ClaimRecord,
    DomainSpec,
    EvidenceSpan,
    FieldSpec,
    build_pydantic_model,
)
from llm_extract.ollama_client import CallResult
from llm_extract.reader import PaperRecord
from llm_extract.resolve import resolve_experiment_lists


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper_record(paper_dir: Path) -> PaperRecord:
    return PaperRecord(
        paper_id="abc12345",
        title="Hydroprocessing of WCO over Ni catalysts",
        first_author="Test Author",
        journal="Fuel",
        publication_year=2024,
        doi="10.1/test",
        material_class="saf_hefa",
        is_review=False,
        output_dir=paper_dir,
        abstract="",
        text_for_extraction="dummy text",
    )


def _make_experiment_model():
    """Build a minimal multi_experiment domain model for testing."""
    spec = DomainSpec(
        domain_name="test_me_domain",
        prompt_template_path="x.txt",
        multi_experiment=True,
        extract_claims=True,
        min_experiments=1,
        max_experiments=10,
        fields=[
            FieldSpec(name="feedstock_type", type="str", scope="paper"),
            FieldSpec(name="active_metal",    type="str", scope="experiment"),
            FieldSpec(name="support_material",type="str", scope="experiment"),
            FieldSpec(name="conversion_pct",  type="float", scope="experiment"),
        ],
    )
    return build_pydantic_model(spec)


PaperModel = _make_experiment_model()
ExperimentModel = PaperModel.__experiment_model__


def _make_paper_instance(experiments_data: list[dict], claims: list[dict] | None = None):
    exps = [ExperimentModel(**e) for e in experiments_data]
    return PaperModel(
        feedstock_type="waste_cooking_oil",
        experiments=exps,
        claims=[ClaimRecord(**c) for c in (claims or [])],
    )


def _make_call_result(parsed: BaseModel) -> CallResult:
    return CallResult(
        model="test_model", ok=True, parsed=parsed,
        raw_response="{}", error=None, attempts=1, elapsed_seconds=0.1,
    )


# ---------------------------------------------------------------------------
# Test: experiment-list resolution (core of the critique)
# ---------------------------------------------------------------------------

def test_three_catalysts_produce_three_experiment_records():
    """A paper comparing Ni/Al2O3, NiMo/SAPO-11, Pt/Al2O3 must produce
    exactly 3 resolved experiment records, not 1."""
    primary = _make_paper_instance([
        {"active_metal": "Ni",   "support_material": "Al2O3",   "conversion_pct": 78.0, "experiment_confidence": 0.9},
        {"active_metal": "NiMo", "support_material": "SAPO-11", "conversion_pct": 95.0, "experiment_confidence": 0.92},
        {"active_metal": "Pt",   "support_material": "Al2O3",   "conversion_pct": 88.0, "experiment_confidence": 0.91},
    ])
    cross = _make_paper_instance([
        {"active_metal": "Ni",   "support_material": "Al2O3",   "conversion_pct": 77.5, "experiment_confidence": 0.88},
        {"active_metal": "NiMo", "support_material": "SAPO-11", "conversion_pct": 94.8, "experiment_confidence": 0.90},
        {"active_metal": "Pt",   "support_material": "Al2O3",   "conversion_pct": 87.5, "experiment_confidence": 0.89},
    ])

    results = resolve_experiment_lists(
        primary.experiments, cross.experiments, None,
        ExperimentModel, numeric_tolerance=0.10,
    )

    assert len(results) == 3
    metals = sorted(r.resolved["active_metal"] for r in results)
    assert metals == ["Ni", "NiMo", "Pt"]
    assert all(r.experiment_match_status == "full_agreement" for r in results)


def test_evidence_span_is_preserved_per_experiment():
    """Evidence spans (source_location, evidence_text) on each experiment
    must flow through resolution unchanged."""
    primary = _make_paper_instance([
        {
            "active_metal": "NiMo", "support_material": "SAPO-11",
            "conversion_pct": 95.0, "experiment_confidence": 0.93,
            "evidence": {
                "source_location": "Table 3, page 8",
                "evidence_text": "NiMo/SAPO-11 gave 95% conversion at 320°C",
            },
        }
    ])
    cross = _make_paper_instance([
        {
            "active_metal": "NiMo", "support_material": "SAPO-11",
            "conversion_pct": 94.8, "experiment_confidence": 0.91,
            "evidence": {
                "source_location": "Table 3, p.8",
                "evidence_text": "NiMo/SAPO-11 achieved 94.8% conversion",
            },
        }
    ])

    results = resolve_experiment_lists(
        primary.experiments, cross.experiments, None,
        ExperimentModel, numeric_tolerance=0.10,
    )
    assert len(results) == 1
    ev_raw = results[0].resolved.get("evidence")
    # evidence can be a dict or an EvidenceSpan BaseModel depending on resolve path
    ev = ev_raw if isinstance(ev_raw, dict) else (ev_raw.model_dump() if ev_raw else {})
    # Evidence comes from primary (its values win on agreement)
    assert ev.get("source_location") == "Table 3, page 8"
    assert ev.get("evidence_text") and "95%" in ev["evidence_text"]


def test_claims_are_merged_deduplicated():
    """Claims from primary and cross-check are unioned, with identical
    (subject, relation, object) triples deduplicated."""
    from llm_extract.pipeline import _merge_claims

    primary_claims = [
        {"subject": "Ni loading", "relation": "increases", "object": "conversion",
         "evidence_text": "Increasing Ni loading improved conversion.", "confidence": 0.9},
        {"subject": "Brønsted acidity", "relation": "increases", "object": "hydroisomerization",
         "evidence_text": "Higher acid sites improved isomerization.", "confidence": 0.88},
    ]
    cross_claims = [
        # Exact same triple as primary[0] — should be de-duplicated
        {"subject": "Ni loading", "relation": "increases", "object": "conversion",
         "evidence_text": "Higher Ni loading gave higher conversion.", "confidence": 0.85},
        # New claim only in cross — should be kept
        {"subject": "pore size", "relation": "decreases", "object": "coke formation",
         "evidence_text": "Larger pores reduced coking.", "confidence": 0.82},
    ]

    merged = _merge_claims(primary_claims, cross_claims)
    # Should have 3 unique claims (not 4)
    assert len(merged) == 3
    subjects = {c["subject"] for c in merged}
    assert subjects == {"Ni loading", "Brønsted acidity", "pore size"}


# ---------------------------------------------------------------------------
# Test: canonicalization runs on output rows
# ---------------------------------------------------------------------------

def test_canonicalization_adds_sibling_columns_on_experiment_rows():
    """NiMo -> Ni-Mo, HZSM-5 -> ZSM-5, both as canonical siblings without
    touching the original extracted text."""
    row = {
        "active_metal": "NiMo",
        "support_material": "HZSM-5",
        "conversion_pct": 95.0,
    }
    out = apply_canonicalization(row)
    assert out["active_metal"] == "NiMo"          # original untouched
    assert out["active_metal_canonical"] == "Ni-Mo"
    assert out["support_material"] == "HZSM-5"
    assert out["support_material_canonical"] == "ZSM-5"


# ---------------------------------------------------------------------------
# Test: full pipeline run end-to-end with mocked Ollama calls
# ---------------------------------------------------------------------------

def _build_fake_multi_experiment_domain(tmp_path: Path):
    """Write a minimal but complete domain YAML + prompt + paper directory
    to tmp_path so the pipeline can be called for real."""
    domains_dir = tmp_path / "domains"
    domains_dir.mkdir()
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    (domains_dir / "test_me.yaml").write_text(
        """
domain_name: test_me
description: integration test domain
prompt_template_path: "../prompts/test_me.txt"
multi_experiment: true
extract_claims: true
min_experiments: 1
max_experiments: 10
fields:
  - name: feedstock_type
    type: str
    scope: paper
  - name: active_metal
    type: str
    scope: experiment
    required: true
  - name: support_material
    type: str
    scope: experiment
  - name: conversion_pct
    type: float
    scope: experiment
binary_targets:
  - name: high_conversion
    field: conversion_pct
    op: ">="
    value: 90.0
    description: "conversion >= 90%"
""",
        encoding="utf-8",
    )
    (prompts_dir / "test_me.txt").write_text(
        "Extract data.\nSchema: {schema}\nText: {text}",
        encoding="utf-8",
    )

    processed_root = tmp_path / "processed"
    paper_dir = processed_root / "TestAuthor_Fuel_2024_abc12345"
    paper_dir.mkdir(parents=True)
    (paper_dir / "metadata.csv").write_text(
        "hash_blake3,title,first_author,journal,publication_year,doi,"
        "material_class,is_review,processing_status,duplicate_status\n"
        "abc12345,WCO Hydroprocessing,Test Author,Fuel,2024,10.1/test,"
        "saf_hefa,False,completed,False\n",
        encoding="utf-8",
    )
    (paper_dir / "cleaned_full_text.txt").write_text(
        "Full text about Ni, NiMo, and Pt catalysts for hydroprocessing.",
        encoding="utf-8",
    )

    out_root = tmp_path / "output"
    out_root.mkdir()
    return domains_dir, processed_root, out_root


def test_full_pipeline_run_multi_experiment_produces_experiment_records(tmp_path):
    """Run the pipeline against a fake paper with mocked Ollama responses.
    Primary and cross-check both return 3 experiments -> output .jsonl must
    contain 3 experiment entries for the single paper."""
    domains_dir, processed_root, out_root = _build_fake_multi_experiment_domain(tmp_path)

    from llm_extract.domain_registry import get_schema
    schema = get_schema("test_me", domains_dir)
    PaperCls = schema.ExtractionFields
    ExpCls = PaperCls.__experiment_model__

    def make_paper_json(metals_and_conversions: list[tuple[str, float]]) -> PaperCls:
        return PaperCls(
            feedstock_type="waste_cooking_oil",
            experiments=[
                ExpCls(
                    active_metal=m, support_material="Al2O3",
                    conversion_pct=c, experiment_confidence=0.9,
                )
                for m, c in metals_and_conversions
            ],
            claims=[
                ClaimRecord(
                    subject="Ni loading", relation="increases", object="conversion",
                    evidence_text="Higher Ni loading increased conversion.", confidence=0.9,
                )
            ],
        )

    primary_parsed = make_paper_json([("Ni", 78.0), ("NiMo", 95.0), ("Pt", 88.0)])
    cross_parsed   = make_paper_json([("Ni", 77.5), ("NiMo", 94.8), ("Pt", 87.5)])

    def fake_extract(model, prompt, response_model):
        if model == "primary_model":
            return _make_call_result(primary_parsed)
        return _make_call_result(cross_parsed)

    from llm_extract.config import AppConfig, ExtractionConfig, ModelsConfig, PathsConfig, LoggingConfig
    config = AppConfig(
        paths=PathsConfig(
            pipeline_output_root=processed_root,
            extraction_output_root=out_root,
            domains_dir=domains_dir,
        ),
        models=ModelsConfig(
            primary="primary_model",
            cross_check="cross_check_model",
            tie_breaker="tie_breaker_model",
            host="http://localhost:11434",
        ),
        extraction=ExtractionConfig(
            schema_name="test_me",
            multi_experiment=True,
        ),
        logging=LoggingConfig(level="INFO", log_dir=out_root / "logs"),
    )

    with patch("llm_extract.pipeline.OllamaExtractor.extract", side_effect=fake_extract):
        from llm_extract.pipeline import run
        records_path = run(config)

    assert records_path.exists()
    lines = [json.loads(l) for l in records_path.read_text().strip().splitlines()]
    assert len(lines) == 1   # one paper

    paper_record = lines[0]
    assert paper_record["agreement_status"] == "full_agreement"
    experiments = paper_record["experiments"]
    assert len(experiments) == 3
    metals = sorted(e["active_metal"] for e in experiments)
    assert metals == ["Ni", "NiMo", "Pt"]

    claims = paper_record["claims"]
    assert len(claims) == 1
    assert claims[0]["subject"] == "Ni loading"


def test_normalize_flattens_experiments_to_one_row_per_experiment(tmp_path):
    """After the pipeline run, normalize.py must expand the experiments list
    into one DataFrame row per experiment, not one row per paper."""
    domains_dir, processed_root, out_root = _build_fake_multi_experiment_domain(tmp_path)

    # Write a synthetic extracted_records.jsonl with 2 papers × 3 experiments
    records_path = out_root / "extracted_records.jsonl"
    for paper_idx in range(2):
        record = {
            "paper_id": f"paper{paper_idx:04d}",
            "doi": f"10.1/test{paper_idx}",
            "title": f"Paper {paper_idx}",
            "first_author": "Author",
            "journal": "Fuel",
            "publication_year": 2024,
            "material_class": "saf_hefa",
            "is_review": False,
            "extracted_at": "2024-01-01T00:00:00+00:00",
            "agreement_status": "full_agreement",
            "models_used": ["m1", "m2"],
            "fields": {"feedstock_type": "waste_cooking_oil"},
            "experiments": [
                {"active_metal": "Ni",   "support_material": "Al2O3",   "conversion_pct": 78.0,
                 "experiment_confidence": 0.9, "evidence": {}, "experiment_match_status": "full_agreement", "unresolved_fields": []},
                {"active_metal": "NiMo", "support_material": "SAPO-11", "conversion_pct": 95.0,
                 "experiment_confidence": 0.92, "evidence": {}, "experiment_match_status": "full_agreement", "unresolved_fields": []},
                {"active_metal": "Pt",   "support_material": "Al2O3",   "conversion_pct": 88.0,
                 "experiment_confidence": 0.91, "evidence": {}, "experiment_match_status": "full_agreement", "unresolved_fields": []},
            ],
            "claims": [],
        }
        with open(records_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    from llm_extract.normalize import normalize
    tables = normalize(records_path, "test_me", domains_dir)

    df = tables["clean"]
    # 2 papers × 3 experiments each = 6 rows
    assert len(df) == 6
    assert "paper_id" in df.columns
    assert "active_metal" in df.columns
    assert "experiment_idx" in df.columns
    assert "active_metal_canonical" in df.columns   # canonicalization was applied

    # Binary target was applied per-experiment row
    assert "high_conversion" in df.columns
    high = df[df["active_metal"] == "NiMo"]["high_conversion"]
    assert all(high == True)
    low = df[df["active_metal"] == "Ni"]["high_conversion"]
    assert all(low == False)
