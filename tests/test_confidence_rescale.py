"""
test_confidence_rescale.py — dedicated tests for _confidence_rescale_validator.

Before this file existed, there was NO direct test of the rescale behavior
anywhere in the suite (only schema-shape checks confirming the field exists
with ge=0.0/le=1.0). That gap is exactly how the bug this file guards
against shipped unnoticed: the rescaler covered top-level scalar confidence
fields (extraction_confidence, experiment_confidence) but never walked into
the nested `claims` list, so a model returning claims[*].confidence on a
Likert (1-3/1-5) scale instead of [0,1] failed Pydantic validation
identically on every retry and lost the whole paper. This happened in
production (paper 2dfd2478, gemma3:12b, confidence=3 on all three claims,
3/3 retries) before this fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_extract.domain_spec import DomainSpec, FieldSpec, build_pydantic_model


def _flat_model():
    spec = DomainSpec(
        domain_name="test_flat",
        prompt_template_path="x.txt",
        multi_experiment=False,
        fields=[FieldSpec(name="yield_pct", type="float", scope="experiment")],
    )
    return build_pydantic_model(spec)


def _multi_experiment_model(extract_claims: bool = True):
    spec = DomainSpec(
        domain_name="test_me",
        prompt_template_path="x.txt",
        multi_experiment=True,
        extract_claims=extract_claims,
        min_experiments=1,
        max_experiments=10,
        fields=[
            FieldSpec(name="feedstock_type", type="str", scope="paper"),
            FieldSpec(name="active_metal", type="str", scope="experiment"),
        ],
    )
    return build_pydantic_model(spec)


# ---------------------------------------------------------------------------
# Flat-domain extraction_confidence
# ---------------------------------------------------------------------------

def test_flat_extraction_confidence_percentage_is_rescaled():
    Model = _flat_model()
    instance = Model(yield_pct=85.0, extraction_confidence=95)
    assert instance.extraction_confidence == pytest.approx(0.95)


def test_flat_extraction_confidence_already_decimal_is_unchanged():
    Model = _flat_model()
    instance = Model(yield_pct=85.0, extraction_confidence=0.8)
    assert instance.extraction_confidence == pytest.approx(0.8)


def test_flat_extraction_confidence_implausible_value_is_clamped_to_one():
    Model = _flat_model()
    instance = Model(yield_pct=85.0, extraction_confidence=150)
    assert instance.extraction_confidence == 1.0


# ---------------------------------------------------------------------------
# Multi-experiment experiment_confidence (nested model, self-healing --
# regression guard that the refactor in this file didn't break this path)
# ---------------------------------------------------------------------------

def test_experiment_confidence_percentage_is_rescaled_per_experiment():
    PaperModel = _multi_experiment_model(extract_claims=False)
    instance = PaperModel(
        feedstock_type="waste_cooking_oil",
        experiments=[
            {"active_metal": "NiMo", "evidence": {}, "experiment_confidence": 90},
            {"active_metal": "Pt", "evidence": {}, "experiment_confidence": 0.6},
        ],
    )
    assert instance.experiments[0].experiment_confidence == pytest.approx(0.9)
    assert instance.experiments[1].experiment_confidence == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# claims[].confidence -- THE bug. Reproduces the exact production failure.
# ---------------------------------------------------------------------------

def test_claims_confidence_likert_scale_is_rescaled_not_rejected():
    """Direct reproduction of the paper 2dfd2478 production failure: model
    returns confidence=3 (a 1-3/1-5 Likert misinterpretation) on every claim.
    Before the fix, this raised ValidationError (le=1.0) identically on
    every retry and the paper was lost. After the fix, 3 is treated as an
    implausible-as-decimal value and rescaled (via the same >1.0 rule used
    for extraction_confidence): 3 is not a perccentage either, so it's
    clamped to 1.0 rather than divided (3/100=0.03 would UNDERSTATE
    confidence for what a "3" almost certainly meant -- high confidence on
    a small integer scale). Either way, the paper is no longer lost.
    """
    PaperModel = _multi_experiment_model(extract_claims=True)
    instance = PaperModel(
        feedstock_type="waste_cooking_oil",
        experiments=[{"active_metal": "NiMo", "evidence": {}, "experiment_confidence": 0.9}],
        claims=[
            {"subject": "Ni loading", "relation": "increases", "object": "conversion",
             "evidence_text": "Higher Ni loading increased conversion.", "confidence": 3},
            {"subject": "Acidity", "relation": "correlates_with", "object": "selectivity",
             "evidence_text": "Acidity correlated with selectivity.", "confidence": 3},
            {"subject": "Temperature", "relation": "increases", "object": "deoxygenation",
             "evidence_text": "Higher temperature increased deoxygenation.", "confidence": 3},
        ],
    )
    assert len(instance.claims) == 3
    for claim in instance.claims:
        assert 0.0 <= claim.confidence <= 1.0


def test_claims_confidence_percentage_is_rescaled():
    PaperModel = _multi_experiment_model(extract_claims=True)
    instance = PaperModel(
        feedstock_type="waste_cooking_oil",
        experiments=[{"active_metal": "NiMo", "evidence": {}, "experiment_confidence": 0.9}],
        claims=[
            {"subject": "Ni loading", "relation": "increases", "object": "conversion",
             "evidence_text": "text", "confidence": 92},
        ],
    )
    assert instance.claims[0].confidence == pytest.approx(0.92)


def test_claims_confidence_already_decimal_is_unchanged():
    PaperModel = _multi_experiment_model(extract_claims=True)
    instance = PaperModel(
        feedstock_type="waste_cooking_oil",
        experiments=[{"active_metal": "NiMo", "evidence": {}, "experiment_confidence": 0.9}],
        claims=[
            {"subject": "Ni loading", "relation": "increases", "object": "conversion",
             "evidence_text": "text", "confidence": 0.85},
        ],
    )
    assert instance.claims[0].confidence == pytest.approx(0.85)


def test_no_claims_field_does_not_crash_extract_claims_false_domain():
    """A domain with extract_claims=False has no `claims` key at all --
    the rescaler must not assume it's present."""
    PaperModel = _multi_experiment_model(extract_claims=False)
    instance = PaperModel(
        feedstock_type="waste_cooking_oil",
        experiments=[{"active_metal": "NiMo", "evidence": {}, "experiment_confidence": 0.9}],
    )
    assert not hasattr(instance, "claims") or instance.__dict__.get("claims") in (None, [])


def test_empty_claims_list_does_not_crash():
    PaperModel = _multi_experiment_model(extract_claims=True)
    instance = PaperModel(
        feedstock_type="waste_cooking_oil",
        experiments=[{"active_metal": "NiMo", "evidence": {}, "experiment_confidence": 0.9}],
        claims=[],
    )
    assert instance.claims == []


def test_saf_hefa_domain_claims_confidence_rescale_end_to_end():
    """Same bug, exercised through the actual shipped saf_hefa domain
    rather than a synthetic minimal one, since that's the domain the
    production failure happened in."""
    from llm_extract.domain_registry import get_schema

    domains_dir = Path(__file__).parent.parent / "domains"
    schema = get_schema("saf_hefa", domains_dir)
    PaperModel = schema.ExtractionFields

    instance = PaperModel(
        feedstock_type="waste_cooking_oil",
        experiments=[{
            "active_metal": "NiMo", "reaction_temperature_C": 350.0,
            "evidence": {}, "experiment_confidence": 0.9,
        }],
        claims=[
            {"subject": "Ni loading", "relation": "increases", "object": "conversion",
             "evidence_text": "Higher Ni loading increased conversion.", "confidence": 3},
        ],
    )
    assert 0.0 <= instance.claims[0].confidence <= 1.0
