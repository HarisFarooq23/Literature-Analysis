"""
test_benchmark_score.py — tests for llm_extract.benchmark.score, the
gold-standard validation tool that produces the F1 table a methods section
needs. Every metric here is checked against a hand-calculated expected
value, not just "does it run without crashing" -- an F1 number that's
silently wrong is worse than no F1 number at all in a published paper.
"""

import json
from pathlib import Path

import pytest

from llm_extract.benchmark.annotation_template import (
    _blank_experiment_dict,
    generate_annotation_template,
    PaperStub,
)
from llm_extract.benchmark.score import score
from llm_extract.domain_registry import get_schema

DOMAINS_DIR = Path(__file__).parent.parent / "domains"


@pytest.fixture(scope="module")
def schema():
    return get_schema("saf_hefa", DOMAINS_DIR)


# --- Annotation template ----------------------------------------------------

def test_blank_experiment_dict_has_all_experiment_scope_fields_null(schema):
    blank = _blank_experiment_dict(schema)
    for f in schema.spec.experiment_fields():
        assert blank[f.name] is None
    assert blank["evidence"] == {"source_location": None, "evidence_text": None}
    # Internal LLM-only metadata should NOT appear in the annotator-facing template
    assert "experiment_confidence" not in blank


def test_generate_annotation_template_shape(schema):
    stubs = [
        PaperStub(paper_id="p1", doi="10.1/a", title="T1", first_author="A", journal="J", publication_year=2024),
        PaperStub(paper_id="p2", doi="10.1/b", title="T2", first_author="B", journal="J", publication_year=2023),
    ]
    templates = generate_annotation_template(schema, stubs, n_experiment_slots=2)
    assert len(templates) == 2
    assert templates[0]["paper_id"] == "p1"
    assert len(templates[0]["experiments"]) == 2
    assert templates[0]["gold_annotator"] is None
    assert "claims" in templates[0]  # saf_hefa has extract_claims=True


# --- Scoring: build small synthetic gold vs predicted JSONL files -----------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _exp(active_metal=None, support_material=None, oil_conversion_pct=None,
         reaction_temperature_C=None, **kwargs) -> dict:
    d = {
        "active_metal": active_metal, "support_material": support_material,
        "oil_conversion_pct": oil_conversion_pct,
        "reaction_temperature_C": reaction_temperature_C,
        "evidence": {"source_location": None, "evidence_text": None},
    }
    d.update(kwargs)
    return d


def test_score_perfect_agreement_gives_f1_of_one(tmp_path):
    """Gold and predicted are IDENTICAL -- every categorical field should
    score precision=recall=f1=1.0, every numeric field should have MAE=0
    and 100% within tolerance."""
    gold = [{
        "paper_id": "p1", "fields": {"feedstock_type": "waste_cooking_oil"},
        "experiments": [_exp(active_metal="Ni", support_material="Al2O3",
                              oil_conversion_pct=80.0, reaction_temperature_C=300.0)],
    }]
    predicted = [{
        "paper_id": "p1", "fields": {"feedstock_type": "waste_cooking_oil"},
        "experiments": [_exp(active_metal="Ni", support_material="Al2O3",
                              oil_conversion_pct=80.0, reaction_temperature_C=300.0)],
    }]
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "predicted.jsonl"
    _write_jsonl(gold_path, gold)
    _write_jsonl(pred_path, predicted)

    result = score(pred_path, gold_path, "saf_hefa", DOMAINS_DIR)
    assert result.n_matched_papers == 1
    assert result.experiment_count_exact_match_fraction == 1.0

    metal_score = result.field_scores["active_metal"]
    assert metal_score.precision == 1.0
    assert metal_score.recall == 1.0
    assert metal_score.f1 == 1.0

    conv_score = result.field_scores["oil_conversion_pct"]
    assert conv_score.mae == 0.0
    assert conv_score.pct_within_tolerance == 1.0


def test_score_categorical_false_positive_and_false_negative(tmp_path):
    """Gold says one support, predicted says a different one on one
    experiment (wrong value = FP+FN); gold has a support value, predicted
    has none on another (missing = FN only). support_material is used here
    (rather than active_metal) because active_metal is required:true in
    the schema and therefore cannot be None even in principle -- see
    test_required_field_cannot_be_null_even_for_scoring below for why that
    matters and is tracked as a separate, pre-existing consideration."""
    gold = [{
        "paper_id": "p1", "fields": {},
        "experiments": [
            _exp(active_metal="Ni", support_material="Al2O3", oil_conversion_pct=78.0, reaction_temperature_C=300.0),
            _exp(active_metal="NiMo", support_material="SAPO-11", oil_conversion_pct=95.0, reaction_temperature_C=310.0),
            _exp(active_metal="Pt", support_material="gamma-Al2O3", oil_conversion_pct=88.0, reaction_temperature_C=320.0),
        ],
    }]
    predicted = [{
        "paper_id": "p1", "fields": {},
        "experiments": [
            _exp(active_metal="Ni", support_material="Al2O3", oil_conversion_pct=78.0, reaction_temperature_C=300.0),   # exact match
            _exp(active_metal="NiMo", support_material=None, oil_conversion_pct=95.0, reaction_temperature_C=310.0),   # FN on support_material
            _exp(active_metal="Pt", support_material="SiO2", oil_conversion_pct=88.0, reaction_temperature_C=320.0),   # wrong value (FP+FN)
        ],
    }]
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "predicted.jsonl"
    _write_jsonl(gold_path, gold)
    _write_jsonl(pred_path, predicted)

    result = score(pred_path, gold_path, "saf_hefa", DOMAINS_DIR)
    assert result.n_experiments_gold == 3
    assert result.n_experiments_predicted == 3

    support_score = result.field_scores["support_material"]
    # TP=1 (Al2O3), FN=2 (SAPO-11 missing, gamma-Al2O3 wrong->also FN), FP=1 (SiO2 wrong->also FP)
    assert support_score.true_positives == 1
    assert support_score.false_negatives == 2
    assert support_score.false_positives == 1
    expected_precision = 1 / (1 + 1)  # TP / (TP+FP) = 0.5
    expected_recall = 1 / (1 + 2)     # TP / (TP+FN) = 0.333...
    assert support_score.precision == pytest.approx(expected_precision)
    assert support_score.recall == pytest.approx(expected_recall)

    # active_metal (required:true) should have scored perfectly here since
    # every experiment has a real value on both sides in this test's design
    metal_score = result.field_scores["active_metal"]
    assert metal_score.true_positives == 3
    assert metal_score.false_positives == 0
    assert metal_score.false_negatives == 0


def test_required_field_cannot_be_null_even_for_scoring():
    """Documents a real, pre-existing schema behaviour this benchmark tool
    surfaces rather than causes: fields marked required:true in a domain
    YAML compile to a NON-Optional Pydantic type (see
    domain_spec._fields_to_pydantic_kwargs), so Pydantic rejects None for
    them unconditionally -- not just in gold/predicted comparison, but
    also for the LLM's OWN structured-output responses in production. If a
    paper genuinely doesn't state a required field for one experiment
    (e.g. oil_conversion_pct truly isn't reported for a side-experiment),
    the model returning null for it would fail Pydantic validation and
    exhaust its retries just like a malformed response would, potentially
    losing an otherwise-extractable paper over one field. This is worth
    knowing when choosing which fields to mark required:true -- reserve it
    for fields that are essentially always reported (as active_metal,
    reaction_temperature_C, oil_conversion_pct, and feedstock_type
    currently are for saf_hefa), not for anything that could plausibly be
    absent from a given experiment."""
    from llm_extract.domain_registry import get_schema
    schema = get_schema("saf_hefa", DOMAINS_DIR)
    ExperimentModel = schema.ExperimentFields
    with pytest.raises(Exception):  # pydantic.ValidationError
        ExperimentModel(active_metal=None, reaction_temperature_C=300.0,
                         oil_conversion_pct=80.0, experiment_confidence=0.9)


def test_score_numeric_within_tolerance_and_outside_tolerance(tmp_path):
    """One experiment within 10% tolerance, one clearly outside it --
    confirms pct_within_tolerance and MAE are computed correctly, not just
    present."""
    gold = [{
        "paper_id": "p1", "fields": {},
        "experiments": [
            _exp(active_metal="Ni", oil_conversion_pct=100.0, reaction_temperature_C=300.0),  # pred: 105 -> within 10%
            _exp(active_metal="Pt", oil_conversion_pct=50.0, reaction_temperature_C=300.0),   # pred: 80 -> way outside
        ],
    }]
    predicted = [{
        "paper_id": "p1", "fields": {},
        "experiments": [
            _exp(active_metal="Ni", oil_conversion_pct=105.0, reaction_temperature_C=300.0),
            _exp(active_metal="Pt", oil_conversion_pct=80.0, reaction_temperature_C=300.0),
        ],
    }]
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "predicted.jsonl"
    _write_jsonl(gold_path, gold)
    _write_jsonl(pred_path, predicted)

    result = score(pred_path, gold_path, "saf_hefa", DOMAINS_DIR, numeric_tolerance=0.10)
    conv_score = result.field_scores["oil_conversion_pct"]
    assert conv_score.n_numeric_comparable == 2
    assert conv_score.n_within_tolerance == 1  # only the 100 vs 105 pair
    assert conv_score.pct_within_tolerance == 0.5
    expected_mae = (abs(100 - 105) + abs(50 - 80)) / 2  # (5 + 30) / 2 = 17.5
    assert conv_score.mae == pytest.approx(expected_mae)


def test_score_missing_experiment_entirely_counted_as_unmatched_gold(tmp_path):
    """Pipeline finds only 1 of 2 gold experiments -- the missed one must
    show up as unmatched_gold_experiments, and the experiment-count exact
    match fraction must reflect this paper as a MISS, not silently
    averaged away."""
    gold = [{
        "paper_id": "p1", "fields": {},
        "experiments": [
            _exp(active_metal="Ni", oil_conversion_pct=78.0, reaction_temperature_C=300.0),
            _exp(active_metal="NiMo", oil_conversion_pct=95.0, reaction_temperature_C=310.0),
        ],
    }]
    predicted = [{
        "paper_id": "p1", "fields": {},
        "experiments": [
            _exp(active_metal="Ni", oil_conversion_pct=78.0, reaction_temperature_C=300.0),
        ],
    }]
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "predicted.jsonl"
    _write_jsonl(gold_path, gold)
    _write_jsonl(pred_path, predicted)

    result = score(pred_path, gold_path, "saf_hefa", DOMAINS_DIR)
    assert result.unmatched_gold_experiments == 1
    assert result.experiment_count_exact_match_fraction == 0.0  # 2 gold vs 1 predicted -> miss


def test_score_blank_gold_experiment_rows_excluded(tmp_path):
    """A gold file built from the annotation template with unused blank
    slots (n_experiment_slots=3 but the paper only had 1 real experiment)
    must not count the 2 blank rows as gold experiments to be matched
    against -- they were never filled in."""
    gold = [{
        "paper_id": "p1", "fields": {},
        "experiments": [
            _exp(active_metal="Ni", oil_conversion_pct=78.0, reaction_temperature_C=300.0),
            _exp(),  # blank slot, never filled in by annotator
            _exp(),  # blank slot, never filled in by annotator
        ],
    }]
    predicted = [{
        "paper_id": "p1", "fields": {},
        "experiments": [
            _exp(active_metal="Ni", oil_conversion_pct=78.0, reaction_temperature_C=300.0),
        ],
    }]
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "predicted.jsonl"
    _write_jsonl(gold_path, gold)
    _write_jsonl(pred_path, predicted)

    result = score(pred_path, gold_path, "saf_hefa", DOMAINS_DIR)
    assert result.n_experiments_gold == 1  # blank rows correctly excluded
    assert result.experiment_count_exact_match_fraction == 1.0


def test_score_only_scores_papers_present_in_both_files(tmp_path):
    """A gold paper the pipeline hasn't processed, and a predicted paper
    with no gold annotation, must both be silently excluded -- not counted
    as failures."""
    gold = [
        {"paper_id": "p1", "fields": {}, "experiments": [_exp(active_metal="Ni", oil_conversion_pct=78.0, reaction_temperature_C=300.0)]},
        {"paper_id": "p_gold_only", "fields": {}, "experiments": [_exp(active_metal="Pt", oil_conversion_pct=88.0, reaction_temperature_C=300.0)]},
    ]
    predicted = [
        {"paper_id": "p1", "fields": {}, "experiments": [_exp(active_metal="Ni", oil_conversion_pct=78.0, reaction_temperature_C=300.0)]},
        {"paper_id": "p_predicted_only", "fields": {}, "experiments": [_exp(active_metal="Pd", oil_conversion_pct=50.0, reaction_temperature_C=300.0)]},
    ]
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "predicted.jsonl"
    _write_jsonl(gold_path, gold)
    _write_jsonl(pred_path, predicted)

    result = score(pred_path, gold_path, "saf_hefa", DOMAINS_DIR)
    assert result.n_matched_papers == 1  # only p1 is in both
    assert result.n_experiments_gold == 1
    assert result.n_experiments_predicted == 1


def test_score_raises_on_no_overlapping_papers(tmp_path):
    gold = [{"paper_id": "only_gold", "fields": {}, "experiments": [_exp(active_metal="Ni", oil_conversion_pct=78.0, reaction_temperature_C=300.0)]}]
    predicted = [{"paper_id": "only_predicted", "fields": {}, "experiments": [_exp(active_metal="Pt", oil_conversion_pct=88.0, reaction_temperature_C=300.0)]}]
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "predicted.jsonl"
    _write_jsonl(gold_path, gold)
    _write_jsonl(pred_path, predicted)

    with pytest.raises(ValueError, match="nothing to score"):
        score(pred_path, gold_path, "saf_hefa", DOMAINS_DIR)


def test_benchmark_result_to_dataframe_and_summary_do_not_crash(tmp_path):
    """Smoke test for the reporting helpers actually used to produce a
    paper-ready table."""
    gold = [{
        "paper_id": "p1", "fields": {},
        "experiments": [_exp(active_metal="Ni", support_material="Al2O3",
                              oil_conversion_pct=78.0, reaction_temperature_C=300.0)],
    }]
    predicted = [{
        "paper_id": "p1", "fields": {},
        "experiments": [_exp(active_metal="Ni", support_material="Al2O3",
                              oil_conversion_pct=79.0, reaction_temperature_C=300.0)],
    }]
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "predicted.jsonl"
    _write_jsonl(gold_path, gold)
    _write_jsonl(pred_path, predicted)

    result = score(pred_path, gold_path, "saf_hefa", DOMAINS_DIR)
    df = result.to_dataframe()
    assert len(df) == len(result.field_scores)
    assert "field" in df.columns
    text_summary = result.summary()
    assert "active_metal" in text_summary
