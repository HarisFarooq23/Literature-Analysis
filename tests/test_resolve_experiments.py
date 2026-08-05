from pydantic import BaseModel
from typing import Optional

from llm_extract.resolve import match_experiments, resolve_experiment_lists


class FakeExperiment(BaseModel):
    active_metal: Optional[str] = None
    support_material: Optional[str] = None
    conversion_pct: Optional[float] = None
    experiment_confidence: float = 0.9
    evidence: dict = {}


def test_match_experiments_identity_key_unambiguous():
    primary = [
        FakeExperiment(active_metal="Ni", support_material="Al2O3", conversion_pct=78),
        FakeExperiment(active_metal="NiMo", support_material="SAPO-11", conversion_pct=95),
    ]
    cross = [
        FakeExperiment(active_metal="NiMo", support_material="SAPO-11", conversion_pct=94),  # same identity, diff conversion
        FakeExperiment(active_metal="Ni", support_material="Al2O3", conversion_pct=79),
    ]
    matches = match_experiments(primary, cross)
    assert len(matches) == 2
    for m in matches:
        assert m.match_method == "identity_key"
        assert m.primary.active_metal == m.cross_check.active_metal
        assert m.primary.support_material == m.cross_check.support_material


def test_match_experiments_disambiguates_duplicate_identity_by_numeric_similarity():
    """Same catalyst tested at two conditions -- identity keys collide since
    the differentiating value (conversion_pct here) is numeric and
    deliberately excluded from the identity key. This USED to fall back to
    raw list-position pairing (which happened to be correct in this
    specific test only because both lists were coincidentally in the same
    order -- see test_resolve_loading_collision.py for the reordered case
    that exposed this as a real bug). It now disambiguates by numeric
    similarity across shared numeric fields instead, which is correct
    regardless of list order and is what match_method="numeric_similarity"
    signals downstream (e.g. in normalize.py's output) as a weaker-than-
    identity_key but still principled match, distinct from a last-resort
    'position' pairing."""
    primary = [
        FakeExperiment(active_metal="Ni", support_material="Al2O3", conversion_pct=60),
        FakeExperiment(active_metal="Ni", support_material="Al2O3", conversion_pct=85),
    ]
    cross = [
        FakeExperiment(active_metal="Ni", support_material="Al2O3", conversion_pct=61),
        FakeExperiment(active_metal="Ni", support_material="Al2O3", conversion_pct=84),
    ]
    matches = match_experiments(primary, cross)
    assert len(matches) == 2
    assert all(m.match_method == "numeric_similarity" for m in matches)
    # Correctly paired by closeness (60<->61, 85<->84) regardless of which
    # order the greedy algorithm resolved them in -- do not assume list
    # position of the result, only that each pair is the numerically
    # closest available partner.
    paired_values = sorted(
        (m.primary.conversion_pct, m.cross_check.conversion_pct) for m in matches
    )
    assert paired_values == [(60, 61), (85, 84)]


def test_match_experiments_unequal_counts_flags_extras():
    primary = [
        FakeExperiment(active_metal="Ni", support_material="Al2O3", conversion_pct=78),
        FakeExperiment(active_metal="NiMo", support_material="SAPO-11", conversion_pct=95),
        FakeExperiment(active_metal="Pt", support_material="Al2O3", conversion_pct=88),
    ]
    cross = [
        FakeExperiment(active_metal="Ni", support_material="Al2O3", conversion_pct=79),
    ]
    matches = match_experiments(primary, cross)
    methods = sorted(m.match_method for m in matches)
    assert methods.count("primary_only") == 2
    assert methods.count("identity_key") == 1


def test_match_experiments_empty_lists():
    assert match_experiments([], []) == []
    matches = match_experiments([FakeExperiment(active_metal="Ni")], [])
    assert len(matches) == 1
    assert matches[0].match_method == "primary_only"


def test_resolve_experiment_lists_full_agreement():
    primary = [FakeExperiment(active_metal="Ni", support_material="Al2O3", conversion_pct=78.0)]
    cross = [FakeExperiment(active_metal="Ni", support_material="Al2O3", conversion_pct=78.2)]  # within tolerance
    results = resolve_experiment_lists(primary, cross, None, FakeExperiment, numeric_tolerance=0.10)
    assert len(results) == 1
    assert results[0].experiment_match_status == "full_agreement"
    assert results[0].resolved["active_metal"] == "Ni"


def test_resolve_experiment_lists_disagreement_resolved_by_tie_breaker():
    primary = [FakeExperiment(active_metal="Ni", support_material="Al2O3", conversion_pct=60.0)]
    cross = [FakeExperiment(active_metal="Pt", support_material="Al2O3", conversion_pct=60.0)]
    tie_breaker = [FakeExperiment(active_metal="Ni", support_material="Al2O3", conversion_pct=60.0)]
    results = resolve_experiment_lists(primary, cross, tie_breaker, FakeExperiment, numeric_tolerance=0.10)
    assert len(results) == 1
    assert results[0].experiment_match_status == "resolved_after_escalation"
    assert results[0].resolved["active_metal"] == "Ni"  # tie_breaker + primary majority


def test_resolve_experiment_lists_three_catalysts_all_preserved():
    """The headline scenario from the critique: a paper testing 3 catalysts
    must produce 3 resolved experiment records, not 1."""
    primary = [
        FakeExperiment(active_metal="Ni", support_material="Al2O3", conversion_pct=78.0),
        FakeExperiment(active_metal="NiMo", support_material="SAPO-11", conversion_pct=95.0),
        FakeExperiment(active_metal="Pt", support_material="Al2O3", conversion_pct=88.0),
    ]
    cross = [
        FakeExperiment(active_metal="Ni", support_material="Al2O3", conversion_pct=77.5),
        FakeExperiment(active_metal="NiMo", support_material="SAPO-11", conversion_pct=94.8),
        FakeExperiment(active_metal="Pt", support_material="Al2O3", conversion_pct=87.5),
    ]
    results = resolve_experiment_lists(primary, cross, None, FakeExperiment, numeric_tolerance=0.10)
    assert len(results) == 3
    metals = sorted(r.resolved["active_metal"] for r in results)
    assert metals == ["Ni", "NiMo", "Pt"]
    assert all(r.experiment_match_status == "full_agreement" for r in results)


def test_resolve_experiment_lists_single_model_only_is_kept_and_flagged():
    primary = [
        FakeExperiment(active_metal="Ni", support_material="Al2O3", conversion_pct=78.0),
        FakeExperiment(active_metal="NiMo", support_material="SAPO-11", conversion_pct=95.0),  # cross missed this one
    ]
    cross = [
        FakeExperiment(active_metal="Ni", support_material="Al2O3", conversion_pct=78.0),
    ]
    results = resolve_experiment_lists(primary, cross, None, FakeExperiment, numeric_tolerance=0.10)
    assert len(results) == 2
    statuses = sorted(r.experiment_match_status for r in results)
    assert statuses == ["full_agreement", "single_model_only"]
    single_only = next(r for r in results if r.experiment_match_status == "single_model_only")
    assert single_only.resolved["active_metal"] == "NiMo"  # data preserved, not discarded
