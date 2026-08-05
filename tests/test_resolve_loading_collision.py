"""
test_resolve_loading_collision.py — regression tests for a real bug found
during code review: two experiments sharing the same catalyst identity
(metal + support) but differing only in a numeric field the identity key
deliberately excludes (loading wt%, temperature, ...) produce IDENTICAL
identity keys. Before the fix in this file's target, that collision fell
through to raw list-position pairing, which silently cross-pairs the wrong
variants whenever the two models happen to list them in a different order
-- e.g. 10 wt% NiMo/Al2O3 gets compared against 5 wt% NiMo/Al2O3's numbers,
producing a spurious "conversion disagrees" flag for BOTH rows even though
each model individually extracted its row correctly.

The fix (_match_within_collision_group / _numeric_distance in resolve.py)
disambiguates same-identity-key collisions by nearest numeric similarity
across whatever numeric fields the experiment model happens to define, so
it generalizes to any domain (loading, temperature, pressure, ...) without
hardcoding a field name.
"""

from typing import Optional

from pydantic import BaseModel

from llm_extract.resolve import match_experiments, resolve_experiment_lists


class FakeExperiment(BaseModel):
    active_metal: Optional[str] = None
    support_material: Optional[str] = None
    metal_loading_wt_pct: Optional[float] = None
    oil_conversion_pct: Optional[float] = None
    experiment_confidence: float = 0.9
    evidence: dict = {}


def test_same_catalyst_different_loading_reordered_is_not_cross_paired():
    """The exact bug: same metal+support at two loadings, listed in
    OPPOSITE order by the two models. Must pair by loading similarity,
    not position."""
    primary = [
        FakeExperiment(active_metal="NiMo", support_material="Al2O3",
                        metal_loading_wt_pct=10.0, oil_conversion_pct=95.0),
        FakeExperiment(active_metal="NiMo", support_material="Al2O3",
                        metal_loading_wt_pct=5.0, oil_conversion_pct=65.0),
    ]
    cross = [
        FakeExperiment(active_metal="NiMo", support_material="Al2O3",
                        metal_loading_wt_pct=5.0, oil_conversion_pct=64.0),
        FakeExperiment(active_metal="NiMo", support_material="Al2O3",
                        metal_loading_wt_pct=10.0, oil_conversion_pct=94.0),
    ]
    matches = match_experiments(primary, cross)
    assert len(matches) == 2
    assert all(m.match_method == "numeric_similarity" for m in matches)
    for m in matches:
        assert m.primary.metal_loading_wt_pct == m.cross_check.metal_loading_wt_pct, (
            f"Cross-paired wrong loading variants: primary={m.primary.metal_loading_wt_pct} "
            f"vs cross={m.cross_check.metal_loading_wt_pct}"
        )


def test_same_catalyst_three_loadings_all_correctly_disambiguated():
    """Extends to 3 loading levels (a realistic loading-series study) to
    confirm the greedy nearest-neighbour matching scales beyond the
    trivial 2-item case."""
    primary = [
        FakeExperiment(active_metal="Ni", support_material="Al2O3",
                        metal_loading_wt_pct=15.0, oil_conversion_pct=97.0),
        FakeExperiment(active_metal="Ni", support_material="Al2O3",
                        metal_loading_wt_pct=5.0, oil_conversion_pct=60.0),
        FakeExperiment(active_metal="Ni", support_material="Al2O3",
                        metal_loading_wt_pct=10.0, oil_conversion_pct=82.0),
    ]
    cross = [
        FakeExperiment(active_metal="Ni", support_material="Al2O3",
                        metal_loading_wt_pct=10.0, oil_conversion_pct=81.0),
        FakeExperiment(active_metal="Ni", support_material="Al2O3",
                        metal_loading_wt_pct=15.0, oil_conversion_pct=96.5),
        FakeExperiment(active_metal="Ni", support_material="Al2O3",
                        metal_loading_wt_pct=5.0, oil_conversion_pct=59.5),
    ]
    matches = match_experiments(primary, cross)
    assert len(matches) == 3
    assert all(m.match_method == "numeric_similarity" for m in matches)
    loadings = sorted(m.primary.metal_loading_wt_pct for m in matches)
    assert loadings == [5.0, 10.0, 15.0]
    for m in matches:
        assert m.primary.metal_loading_wt_pct == m.cross_check.metal_loading_wt_pct


def test_full_resolution_end_to_end_gives_full_agreement_not_false_conflict():
    """Confirms the fix actually prevents the downstream symptom (spurious
    disagreement on every field) reported in the original bug -- not just
    that matching looks right in isolation."""
    primary = [
        FakeExperiment(active_metal="NiMo", support_material="Al2O3",
                        metal_loading_wt_pct=10.0, oil_conversion_pct=95.0),
        FakeExperiment(active_metal="NiMo", support_material="Al2O3",
                        metal_loading_wt_pct=5.0, oil_conversion_pct=65.0),
    ]
    cross = [
        FakeExperiment(active_metal="NiMo", support_material="Al2O3",
                        metal_loading_wt_pct=5.0, oil_conversion_pct=64.5),
        FakeExperiment(active_metal="NiMo", support_material="Al2O3",
                        metal_loading_wt_pct=10.0, oil_conversion_pct=94.5),
    ]
    results = resolve_experiment_lists(primary, cross, None, FakeExperiment, numeric_tolerance=0.10)
    assert len(results) == 2
    assert all(r.experiment_match_status == "full_agreement" for r in results), (
        "Correct pairing should show full agreement (values are within "
        "tolerance for the SAME loading variant); a mispairing would show "
        "false conflict on oil_conversion_pct instead."
    )


def test_asymmetric_collision_more_primary_than_cross():
    """Primary reports 3 loading variants, cross_check only reports 2 of
    them (missed one row) -- the 2 that ARE present on both sides must
    still be correctly matched by loading, and the extra primary-side
    experiment must be flagged primary_only, not silently mispaired."""
    primary = [
        FakeExperiment(active_metal="Pt", support_material="Al2O3",
                        metal_loading_wt_pct=1.0, oil_conversion_pct=70.0),
        FakeExperiment(active_metal="Pt", support_material="Al2O3",
                        metal_loading_wt_pct=2.0, oil_conversion_pct=85.0),
        FakeExperiment(active_metal="Pt", support_material="Al2O3",
                        metal_loading_wt_pct=3.0, oil_conversion_pct=90.0),
    ]
    cross = [
        FakeExperiment(active_metal="Pt", support_material="Al2O3",
                        metal_loading_wt_pct=3.0, oil_conversion_pct=89.5),
        FakeExperiment(active_metal="Pt", support_material="Al2O3",
                        metal_loading_wt_pct=1.0, oil_conversion_pct=70.5),
    ]
    matches = match_experiments(primary, cross)
    matched_pairs = [m for m in matches if m.primary is not None and m.cross_check is not None]
    singles = [m for m in matches if m.match_method == "primary_only"]

    assert len(matched_pairs) == 2
    assert len(singles) == 1
    assert singles[0].primary.metal_loading_wt_pct == 2.0
    for m in matched_pairs:
        assert m.primary.metal_loading_wt_pct == m.cross_check.metal_loading_wt_pct


def test_greedy_suboptimal_case_is_solved_exactly():
    """The specific scenario where a naive greedy nearest-neighbour matcher
    gives a strictly worse total assignment than the optimum: primary
    loadings [10, 12] vs cross_check [11, 9]. Greedy pairs (10,11) first
    (smallest single pairwise distance), forcing (12,9) -- total distance
    worse than the true optimum (10,9)+(12,11). The Hungarian-algorithm
    solver must find the globally optimal pairing, not the greedy one.
    """
    primary = [
        FakeExperiment(active_metal="Ni", support_material="Al2O3",
                        metal_loading_wt_pct=10.0, oil_conversion_pct=50.0),
        FakeExperiment(active_metal="Ni", support_material="Al2O3",
                        metal_loading_wt_pct=12.0, oil_conversion_pct=60.0),
    ]
    cross = [
        FakeExperiment(active_metal="Ni", support_material="Al2O3",
                        metal_loading_wt_pct=11.0, oil_conversion_pct=55.0),
        FakeExperiment(active_metal="Ni", support_material="Al2O3",
                        metal_loading_wt_pct=9.0, oil_conversion_pct=45.0),
    ]
    matches = match_experiments(primary, cross)
    assert len(matches) == 2
    pairing = {m.primary.metal_loading_wt_pct: m.cross_check.metal_loading_wt_pct for m in matches}
    # Optimal: 10<->9, 12<->11 (sum |diff| = 1+1 = 2)
    # Greedy would give: 10<->11, 12<->9 (sum |diff| = 1+3 = 4)
    assert pairing == {10.0: 9.0, 12.0: 11.0}


def test_no_numeric_fields_available_degrades_to_position_not_crash():
    """If an experiment model somehow has NO numeric fields at all (edge
    case, unlikely in practice), the collision-group matcher must not
    crash -- it degrades gracefully to original relative order, matching
    pre-fix behaviour rather than raising."""
    class NoNumericExperiment(BaseModel):
        active_metal: Optional[str] = None
        support_material: Optional[str] = None
        experiment_confidence: float = 0.9
        evidence: dict = {}

    primary = [
        NoNumericExperiment(active_metal="Ni", support_material="Al2O3"),
        NoNumericExperiment(active_metal="Ni", support_material="Al2O3"),
    ]
    cross = [
        NoNumericExperiment(active_metal="Ni", support_material="Al2O3"),
        NoNumericExperiment(active_metal="Ni", support_material="Al2O3"),
    ]
    matches = match_experiments(primary, cross)
    assert len(matches) == 2
    # No crash is the main assertion; pairing is arbitrary-but-stable here
    # since there is genuinely nothing to disambiguate with.
