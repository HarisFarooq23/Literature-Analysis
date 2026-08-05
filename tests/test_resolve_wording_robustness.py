"""
test_resolve_wording_robustness.py — regression tests for the specific
failure mode diagnosed in support: "surface area disagree" (and every other
numeric field) firing on nearly every experiment in a multi-catalyst paper.

Root cause: primary and cross_check describe the SAME catalyst using
different wording ("NiMo" vs "Ni-Mo", "HZSM-5" vs "ZSM-5"), the raw-text
identity key doesn't match, matching falls through to position-pairing, and
if the two models happened to list catalysts in a different order, position-
pairing then compares two UNRELATED experiments -- producing disagreement on
every single field. canonicalization-aware identity keys (see
resolve._experiment_identity_key) are the fix; these tests lock in that the
fix actually prevents the failure mode end-to-end, not just that the
canonicalize.py functions work in isolation (already covered by
test_canonicalize.py).
"""

from typing import Optional

from pydantic import BaseModel

from llm_extract.resolve import match_experiments, resolve_experiment_lists


class FakeExperiment(BaseModel):
    active_metal: Optional[str] = None
    support_material: Optional[str] = None
    bet_surface_area_m2g: Optional[float] = None
    reaction_temperature_C: Optional[float] = None
    oil_conversion_pct: Optional[float] = None
    experiment_confidence: float = 0.9
    evidence: dict = {}


def test_wording_variance_alone_does_not_cause_position_fallback():
    """The core regression: same catalysts, same order, but described with
    different (equally valid) wording by each model. Must match by
    canonicalized identity, not fall through to position pairing (which
    would happen to give the right answer here anyway, by luck of order --
    the NEXT test proves why relying on that luck is the actual bug)."""
    primary = [
        FakeExperiment(active_metal="NiMo", support_material="HZSM-5",
                        bet_surface_area_m2g=215.0, oil_conversion_pct=94.0),
        FakeExperiment(active_metal="Pt", support_material="gamma-Al2O3",
                        bet_surface_area_m2g=180.0, oil_conversion_pct=88.0),
    ]
    cross = [
        FakeExperiment(active_metal="Ni-Mo", support_material="ZSM-5",
                        bet_surface_area_m2g=213.0, oil_conversion_pct=93.5),
        FakeExperiment(active_metal="Pt", support_material="gamma-Al2O3",
                        bet_surface_area_m2g=181.0, oil_conversion_pct=87.5),
    ]
    matches = match_experiments(primary, cross)
    assert len(matches) == 2
    assert all(m.match_method == "identity_key" for m in matches), (
        "Wording variants of the SAME catalyst must match by canonicalized "
        "identity key, not fall through to position pairing."
    )


def test_wording_variance_with_reordering_would_break_naive_position_matching():
    """The actual failure mode from support: same two catalysts, but each
    model lists them in a DIFFERENT order (both correctly reading the
    paper's table, e.g. sorted differently by activity vs. as-listed). A
    naive position-based pairing scheme would pair NiMo (primary, position
    0) with Pt (cross_check, position 0) -- two completely different
    catalysts -- and every single numeric field would show 'disagreement'
    even though both models extracted their respective rows perfectly.
    canonicalized identity-key matching must catch this and pair correctly
    regardless of list order.
    """
    primary = [
        FakeExperiment(active_metal="NiMo", support_material="HZSM-5",
                        bet_surface_area_m2g=215.0, reaction_temperature_C=320.0,
                        oil_conversion_pct=94.0),
        FakeExperiment(active_metal="Pt", support_material="gamma-Al2O3",
                        bet_surface_area_m2g=180.0, reaction_temperature_C=300.0,
                        oil_conversion_pct=88.0),
    ]
    # cross_check lists the SAME two catalysts in the OPPOSITE order, using
    # different (but equivalent) wording for the first one.
    cross = [
        FakeExperiment(active_metal="Pt", support_material="gamma-Al2O3",
                        bet_surface_area_m2g=181.0, reaction_temperature_C=301.0,
                        oil_conversion_pct=87.5),
        FakeExperiment(active_metal="Ni/Mo", support_material="ZSM5",
                        bet_surface_area_m2g=213.0, reaction_temperature_C=319.0,
                        oil_conversion_pct=93.5),
    ]

    matches = match_experiments(primary, cross)
    assert len(matches) == 2
    assert all(m.match_method == "identity_key" for m in matches), (
        "Reordered experiments with wording variance must still match by "
        "identity key -- if this falls back to position pairing instead, "
        "NiMo/HZSM-5 gets compared against Pt/gamma-Al2O3, which is the "
        "exact 'everything disagrees' bug being regression-tested here."
    )

    # Confirm the resolution is actually correct: NiMo paired with NiMo,
    # Pt paired with Pt, both showing full (or near-full) agreement, NOT
    # spurious disagreement on every field.
    results = resolve_experiment_lists(primary, cross, None, FakeExperiment, numeric_tolerance=0.10)
    assert len(results) == 2
    metals = sorted(r.resolved["active_metal"].replace("/", "").replace("-", "") for r in results)
    assert metals == ["NiMo", "Pt"]
    # Both should be full_agreement (values are within 10% tolerance of
    # each other for the matched catalyst, not compared cross-catalyst)
    assert all(r.experiment_match_status == "full_agreement" for r in results)


def test_naive_raw_text_matching_would_have_failed_this_case():
    """Sanity check that the SCENARIO in the previous test is genuinely
    hard for raw-text matching -- i.e. that this test suite is actually
    exercising the canonicalization fix, not a scenario that would have
    passed anyway. Confirms the raw (non-canonicalized) identity strings
    for the swapped-order NiMo variant really do differ before
    canonicalization is applied."""
    from llm_extract.canonicalize import canonicalize_metal

    raw_primary_metal = "NiMo"
    raw_cross_metal = "Ni/Mo"
    assert raw_primary_metal.strip().lower() != raw_cross_metal.strip().lower(), (
        "Test setup error: these raw strings must differ pre-canonicalization "
        "for this regression test to be meaningful."
    )
    # But after canonicalization, they must be identical -- this is what
    # resolve._experiment_identity_key relies on.
    assert canonicalize_metal(raw_primary_metal)[1] == canonicalize_metal(raw_cross_metal)[1]


def test_genuinely_different_catalysts_are_never_falsely_merged():
    """Guard against over-eager canonicalization: two experiments that
    really ARE different catalysts (not just wording variants of the same
    one) must never be matched by identity key."""
    primary = [FakeExperiment(active_metal="Ni", support_material="Al2O3", oil_conversion_pct=78.0)]
    cross = [FakeExperiment(active_metal="Pt", support_material="Al2O3", oil_conversion_pct=88.0)]
    matches = match_experiments(primary, cross)
    assert len(matches) == 1
    # Falls through to position pairing (the only option with 1 vs 1 and no
    # identity match) -- this is expected and correct; the point is it does
    # NOT claim an identity_key match between genuinely different catalysts.
    assert matches[0].match_method == "position"
