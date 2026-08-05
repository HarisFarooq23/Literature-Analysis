from llm_extract.resolve import diagnose_disagreement_pattern


def test_no_diagnostic_for_single_field_disagreement():
    """One field disagreeing out of many is ordinary noise -- no special
    diagnosis needed."""
    result = diagnose_disagreement_pattern(
        disagreeing_fields=["bet_surface_area_m2g"],
        total_comparable_fields=20,
    )
    assert result is None


def test_no_diagnostic_below_threshold():
    result = diagnose_disagreement_pattern(
        disagreeing_fields=["a", "b", "c"],
        total_comparable_fields=20,  # 15% disagreement, well below default 60% threshold
    )
    assert result is None


def test_diagnostic_fires_at_threshold():
    result = diagnose_disagreement_pattern(
        disagreeing_fields=list("abcdefghijklmno"),  # 15/20 = 75%
        total_comparable_fields=20,
    )
    assert result is not None
    assert "different experiments" in result or "DIFFERENT" in result


def test_diagnostic_mentions_multi_experiment_as_fix():
    result = diagnose_disagreement_pattern(
        disagreeing_fields=list("abcdefghij"),
        total_comparable_fields=10,  # 100% disagreement
    )
    assert result is not None
    assert "multi_experiment" in result


def test_diagnostic_handles_zero_comparable_fields_gracefully():
    result = diagnose_disagreement_pattern(
        disagreeing_fields=[],
        total_comparable_fields=0,
    )
    assert result is None


def test_diagnostic_custom_threshold():
    # 3/10 = 30%, below default 60% threshold but above a custom 25% threshold
    result_default = diagnose_disagreement_pattern(
        disagreeing_fields=list("abc"), total_comparable_fields=10,
    )
    result_custom = diagnose_disagreement_pattern(
        disagreeing_fields=list("abc"), total_comparable_fields=10, threshold=0.25,
    )
    assert result_default is None
    assert result_custom is not None
