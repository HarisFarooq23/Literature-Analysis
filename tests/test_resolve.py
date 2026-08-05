"""
resolve.py is domain-agnostic by design, so these tests build a small
throwaway Pydantic model directly rather than depending on any real
domains/*.yaml content — that keeps these tests stable even if biodiesel.yaml
(or any other domain) changes.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from llm_extract.resolve import compare_two, resolve_with_tie_breaker


class _DummyFields(BaseModel):
    catalyst_type: Literal["heterogeneous_base", "enzymatic", "ionic_liquid", "not_specified"]
    cetane_number: Optional[float] = None
    oxidative_stability_h: Optional[float] = None
    extraction_confidence: float = Field(ge=0.0, le=1.0)


def _fields(**overrides) -> _DummyFields:
    base = dict(
        catalyst_type="heterogeneous_base",
        cetane_number=54.0,
        oxidative_stability_h=None,
        extraction_confidence=0.9,
    )
    base.update(overrides)
    return _DummyFields(**base)


def test_identical_fields_produce_no_disagreement():
    a, b = _fields(), _fields()
    assert compare_two(a, b, numeric_tolerance=0.10) == []


def test_numeric_within_tolerance_agrees():
    a = _fields(cetane_number=54.0)
    b = _fields(cetane_number=55.5)  # ~2.7% relative difference
    assert "cetane_number" not in compare_two(a, b, numeric_tolerance=0.10)


def test_numeric_outside_tolerance_disagrees():
    a = _fields(cetane_number=50.0)
    b = _fields(cetane_number=60.0)  # ~17% relative difference
    assert "cetane_number" in compare_two(a, b, numeric_tolerance=0.10)


def test_categorical_mismatch_disagrees():
    a = _fields(catalyst_type="heterogeneous_base")
    b = _fields(catalyst_type="enzymatic")
    assert "catalyst_type" in compare_two(a, b, numeric_tolerance=0.10)


def test_one_null_one_value_counts_as_disagreement():
    a = _fields(oxidative_stability_h=None)
    b = _fields(oxidative_stability_h=7.5)
    assert "oxidative_stability_h" in compare_two(a, b, numeric_tolerance=0.10)


def test_extraction_confidence_is_excluded_from_agreement():
    a = _fields(extraction_confidence=0.5)
    b = _fields(extraction_confidence=0.99)
    assert "extraction_confidence" not in compare_two(a, b, numeric_tolerance=0.10)


def test_tie_breaker_resolves_categorical_by_majority():
    primary = _fields(catalyst_type="heterogeneous_base")
    cross = _fields(catalyst_type="enzymatic")
    tie = _fields(catalyst_type="heterogeneous_base")
    result = resolve_with_tie_breaker(primary, cross, tie, ["catalyst_type"], 0.10)
    assert result.resolved["catalyst_type"] == "heterogeneous_base"
    assert result.unresolved_fields == []
    assert result.agreement_status == "resolved_after_escalation"


def test_tie_breaker_resolves_numeric_by_median():
    primary = _fields(cetane_number=50.0)
    cross = _fields(cetane_number=60.0)
    tie = _fields(cetane_number=52.0)
    result = resolve_with_tie_breaker(primary, cross, tie, ["cetane_number"], 0.10)
    assert result.resolved["cetane_number"] == 52.0


def test_three_way_split_categorical_is_unresolved():
    primary = _fields(catalyst_type="heterogeneous_base")
    cross = _fields(catalyst_type="enzymatic")
    tie = _fields(catalyst_type="ionic_liquid")
    result = resolve_with_tie_breaker(primary, cross, tie, ["catalyst_type"], 0.10)
    assert "catalyst_type" in result.unresolved_fields
    assert result.resolved["catalyst_type"] is None
    assert result.agreement_status == "unresolved_conflict"


def test_missing_tie_breaker_falls_back_to_two_models():
    primary = _fields(cetane_number=50.0)
    cross = _fields(cetane_number=60.0)
    result = resolve_with_tie_breaker(primary, cross, None, ["cetane_number"], 0.10)
    assert result.resolved["cetane_number"] == 55.0  # true median of [50.0, 60.0]
