"""
These tests run against the real domains/biodiesel.yaml shipped with the
package (not a synthetic fixture) -- they double as a regression check that
the shipped domain spec stays loadable and produces the expected target
columns as the YAML evolves.
"""

import json
from pathlib import Path

from llm_extract.normalize import normalize, split_success_failure

_DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains"


def _record(paper_id: str, fields: dict | None, **overrides) -> dict:
    base = {
        "paper_id": paper_id,
        "doi": "10.1/x", "title": "t", "first_author": "A", "journal": "J",
        "publication_year": 2022, "material_class": "catalyst", "is_review": False,
        "extracted_at": "2026-01-01T00:00:00+00:00",
        "agreement_status": "full_agreement" if fields else "extraction_failed",
        "models_used": ["granite4.1:8b", "gemma3:12b"],
        "disagreeing_fields": [], "unresolved_fields": [],
        "primary_confidence": 0.9, "cross_check_confidence": 0.9,
        "fields": fields,
    }
    base.update(overrides)
    return base


def test_split_success_failure_separates_correctly():
    records = [_record("p1", {"cetane_number": 54.0}), _record("p2", None)]
    successes, failures = split_success_failure(records)
    assert len(successes) == 1 and len(failures) == 1


def test_normalize_end_to_end_against_real_biodiesel_domain(tmp_path: Path):
    records_path = tmp_path / "extracted_records.jsonl"
    rows = [
        _record("p1", {
            "catalyst_type": "heterogeneous_base", "catalyst_loading_wt_pct": 3.0,
            "methanol_to_oil_molar_ratio": 9.0, "reaction_temperature_C": 60.0,
            "reaction_time_min": 90.0, "wco_ffa_content_pct": None,
            "fame_yield_pct": 96.0, "cetane_number": 54.0,
            "kinematic_viscosity_40C_cSt": 4.8, "cfpp_or_pour_point_C": -3.0,
            "oxidative_stability_h": 8.0, "density_kg_m3": 880.0,
            "flash_point_C": 170.0, "reference_standard": "ASTM_D6751",
            "extraction_confidence": 0.92,
        }),
        _record("p2", None, errors={"granite4.1:8b": "timeout"}),
    ]
    with open(records_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    tables = normalize(records_path, schema_name="biodiesel", domains_dir=_DOMAINS_DIR)
    assert len(tables["clean"]) == 1
    assert len(tables["failures"]) == 1
    assert tables["clean"].loc[0, "meets_cetane_spec"] == True
    assert tables["clean"].loc[0, "meets_viscosity_spec"] == True
    assert tables["clean"].loc[0, "meets_oxidative_spec"] == True
    assert tables["clean"].loc[0, "meets_cfpp_target"] == True


def test_normalize_raises_if_nothing_succeeded(tmp_path: Path):
    records_path = tmp_path / "extracted_records.jsonl"
    with open(records_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_record("p1", None)) + "\n")

    import pytest
    with pytest.raises(ValueError):
        normalize(records_path, schema_name="biodiesel", domains_dir=_DOMAINS_DIR)
