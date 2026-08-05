"""
test_writer.py — coverage for writer.py's schema/data-dictionary generation.

No test file existed for writer.py before this one. Scope here is
deliberately focused on what changed: documentation of the multi_experiment
meta-columns (experiment_id, ml_core_features_complete, etc.) that
to_dataframe_multi_experiment adds but that _build_json_schema /
_build_data_dictionary never described until now.
"""

from __future__ import annotations

from pathlib import Path

from llm_extract.domain_registry import get_schema
from llm_extract.writer import _build_data_dictionary, _build_json_schema

_DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains"


def test_flat_domain_schema_has_no_multi_experiment_meta_columns():
    """biodiesel is a flat domain -- must not gain experiment_id etc."""
    schema = get_schema("biodiesel", _DOMAINS_DIR)
    json_schema = _build_json_schema(schema)
    assert "experiment_id" not in json_schema["properties"]
    assert "ml_core_features_complete" not in json_schema["properties"]

    doc = _build_data_dictionary(schema)
    assert "experiment_id" not in doc


def test_saf_hefa_schema_documents_experiment_id():
    schema = get_schema("saf_hefa", _DOMAINS_DIR)
    json_schema = _build_json_schema(schema)
    assert "experiment_id" in json_schema["properties"]
    assert json_schema["properties"]["experiment_id"]["type"] == "string"
    assert "evidence_text" in json_schema["properties"]
    assert "match_method" in json_schema["properties"]


def test_experiment_id_documentation_does_not_overclaim_cross_extraction_stability():
    """Regression guard: experiment_id is unique/stable within one dataset
    build (repeated build-dataset against the same extracted_records.jsonl)
    but NOT guaranteed stable across a full re-extraction, since exp_idx
    comes from resolve.py's tier-bucketed cross-model matching, not fixed
    paper structure. An earlier version of this documentation called it
    "stable, globally unique" without that scope, which is misleading for
    anyone using it as a long-lived RAG/vector-index primary key across
    pipeline versions. Both the JSON schema and the human-readable data
    dictionary must state the caveat, not just one of them.
    """
    schema = get_schema("saf_hefa", _DOMAINS_DIR)

    json_schema = _build_json_schema(schema)
    schema_desc = json_schema["properties"]["experiment_id"]["description"]
    assert "NOT guaranteed stable across a full re-extraction" in schema_desc
    assert "evidence_location" in schema_desc  # points at the better anchor

    doc = _build_data_dictionary(schema)
    doc_row = next(line for line in doc.splitlines() if "`experiment_id`" in line)
    assert "NOT guaranteed stable across a full re-extraction" in doc_row
    assert "evidence_location" in doc_row


def test_saf_hefa_schema_documents_ml_core_features_complete():
    """saf_hefa declares ml_core_fields -- the completeness column must be
    documented, and the description must name the actual fields it checks
    (this fails loudly if ml_core_fields is edited without keeping the
    generated description consistent)."""
    schema = get_schema("saf_hefa", _DOMAINS_DIR)
    json_schema = _build_json_schema(schema)
    assert "ml_core_features_complete" in json_schema["properties"]
    desc = json_schema["properties"]["ml_core_features_complete"]["description"]
    for f in schema.spec.ml_core_fields:
        assert f in desc


def test_domain_without_ml_core_fields_has_no_completeness_column(tmp_path: Path):
    """A multi_experiment domain that does NOT declare ml_core_fields must
    not get a misleading always-absent-meaning column."""
    import yaml
    domains_dir = tmp_path / "domains"
    domains_dir.mkdir()
    (domains_dir / "test_no_ml_core.yaml").write_text(yaml.dump({
        "domain_name": "test_no_ml_core",
        "prompt_template_path": "p.txt",
        "multi_experiment": True,
        "fields": [{"name": "active_metal", "type": "str", "scope": "experiment"}],
    }), encoding="utf-8")
    (domains_dir / "p.txt").write_text("{schema} {text}", encoding="utf-8")

    schema = get_schema("test_no_ml_core", domains_dir)
    json_schema = _build_json_schema(schema)
    assert "ml_core_features_complete" not in json_schema["properties"]

    doc = _build_data_dictionary(schema)
    assert "ml_core_features_complete" not in doc


def test_saf_hefa_data_dictionary_documents_new_kinetics_and_summary_fields():
    schema = get_schema("saf_hefa", _DOMAINS_DIR)
    doc = _build_data_dictionary(schema)
    for field in ["kinetic_model", "rate_constant", "pre_exponential_factor_is_log10",
                  "structure_property_summary", "experiment_id"]:
        assert f"`{field}`" in doc, f"{field} missing from data_dictionary.md"


def test_data_dictionary_renders_without_error_for_every_shipped_domain():
    for domain_name in ["biodiesel", "methanol_synthesis", "mof_defect_engineering", "saf_hefa"]:
        schema = get_schema(domain_name, _DOMAINS_DIR)
        doc = _build_data_dictionary(schema)
        assert domain_name in doc
        json_schema = _build_json_schema(schema)
        assert "properties" in json_schema
