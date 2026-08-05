from pathlib import Path

import pandas as pd
import pytest
import yaml

from llm_extract.domain_spec import (
    FieldSpec,
    build_binary_targets_generic,
    build_pydantic_model,
    field_docs,
    load_domain_spec,
    load_prompt_template,
)

_MINIMAL_DOMAIN = {
    "domain_name": "test_domain",
    "description": "A minimal domain for testing.",
    "prompt_template_path": "test_prompt.txt",
    "fields": [
        {"name": "catalyst_family", "type": "str", "enum": ["A", "B", "not_specified"], "required": True},
        {"name": "yield_pct", "type": "float", "description": "Product yield", "unit": "%"},
        {"name": "is_stable", "type": "bool", "description": "Stability flag"},
    ],
    "binary_targets": [
        {"name": "high_yield", "field": "yield_pct", "op": ">=", "value": 80.0, "description": "Yield >= 80%"},
    ],
}


def _write_domain(tmp_path: Path, domain_dict: dict) -> Path:
    yaml_path = tmp_path / "test_domain.yaml"
    yaml_path.write_text(yaml.dump(domain_dict), encoding="utf-8")
    (tmp_path / "test_prompt.txt").write_text("Schema: {schema}\nText: {text}\n", encoding="utf-8")
    return yaml_path


def test_load_domain_spec_round_trips(tmp_path: Path):
    yaml_path = _write_domain(tmp_path, _MINIMAL_DOMAIN)
    spec = load_domain_spec(yaml_path)
    assert spec.domain_name == "test_domain"
    assert {f.name for f in spec.fields} == {"catalyst_family", "yield_pct", "is_stable"}
    assert spec.binary_targets[0].name == "high_yield"


def test_load_domain_spec_rejects_unknown_target_field(tmp_path: Path):
    bad = {**_MINIMAL_DOMAIN, "binary_targets": [
        {"name": "x", "field": "does_not_exist", "op": ">=", "value": 1.0}
    ]}
    yaml_path = _write_domain(tmp_path, bad)
    with pytest.raises(ValueError, match="unknown field"):
        load_domain_spec(yaml_path)


def test_load_domain_spec_rejects_between_without_two_values(tmp_path: Path):
    bad = {**_MINIMAL_DOMAIN, "binary_targets": [
        {"name": "x", "field": "yield_pct", "op": "between", "value": 5.0}
    ]}
    yaml_path = _write_domain(tmp_path, bad)
    with pytest.raises(ValueError, match="between"):
        load_domain_spec(yaml_path)


def test_load_prompt_template_reads_relative_path(tmp_path: Path):
    yaml_path = _write_domain(tmp_path, _MINIMAL_DOMAIN)
    spec = load_domain_spec(yaml_path)
    text = load_prompt_template(yaml_path, spec)
    assert "{schema}" in text and "{text}" in text


def test_build_pydantic_model_enforces_enum_and_optionality(tmp_path: Path):
    yaml_path = _write_domain(tmp_path, _MINIMAL_DOMAIN)
    spec = load_domain_spec(yaml_path)
    Model = build_pydantic_model(spec)

    instance = Model(catalyst_family="A", yield_pct=None, is_stable=None, extraction_confidence=0.8)
    assert instance.yield_pct is None

    with pytest.raises(Exception):
        Model(catalyst_family="not_a_valid_enum_value", extraction_confidence=0.8)


def test_build_pydantic_model_requires_required_field(tmp_path: Path):
    yaml_path = _write_domain(tmp_path, _MINIMAL_DOMAIN)
    spec = load_domain_spec(yaml_path)
    Model = build_pydantic_model(spec)

    with pytest.raises(Exception):
        Model(extraction_confidence=0.8)  # catalyst_family is required, omitted here


def test_build_pydantic_model_always_adds_confidence_field(tmp_path: Path):
    yaml_path = _write_domain(tmp_path, _MINIMAL_DOMAIN)
    spec = load_domain_spec(yaml_path)
    Model = build_pydantic_model(spec)
    schema = Model.model_json_schema()
    assert "extraction_confidence" in schema["properties"]


def test_build_binary_targets_generic_respects_nullability(tmp_path: Path):
    df = pd.DataFrame({"yield_pct": [85.0, 50.0, None]})
    spec = load_domain_spec(_write_domain(tmp_path, _MINIMAL_DOMAIN))
    out = build_binary_targets_generic(df, spec)

    assert out.loc[0, "high_yield"] == True
    assert out.loc[1, "high_yield"] == False
    assert pd.isna(out.loc[2, "high_yield"])  # not reported != fails the target


def test_build_binary_targets_generic_skips_missing_columns(tmp_path: Path):
    df = pd.DataFrame({"some_other_column": [1, 2, 3]})
    spec = load_domain_spec(_write_domain(tmp_path, _MINIMAL_DOMAIN))
    out = build_binary_targets_generic(df, spec)
    assert "high_yield" not in out.columns  # field wasn't extracted this run -- no crash


def test_field_docs_uses_description_and_unit(tmp_path: Path):
    yaml_path = _write_domain(tmp_path, _MINIMAL_DOMAIN)
    spec = load_domain_spec(yaml_path)
    docs = {name: (desc, unit) for name, desc, unit in field_docs(spec)}
    assert docs["yield_pct"] == ("Product yield", "%")


def test_field_reminders_block_empty_when_no_field_emphasized(tmp_path: Path):
    spec = load_domain_spec(_write_domain(tmp_path, _MINIMAL_DOMAIN))
    assert spec.field_reminders_block() == ""


def test_field_reminders_block_renders_only_emphasized_fields(tmp_path: Path):
    domain = {
        **_MINIMAL_DOMAIN,
        "fields": [
            {"name": "catalyst_family", "type": "str", "enum": ["A", "B", "not_specified"],
             "required": True, "emphasize": True,
             "description": "Isolate from support -- do not fold loading in."},
            {"name": "yield_pct", "type": "float", "description": "Product yield", "unit": "%"},
        ],
    }
    spec = load_domain_spec(_write_domain(tmp_path, domain))
    block = spec.field_reminders_block()
    assert "catalyst_family" in block
    assert "Isolate from support" in block
    assert "yield_pct" not in block  # not emphasized -> not repeated as prose


def test_field_reminders_block_collapses_folded_yaml_newlines(tmp_path: Path):
    """YAML's '>' folded-scalar descriptions contain embedded newlines from
    line-wrapping the source file; the rendered reminder should read as one
    clean sentence/paragraph, not reproduce the YAML file's line breaks."""
    domain = {
        **_MINIMAL_DOMAIN,
        "binary_targets": [],
        "fields": [
            {"name": "catalyst_family", "type": "str", "enum": ["A", "B", "not_specified"],
             "required": True, "emphasize": True,
             "description": "First line of the description\ncontinues on a second line\nand a third."},
        ],
    }
    spec = load_domain_spec(_write_domain(tmp_path, domain))
    block = spec.field_reminders_block()
    assert "First line of the description continues on a second line and a third." in block


# ---------------------------------------------------------------------------
# FieldSpec.max_length
# ---------------------------------------------------------------------------

def test_max_length_only_valid_for_str_type():
    with pytest.raises(ValueError, match="max_length is only valid for type 'str'"):
        FieldSpec(name="yield_pct", type="float", max_length=100)


def test_max_length_redundant_with_enum_rejected():
    with pytest.raises(ValueError, match="max_length is redundant with enum"):
        FieldSpec(name="catalyst_family", type="str", enum=["A", "B"], max_length=100)


def test_max_length_enforced_on_built_model(tmp_path: Path):
    domain = {
        **_MINIMAL_DOMAIN,
        "binary_targets": [],
        "fields": [
            {"name": "summary", "type": "str", "max_length": 20},
        ],
    }
    spec = load_domain_spec(_write_domain(tmp_path, domain))
    Model = build_pydantic_model(spec)

    Model(summary="short enough", extraction_confidence=0.8)  # 12 chars, fine
    with pytest.raises(Exception):  # pydantic ValidationError
        Model(summary="this string is definitely longer than twenty characters",
              extraction_confidence=0.8)


def test_max_length_none_by_default_no_constraint(tmp_path: Path):
    yaml_path = _write_domain(tmp_path, _MINIMAL_DOMAIN)
    spec = load_domain_spec(yaml_path)
    Model = build_pydantic_model(spec)
    # catalyst_family has no max_length declared -- an arbitrarily long
    # value (within reason) must not be rejected.
    Model(catalyst_family="A", extraction_confidence=0.8)


# ---------------------------------------------------------------------------
# DomainSpec.ml_core_fields
# ---------------------------------------------------------------------------

def test_ml_core_fields_unknown_field_raises(tmp_path: Path):
    domain = {**_MINIMAL_DOMAIN, "ml_core_fields": ["not_a_real_field"]}
    with pytest.raises(ValueError, match="ml_core_fields references unknown field"):
        load_domain_spec(_write_domain(tmp_path, domain))


def test_ml_core_fields_valid_field_names_pass(tmp_path: Path):
    domain = {**_MINIMAL_DOMAIN, "ml_core_fields": ["yield_pct", "is_stable"]}
    spec = load_domain_spec(_write_domain(tmp_path, domain))
    assert spec.ml_core_fields == ["yield_pct", "is_stable"]


def test_ml_core_fields_defaults_to_empty_list(tmp_path: Path):
    yaml_path = _write_domain(tmp_path, _MINIMAL_DOMAIN)
    spec = load_domain_spec(yaml_path)
    assert spec.ml_core_fields == []
