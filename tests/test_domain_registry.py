"""
Regression/smoke tests: every domain shipped in domains/ must load, build a
valid Pydantic model, and load its prompt template without error. This is
what should fail loudly if someone edits a domain YAML into an inconsistent
state (e.g. a binary target referencing a renamed field).
"""

from pathlib import Path

import pytest

from llm_extract.domain_registry import get_schema
from llm_extract.pipeline import _build_prompt

_DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains"
_SHIPPED_DOMAINS = ["biodiesel", "methanol_synthesis", "mof_defect_engineering", "saf_hefa"]


@pytest.mark.parametrize("domain_name", _SHIPPED_DOMAINS)
def test_shipped_domain_loads_and_builds_model(domain_name: str):
    schema = get_schema(domain_name, _DOMAINS_DIR)
    assert schema.domain_name == domain_name
    assert "{schema}" in schema.PROMPT_TEMPLATE
    assert "{text}" in schema.PROMPT_TEMPLATE

    json_schema = schema.ExtractionFields.model_json_schema()
    if schema.multi_experiment:
        # extraction_confidence is a flat-domain-only field (see
        # domain_spec.build_pydantic_model's docstring); multi_experiment
        # domains carry experiment_confidence per experiment instead.
        experiment_schema = schema.ExperimentFields.model_json_schema()
        assert "experiment_confidence" in experiment_schema["properties"]
        assert "experiments" in json_schema["properties"]
    else:
        assert "extraction_confidence" in json_schema["properties"]
    # every declared field should round-trip into the dynamically built
    # model, at whichever level (paper vs experiment) it's scoped to.
    for f in schema.spec.fields:
        if schema.multi_experiment and f.scope == "experiment":
            assert f.name in schema.ExperimentFields.model_json_schema()["properties"]
        else:
            assert f.name in json_schema["properties"]


@pytest.mark.parametrize("domain_name", _SHIPPED_DOMAINS)
def test_shipped_domain_prompt_actually_formats_without_error(domain_name: str):
    """Regression test for a real bug found in prompts/saf_hefa.txt: checking
    that '{schema}' and '{text}' appear as SUBSTRINGS in the raw prompt text
    (the test above) is not sufficient -- a literal, unescaped '{' or '}'
    ANYWHERE else in the file (e.g. an instruction like "start with '{' and
    end with '}'") makes str.format() raise KeyError at the exact call site
    pipeline.py uses in production, even though every other check on this
    file passes. This test renders the prompt exactly the way pipeline.py's
    _build_prompt does, with a real field_reminders value, so a stray brace
    anywhere in a domain's prompt file fails the test suite instead of every
    paper in a live run.
    """
    schema = get_schema(domain_name, _DOMAINS_DIR)
    rendered = _build_prompt(
        schema.PROMPT_TEMPLATE,
        schema.ExtractionFields,
        text="sample paper text",
        field_reminders=schema.spec.field_reminders_block(),
    )
    assert "sample paper text" in rendered
    assert "{schema}" not in rendered  # placeholder must have been substituted
    assert "{text}" not in rendered


@pytest.mark.parametrize("domain_name", _SHIPPED_DOMAINS)
def test_shipped_domain_binary_targets_reference_real_fields(domain_name: str):
    schema = get_schema(domain_name, _DOMAINS_DIR)
    field_names = {f.name for f in schema.spec.fields}
    for bt in schema.spec.binary_targets:
        assert bt.field in field_names


def test_unknown_domain_raises_helpful_error():
    with pytest.raises(ValueError, match="No domain spec found"):
        get_schema("nonexistent_domain", _DOMAINS_DIR)
