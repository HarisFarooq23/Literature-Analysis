"""
test_saf_hefa_domain.py — regression tests for the saf_hefa.yaml domain
specifically (as opposed to the generic multi_experiment machinery already
covered by test_domain_spec.py / test_resolve_experiments.py). Locks in:

  - the domain loads and builds without error
  - the 9 catalyst-preparation/support-taxonomy fields added for commercial
    catalyst-informatics use are present with the correct type/scope/enum
  - a realistic multi-catalyst paper record (varying synthesis method,
    calcination, reduction, promoter loading, support type) validates end
    to end through the actual Pydantic model the pipeline uses
  - the confidence-rescale fix and multi_experiment shape both still work
    together on this specific domain, not just in the abstract
"""

from pathlib import Path

import pytest

from llm_extract.domain_registry import get_schema
from llm_extract.pipeline import _build_prompt

DOMAINS_DIR = Path(__file__).parent.parent / "domains"


@pytest.fixture(scope="module")
def schema():
    return get_schema("saf_hefa", DOMAINS_DIR)


def test_domain_loads_and_is_multi_experiment(schema):
    assert schema.domain_name == "saf_hefa"
    assert schema.multi_experiment is True
    assert schema.extract_claims is True


NEW_FIELD_SPECS = {
    "support_type": ("str", True),          # (type, is_enum)
    "synthesis_method": ("str", True),
    "calcination_temperature_C": ("float", False),
    "calcination_time_h": ("float", False),
    "calcination_atmosphere": ("str", True),
    "reduction_temperature_C": ("float", False),
    "reduction_time_h": ("float", False),
    "promoter_loading_wt_pct": ("float", False),
    "particle_size_method": ("str", True),
    "catalyst_type": ("str", True),
    "co2_uptake_mmol_g": ("float", False),
    "tpr_peak_low_C": ("float", False),
    "tpr_peak_high_C": ("float", False),
}


@pytest.mark.parametrize("field_name,expected", NEW_FIELD_SPECS.items())
def test_new_field_present_with_correct_scope(schema, field_name, expected):
    field_specs = {f.name: f for f in schema.spec.fields}
    assert field_name in field_specs, f"{field_name} missing from saf_hefa.yaml"
    spec = field_specs[field_name]
    assert spec.scope == "experiment", f"{field_name} must be scope: experiment"
    expected_type, is_enum = expected
    assert spec.type == expected_type
    if is_enum:
        assert spec.enum is not None and len(spec.enum) > 0, f"{field_name} should be a categorical enum"
    else:
        assert spec.enum is None


# Regression test for the specific bug this file documents: prompts/saf_hefa.txt
# once instructed the model to populate field names (surface_area_m2_g,
# acid_site_density, tpo_peaks_C, tpd_peaks_C, catalyst_type, ...) that did
# not match -- or in three cases, did not exist as -- any field actually
# declared in saf_hefa.yaml. Schema-constrained generation mostly papers over
# this (the model is grammar-constrained to the real schema regardless of
# what the prompt text says), but the format='json' fallback path in
# ollama_client.py has no such constraint: on that path the model follows
# the prompt's field names literally, producing keys that fail Pydantic
# validation and burn a retry that would otherwise have succeeded. Any field
# name explicitly called out in the prompt text must exist in the schema, or
# this exact failure mode recurs invisibly the next time someone edits
# either file without updating the other.
# the prompt's field names literally, producing keys that fail Pydantic
# validation and burn a retry that would otherwise have succeeded. Any field
# name explicitly called out in the prompt text (or auto-generated into it
# via FieldSpec.emphasize -- see domain_spec.py) must exist in the schema,
# or this exact failure mode recurs invisibly the next time someone edits
# either file without updating the other.
#
# Split in two because they're checked differently: HAND-WRITTEN rules
# (cross-field reasoning that can't be expressed as one field's description,
# e.g. the three-temperatures disambiguation) live literally in
# prompts/saf_hefa.txt and are checked against the raw, unrendered template.
# AUTO-GENERATED rules (FieldSpec.emphasize: true fields, e.g. active_metal's
# isolation instruction) are injected at render time via
# DomainSpec.field_reminders_block() -- checking for them in the raw
# template would fail even though the rendered prompt the model actually
# sees is correct, so these are checked post-render instead.
_HAND_WRITTEN_RULE_FIELD_NAMES = [
    "synthesis_method", "calcination_temperature_C",
    "reduction_temperature_C", "reaction_temperature_C", "deoxygenation_pathway",
    "downstream_processing", "astm_reference_standard", "reducibility_temp_C",
    "tpr_peak_low_C", "tpr_peak_high_C", "co2_uptake_mmol_g", "catalyst_type",
    "kinetic_model", "rate_constant", "rate_constant_unit_reported",
    "reaction_order", "pre_exponential_factor", "pre_exponential_factor_is_log10",
    "structure_property_summary",
]

_EMPHASIZED_FIELD_NAMES = ["active_metal", "support_type", "promoter_loading_wt_pct"]


def test_hand_written_rules_reference_fields_that_exist_in_schema(schema):
    prompt_text = schema.PROMPT_TEMPLATE
    field_names = schema.spec.field_names()
    for name in _HAND_WRITTEN_RULE_FIELD_NAMES:
        assert name in prompt_text, (
            f"Expected prompts/saf_hefa.txt to reference '{name}' (update this "
            "test's list if the prompt was intentionally rewritten)"
        )
        assert name in field_names, (
            f"prompts/saf_hefa.txt references field '{name}' which does not exist "
            f"in saf_hefa.yaml. Known fields: {sorted(field_names)}"
        )


def test_emphasized_fields_are_flagged_in_yaml_and_reach_the_rendered_prompt(schema):
    """active_metal / support_type / promoter_loading_wt_pct must each set
    emphasize: true in saf_hefa.yaml (so their non-trivial extraction logic
    gets a standalone prose rule, not just a schema-embedded description --
    see FieldSpec.emphasize's docstring for why that distinction matters for
    the 8-12B-class local models this pipeline targets), and the generated
    reminder text must actually reach the fully-rendered prompt.
    """
    field_specs = {f.name: f for f in schema.spec.fields}
    for name in _EMPHASIZED_FIELD_NAMES:
        assert name in field_specs, f"{name} missing from saf_hefa.yaml"
        assert field_specs[name].emphasize, (
            f"{name} should set emphasize: true in saf_hefa.yaml -- its "
            "extraction logic is non-trivial enough to need a standalone "
            "prose rule, not just a schema-embedded description."
        )

    reminders = schema.spec.field_reminders_block()
    for name in _EMPHASIZED_FIELD_NAMES:
        assert name in reminders

    rendered = _build_prompt(
        schema.PROMPT_TEMPLATE, schema.ExtractionFields,
        text="sample paper text", field_reminders=reminders,
    )
    for name in _EMPHASIZED_FIELD_NAMES:
        assert name in rendered, (
            f"'{name}' reminder was generated but did not reach the final "
            "rendered prompt -- check prompts/saf_hefa.txt has a "
            "{field_reminders} placeholder."
        )


def test_support_type_enum_covers_common_catalyst_informatics_classes(schema):
    field_specs = {f.name: f for f in schema.spec.fields}
    enum = set(field_specs["support_type"].enum)
    expected_minimum = {"zeolite", "mesoporous_silica", "metal_oxide",
                         "mixed_metal_oxide", "activated_carbon", "MOF", "not_specified"}
    assert expected_minimum <= enum


def test_synthesis_method_enum_covers_common_prep_routes(schema):
    field_specs = {f.name: f for f in schema.spec.fields}
    enum = set(field_specs["synthesis_method"].enum)
    expected_minimum = {"incipient_wetness_impregnation", "co_precipitation",
                         "sol_gel", "deposition_precipitation", "not_specified"}
    assert expected_minimum <= enum


def test_realistic_multi_catalyst_paper_validates_end_to_end(schema):
    """A realistic 2-catalyst paper record exercising every new field,
    varying synthesis method and calcination/reduction conditions between
    the two catalysts (the exact scenario this domain exists to handle
    correctly -- distinct experiment rows, not one averaged record)."""
    PaperModel = schema.ExtractionFields
    ExperimentModel = schema.ExperimentFields

    instance = PaperModel(
        feedstock_type="waste_cooking_oil",
        ffa_content_pct=3.2,
        experiments=[
            ExperimentModel(
                active_metal="NiMo",
                support_material="gamma-Al2O3",
                support_type="metal_oxide",
                promoter="P",
                promoter_loading_wt_pct=2.0,
                metal_loading_wt_pct=15.0,
                synthesis_method="incipient_wetness_impregnation",
                calcination_temperature_C=500.0,
                calcination_time_h=4.0,
                calcination_atmosphere="static_air",
                reduction_temperature_C=400.0,
                reduction_time_h=2.0,
                bet_surface_area_m2g=215.0,
                metal_particle_size_nm=8.5,
                particle_size_method="TEM",
                reaction_temperature_C=320.0,
                oil_conversion_pct=94.0,
                experiment_confidence=95,  # deliberately a percentage -- must auto-rescale
            ),
            ExperimentModel(
                active_metal="Pt",
                support_material="HZSM-5",
                support_type="zeolite",
                metal_loading_wt_pct=1.0,
                synthesis_method="ion_exchange",
                calcination_temperature_C=550.0,
                calcination_atmosphere="flowing_air",
                reduction_temperature_C=350.0,
                bet_surface_area_m2g=350.0,
                metal_particle_size_nm=2.1,
                particle_size_method="XRD_Scherrer",
                reaction_temperature_C=300.0,
                oil_conversion_pct=88.0,
                experiment_confidence=0.9,
            ),
        ],
        claims=[],
    )

    assert len(instance.experiments) == 2
    ni_mo, pt = instance.experiments

    # Confidence-rescale fix verified on THIS domain, not just in the abstract
    assert abs(ni_mo.experiment_confidence - 0.95) < 1e-9

    # Distinct prep conditions preserved per experiment (not merged/averaged)
    assert ni_mo.synthesis_method == "incipient_wetness_impregnation"
    assert pt.synthesis_method == "ion_exchange"
    assert ni_mo.calcination_temperature_C == 500.0
    assert pt.calcination_temperature_C == 550.0
    assert ni_mo.particle_size_method == "TEM"
    assert pt.particle_size_method == "XRD_Scherrer"
    assert ni_mo.support_type == "metal_oxide"
    assert pt.support_type == "zeolite"


def test_promoter_and_promoter_loading_independently_nullable(schema):
    """Confirms promoter_loading_wt_pct does not require promoter to be
    set at the schema level (the 'don't report a loading with no promoter'
    rule is prompt guidance for the LLM, not a hard Pydantic constraint --
    a hard constraint here would reject legitimate partial extractions,
    e.g. a paper stating a promoter loading in a table without repeating
    the promoter's identity in that same row)."""
    ExperimentModel = schema.ExperimentFields
    inst = ExperimentModel(
        active_metal="Ni",
        oil_conversion_pct=80.0,
        reaction_temperature_C=300.0,
        promoter=None,
        promoter_loading_wt_pct=None,
        experiment_confidence=0.9,
    )
    assert inst.promoter is None
    assert inst.promoter_loading_wt_pct is None


def test_binary_targets_still_reference_valid_experiment_scope_fields(schema):
    """The existing ASTM binary targets must still validate against the
    updated field list (i.e. adding new fields must not have broken the
    scope check in domain_spec.load_domain_spec)."""
    target_names = {bt.name for bt in schema.spec.binary_targets}
    assert "meets_freezing_point_spec" in target_names
    assert "high_conversion" in target_names
