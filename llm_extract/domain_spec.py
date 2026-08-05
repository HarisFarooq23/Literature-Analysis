"""
domain_spec.py — turns a YAML domain specification into a working Pydantic
extraction schema and a generic binary-target builder, with no Python code
required per domain.

This is what makes adding methanol synthesis, MOF defect engineering, or any
other catalysis topic a YAML-editing task instead of a coding task. The rest
of the package (reader, ollama_client, pipeline, normalize, writer) only
ever sees the resulting Pydantic model + functions — it has no idea whether
they came from YAML or from a hand-written Python module.

Two extraction shapes are supported, selected per-domain by
DomainSpec.multi_experiment:

  - multi_experiment: false (default) — one flat record per paper. Every
    domain that predates this feature (biodiesel, methanol_synthesis,
    mof_defect_engineering) uses this shape and is unaffected by anything
    below it.

  - multi_experiment: true — many catalysis papers report several
    catalysts (or several conditions for one catalyst) side by side in the
    same table, e.g. Ni/Al2O3 vs NiMo/SAPO-11 vs Pt/Al2O3. A flat, one-
    record-per-paper schema forces each extracting model to arbitrarily
    pick one row and silently discard the rest. Two models independently
    picking DIFFERENT rows then look like they "disagree" on every numeric
    field (BET area, conversion, temperature, ...) even though both
    extractions are individually correct -- they just describe different
    experiments. This is a schema-design bug, not a measurement-noise
    problem, and no amount of retuning the numeric tolerance fixes it.
    multi_experiment: true is the actual fix: the paper-level model becomes
        { <paper-scope fields>, experiments: [ <experiment-scope fields,
          each with its own evidence span>, ... ], claims: [ <causal/
          comparative claim>, ... ] }
    so the unit of analysis downstream is one experiment, not one paper,
    and cross-model comparison happens per-matched-experiment (see
    resolve.match_experiments), never across unrelated rows.

Each FieldSpec carries a `scope` of "paper" (feedstock, bibliographic-style
facts that hold for the whole paper) or "experiment" (anything that can
differ between catalysts/conditions reported in the same paper: catalyst
identity, conditions, outcomes). scope is ignored when multi_experiment is
false.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
import yaml
from pydantic import BaseModel, Field, create_model, model_validator

_TYPE_MAP: dict[str, Any] = {"str": str, "float": float, "int": int, "bool": bool}
_OPS = {">=", "<=", "==", "!=", "between"}

# Field names on which the LLM self-reports a confidence score. Both are
# rescaled from a 0-100 percentage to 0-1 if a model returns a percentage
# despite the schema/prompt asking for a decimal -- see
# _confidence_rescale_validator below for the full rationale.
_CONFIDENCE_FIELD_NAMES = ("extraction_confidence", "experiment_confidence")


class FieldSpec(BaseModel):
    name: str
    type: Literal["str", "float", "int", "bool"]
    description: str = ""
    unit: str = "-"
    enum: Optional[list[str]] = None          # only meaningful when type == "str"
    required: bool = False
    # Only meaningful when DomainSpec.multi_experiment is true. "experiment"
    # fields can differ between catalysts/conditions reported in the same
    # paper (catalyst identity, conditions, outcomes); "paper" fields are
    # assumed constant across every experiment in the paper (feedstock,
    # bibliographic-style facts). Ignored entirely in flat (single-record)
    # domains.
    scope: Literal["paper", "experiment"] = "experiment"
    # Only meaningful for type == "str". Caps the field at N characters via
    # Pydantic's Field(max_length=...), enforced at validation time (i.e.
    # the model's response is REJECTED and retried if it exceeds this, not
    # silently truncated -- truncating a free-text synthesis field
    # post-hoc can cut it off mid-sentence with a misleading claim). Exists
    # for free-text fields that are a genuine hallucination/verbosity risk
    # (e.g. structure_property_summary) where an unbounded string could
    # balloon into an ungrounded essay, inflate output-token cost across a
    # 30-experiment paper, or reproduce a long verbatim passage from the
    # source text. Do not use this for fields with a natural short answer
    # (support_material, promoter, etc.) -- it adds a failure mode
    # (retry-triggering) for no benefit there.
    max_length: Optional[int] = None
    # If true, this field's name+description is ALSO rendered as a
    # standalone, numbered prose reminder in the prompt (see
    # DomainSpec.field_reminders_block below), in addition to living in
    # the JSON schema description as normal.
    #
    # Why this exists instead of just relying on the schema description:
    # the schema-embedded description IS visible to the model (it's part
    # of the JSON dumped into {schema}), but this package's own
    # _confidence_rescale_validator documents empirically observed cases
    # (gemma3:12b, granite4.1) where these specific small (8-12B) local
    # models fail to obey a constraint that lives ONLY in schema metadata
    # -- even one as simple as a numeric minimum/maximum -- and need it
    # restated as an explicit, standalone instruction to follow reliably.
    # A field likely to have the same problem (non-trivial extraction
    # logic: isolating one substring from another, a multi-branch mapping
    # rule, a cross-field dependency) should set emphasize: true rather
    # than have its rule hand-copied into the prompt .txt file, which is
    # what let prompts/saf_hefa.txt drift out of sync with this YAML in
    # the first place (see test_prompt_only_references_fields_that_exist_
    # in_schema). One source of truth (this file), rendered into both
    # places automatically.
    emphasize: bool = False

    def model_post_init(self, __context: Any) -> None:
        if self.enum and self.type != "str":
            raise ValueError(f"Field '{self.name}': enum is only valid for type 'str'")
        if self.max_length is not None and self.type != "str":
            raise ValueError(f"Field '{self.name}': max_length is only valid for type 'str'")
        if self.max_length is not None and self.enum:
            raise ValueError(
                f"Field '{self.name}': max_length is redundant with enum -- "
                "enum values already bound the string to a fixed, known set."
            )


class EvidenceSpan(BaseModel):
    """Where in the source document a value/experiment came from. Added
    automatically to every experiment record in multi_experiment domains —
    not something a domain author declares in YAML. Both fields are
    optional because not every model reliably reports them; a missing
    evidence span is a data-quality signal (lower trust), not a hard
    extraction failure."""

    source_location: Optional[str] = Field(
        default=None,
        description="Where this experiment's data was found, e.g. 'Table 3, "
                     "page 8' or 'Results section, paragraph 2'.",
    )
    evidence_text: Optional[str] = Field(
        default=None,
        description="Verbatim sentence(s) or table row supporting the "
                     "extracted values for this experiment.",
    )


class ClaimRecord(BaseModel):
    """A single machine-readable causal/comparative claim, e.g.
    {subject: 'Bronsted acidity', relation: 'increases', object:
    'hydroisomerization selectivity'}. Domain-independent: the same shape
    is used for every multi_experiment domain, so there is nothing to
    declare per-domain beyond turning extract_claims on."""

    subject: str = Field(description="The factor being varied or measured, e.g. 'Ni loading'.")
    relation: Literal["increases", "decreases", "has_no_effect_on", "is_required_for", "correlates_with"]
    object: str = Field(description="The outcome affected, e.g. 'conversion'.")
    evidence_text: str = Field(description="Verbatim sentence the claim was extracted from.")
    confidence: float = Field(ge=0.0, le=1.0)


class BinaryTargetSpec(BaseModel):
    name: str                    # output column name, e.g. "meets_cetane_spec"
    field: str                   # which extracted field this target is derived from
    op: Literal[">=", "<=", "==", "!=", "between"]
    value: Any                   # float, or [lo, hi] when op == "between"
    description: str = ""


class DomainSpec(BaseModel):
    domain_name: str
    description: str = ""
    prompt_template_path: str    # path to the editable .txt prompt, relative to the YAML file
    fields: list[FieldSpec]
    binary_targets: list[BinaryTargetSpec] = Field(default_factory=list)
    multi_experiment: bool = False
    extract_claims: bool = False
    min_experiments: int = 1
    max_experiments: int = 20
    # Optional, purely advisory to normalize.py: the minimal set of field
    # names a shallow-ML structure-property-performance model would need as
    # non-null to be usable (typically a mix of structural/characterization
    # descriptors as features and at least one performance metric as
    # target). NOT enforced at extraction time -- these fields stay
    # optional/nullable like any other field, since requiring them would
    # fail whole papers over one missing characterization value (the same
    # reasoning oil_conversion_pct's own field description already gives).
    # Purely a downstream convenience: normalize.py uses this list to add an
    # `ml_core_features_complete` column so a consumer building a shallow-ML
    # dataset doesn't have to reverse-engineer, from 40+ columns, which
    # subset defines a usable row. Empty by default (no completeness column
    # added) so domains that don't declare this aren't affected.
    ml_core_fields: list[str] = Field(default_factory=list)

    def field_names(self) -> set[str]:
        return {f.name for f in self.fields}

    def paper_fields(self) -> list[FieldSpec]:
        return [f for f in self.fields if f.scope == "paper"]

    def experiment_fields(self) -> list[FieldSpec]:
        return [f for f in self.fields if f.scope == "experiment"]

    def field_reminders_block(self) -> str:
        """Render every emphasize: true field as a standalone, numbered
        prose instruction, generated from the SAME description text that
        also ends up in the JSON schema -- see FieldSpec.emphasize for why
        both copies exist and why this one is generated, not hand-written.
        Returns '' if no field in this domain sets emphasize: true, so a
        prompt template with a {field_reminders} placeholder degrades to
        an empty section rather than an error for domains that don't use
        this feature.
        """
        emphasized = [f for f in self.fields if f.emphasize]
        if not emphasized:
            return ""
        lines = [
            "IMPORTANT FIELD-SPECIFIC REMINDERS (generated from the schema "
            "below -- read carefully, these are common extraction-error "
            "sources, not optional detail):",
            "",
        ]
        for i, f in enumerate(emphasized, start=1):
            desc = " ".join(f.description.split())  # collapse YAML's folded newlines
            lines.append(f"{i}. {f.name}: {desc}")
        return "\n".join(lines)


def load_domain_spec(yaml_path: Path) -> DomainSpec:
    raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    spec = DomainSpec.model_validate(raw)

    field_names = spec.field_names()
    for bt in spec.binary_targets:
        if bt.field not in field_names:
            raise ValueError(
                f"Domain '{spec.domain_name}': binary target '{bt.name}' references "
                f"unknown field '{bt.field}'. Known fields: {sorted(field_names)}"
            )
        if bt.op == "between" and not (isinstance(bt.value, list) and len(bt.value) == 2):
            raise ValueError(
                f"Domain '{spec.domain_name}': binary target '{bt.name}' uses op='between' "
                "and must give value as a two-element list [low, high]."
            )
        if spec.multi_experiment:
            target_field = next(f for f in spec.fields if f.name == bt.field)
            if target_field.scope != "experiment":
                raise ValueError(
                    f"Domain '{spec.domain_name}': binary target '{bt.name}' references "
                    f"paper-scope field '{bt.field}'. Binary targets in multi_experiment "
                    "domains must reference experiment-scope fields, since they are "
                    "evaluated per experiment row in the flattened output."
                )

    if spec.multi_experiment and not spec.experiment_fields():
        raise ValueError(
            f"Domain '{spec.domain_name}': multi_experiment is true but no field has "
            "scope: experiment. At least one experiment-scope field is required, or "
            "there is nothing to repeat per experiment."
        )

    unknown_ml_core = [f for f in spec.ml_core_fields if f not in field_names]
    if unknown_ml_core:
        raise ValueError(
            f"Domain '{spec.domain_name}': ml_core_fields references unknown field(s) "
            f"{unknown_ml_core}. Known fields: {sorted(field_names)}"
        )

    return spec


def load_prompt_template(yaml_path: Path, spec: DomainSpec) -> str:
    prompt_path = (Path(yaml_path).parent / spec.prompt_template_path).resolve()
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Domain '{spec.domain_name}' points at prompt_template_path="
            f"'{spec.prompt_template_path}', resolved to {prompt_path}, which does not exist."
        )
    return prompt_path.read_text(encoding="utf-8")


def _fields_to_pydantic_kwargs(field_specs: list[FieldSpec]) -> dict[str, tuple[Any, Any]]:
    fields: dict[str, tuple[Any, Any]] = {}
    for f in field_specs:
        if f.enum:
            py_type = Literal[tuple(f.enum)]  # type: ignore[valid-type]
        else:
            py_type = _TYPE_MAP[f.type]
        default = ... if f.required else None
        annotated_type = py_type if f.required else Optional[py_type]
        fields[f.name] = (
            annotated_type,
            Field(default=default, description=f.description or None, max_length=f.max_length),
        )
    return fields


def _rescale_one(v: Any) -> Any:
    """Rescale a single value if it's a plausible percentage/Likert-style
    confidence score outside [0, 1]. Shared by the top-level scalar fields
    and the nested per-claim rescale below so the two paths can't drift
    apart the way the top-level-only version of this function did (see
    _confidence_rescale_validator's docstring: an observed paper loss --
    claims[].confidence=3, a 1-3/1-5 Likert misinterpretation by
    gemma3:12b -- was NOT caught because this rescaling previously only
    ever looked at the top-level dict keys, never into a nested list of
    claim objects).
    """
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 1.0:
        return min(v / 100.0, 1.0)
    return v


def _confidence_rescale_validator(cls, values: dict) -> dict:
    """Pre-validation rescaler for LLM-returned confidence scores.

    The generated JSON schema states `minimum: 0.0, maximum: 1.0` for every
    confidence field, but several models (observed: gemma3:12b, granite4.1)
    interpret "confidence" semantically as a percentage OR a small Likert
    scale (1-3, 1-5) and return values on that scale regardless, e.g.
    `"extraction_confidence": 95` or `"confidence": 3`. Pydantic then raises
    ValidationError on `le=1.0`, and because the model's underlying
    interpretation of the field doesn't change between retries, every one
    of the configured retries fails identically and the paper is lost
    outright -- this is not hypothetical, it is exactly what happened to
    paper 2dfd2478 in production (gemma3:12b returned claims[*].confidence
    = 3 on all three claims, on all three retries, before this fix).

    Covers TWO shapes, both needed:
      1. Top-level scalar fields named in _CONFIDENCE_FIELD_NAMES
         (extraction_confidence / experiment_confidence).
      2. Any nested `claims` list (present when spec.extract_claims is
         true) -- each ClaimRecord's `confidence` field is rescaled the
         same way. This is domain-independent: every multi_experiment
         domain with extract_claims=true gets this for free, not just
         saf_hefa, since claims/ClaimRecord are shared machinery (see
         build_pydantic_model), not something a domain YAML declares.

    A value that isn't a plausible percentage either (e.g. 150) is clamped
    to 1.0 rather than producing an out-of-range 1.5. Values already in
    [0, 1] pass through unchanged in both shapes.

    Applied identically to extraction_confidence (flat and multi_experiment
    paper-level) and experiment_confidence (multi_experiment experiment-
    level) via _CONFIDENCE_FIELD_NAMES, so the fix is not duplicated between
    the two schema shapes in build_pydantic_model.
    """
    if not isinstance(values, dict):
        return values

    for key in _CONFIDENCE_FIELD_NAMES:
        if key in values:
            values[key] = _rescale_one(values.get(key))

    claims = values.get("claims")
    if isinstance(claims, list):
        for claim in claims:
            if isinstance(claim, dict) and "confidence" in claim:
                claim["confidence"] = _rescale_one(claim.get("confidence"))

    return values


def _attach_confidence_rescaler(model_cls: type[BaseModel], model_name: str) -> type[BaseModel]:
    """create_model() doesn't accept validators as a direct kwarg, so the
    rescaling validator is attached by subclassing with the same name
    (transparent to every caller -- model_json_schema(), model_dump(), etc.
    all behave identically on the subclass)."""
    validator_fn = model_validator(mode="before")(_confidence_rescale_validator)
    return type(model_name, (model_cls,), {"_rescale_confidence": validator_fn})


def build_pydantic_model(spec: DomainSpec) -> type[BaseModel]:
    """Dynamically build the extraction model from a DomainSpec.

    Two shapes, both returned as a single top-level Pydantic model class so
    every other module (ollama_client, resolve, reader) only ever needs to
    know "this is a BaseModel subclass with a JSON schema" — same contract
    regardless of shape:

    multi_experiment=False (default): flat ExtractionFields model.
    extraction_confidence is added automatically — not something a domain
    author redeclares in YAML.

    multi_experiment=True: a PaperExtraction model containing the paper-
    scope fields inline, plus `experiments: list[ExperimentFields]`
    (bounded by spec.min_experiments/max_experiments so a degenerate
    extraction with 0 or 500 "experiments" fails validation instead of
    silently producing garbage) where each ExperimentFields combines the
    experiment-scope fields with an `evidence: EvidenceSpan` and its own
    `experiment_confidence`, plus `claims: list[ClaimRecord]` when
    spec.extract_claims is true.

    Both shapes get the confidence-rescale validator (see
    _confidence_rescale_validator) on whichever model actually carries a
    confidence field.
    """
    if not spec.multi_experiment:
        fields = _fields_to_pydantic_kwargs(spec.fields)
        fields["extraction_confidence"] = (
            float,
            Field(
                ge=0.0, le=1.0,
                description=(
                    "Your confidence (a decimal between 0.0 and 1.0, NOT a "
                    "percentage) that the values above were explicitly stated "
                    "in the source text and not inferred or guessed. Use 0.95 "
                    "for high confidence, 0.70 for moderate, 0.40 for low. "
                    "IMPORTANT: write 0.90, never 90."
                ),
            ),
        )
        model_name = "".join(p.capitalize() for p in spec.domain_name.split("_")) + "ExtractionFields"
        model_cls = create_model(model_name, **fields)  # type: ignore[call-overload]
        return _attach_confidence_rescaler(model_cls, model_name)

    # --- multi_experiment=True: nested paper -> experiments[] -> claims[] ---
    experiment_fields = _fields_to_pydantic_kwargs(spec.experiment_fields())
    experiment_fields["evidence"] = (
        EvidenceSpan,
        Field(default_factory=EvidenceSpan,
              description="Where in the source document this experiment's data came from."),
    )
    experiment_fields["experiment_confidence"] = (
        float,
        Field(
            ge=0.0, le=1.0,
            description=(
                "Your confidence (a decimal between 0.0 and 1.0, NOT a "
                "percentage) that THIS experiment's values were explicitly "
                "stated in the source text and not inferred or guessed. "
                "IMPORTANT: write 0.90, never 90."
            ),
        ),
    )
    experiment_model_name = "".join(p.capitalize() for p in spec.domain_name.split("_")) + "ExperimentFields"
    ExperimentFields = create_model(experiment_model_name, **experiment_fields)  # type: ignore[call-overload]
    ExperimentFields = _attach_confidence_rescaler(ExperimentFields, experiment_model_name)

    paper_fields = _fields_to_pydantic_kwargs(spec.paper_fields())
    paper_fields["experiments"] = (
        list[ExperimentFields],  # type: ignore[valid-type]
        Field(
            min_length=spec.min_experiments,
            max_length=spec.max_experiments,
            description="One entry per distinct catalyst/condition combination "
                         "reported in the paper, NOT one entry per paper. If the "
                         "paper tests 5 catalysts, this list must have 5 entries. "
                         "Never merge, average, or discard rows to fit one entry.",
        ),
    )
    if spec.extract_claims:
        paper_fields["claims"] = (
            list[ClaimRecord],
            Field(
                default_factory=list,
                description="Causal or comparative statements the paper makes "
                             "about how a factor affects an outcome, independent "
                             "of which specific experiment(s) support them.",
            ),
        )

    paper_model_name = "".join(p.capitalize() for p in spec.domain_name.split("_")) + "PaperExtraction"
    PaperExtraction = create_model(paper_model_name, **paper_fields)  # type: ignore[call-overload]
    # Paper-scope fields don't include extraction_confidence in multi_experiment
    # mode (see this function's docstring), but claims[].confidence lives HERE,
    # at the paper level, sibling to `experiments` -- so this model still needs
    # the rescale validator attached for that field to self-heal. Before this
    # was added, claims[].confidence had NO rescaling at all (unlike
    # experiment_confidence, which self-heals because ExperimentFields gets
    # its own copy of this validator and each experiment validates as an
    # independent nested model) -- this is what caused a real production
    # paper loss when a model returned a Likert-scale confidence on claims.
    PaperExtraction = _attach_confidence_rescaler(PaperExtraction, paper_model_name)
    # Stash the experiment model on the paper model so resolve.py/pipeline.py/
    # normalize.py can get at it without re-deriving it from the spec again.
    PaperExtraction.__experiment_model__ = ExperimentFields  # type: ignore[attr-defined]
    return PaperExtraction


def _apply_op(series: pd.Series, op: str, value: Any) -> pd.Series:
    if op == ">=":
        return series >= value
    if op == "<=":
        return series <= value
    if op == "==":
        return series == value
    if op == "!=":
        return series != value
    if op == "between":
        lo, hi = value
        return series.between(lo, hi)
    raise ValueError(f"Unsupported op: {op}")  # unreachable given DomainSpec validation


def build_binary_targets_generic(df: pd.DataFrame, spec: DomainSpec) -> pd.DataFrame:
    """Generic replacement for a hand-written build_binary_targets(). Produces
    nullable booleans: a row with a missing underlying value stays pd.NA, it
    never silently becomes False ('fails the spec') — same rule enforced
    once for every domain instead of once per domain module."""
    df = df.copy()
    for bt in spec.binary_targets:
        if bt.field not in df.columns:
            continue  # field wasn't extracted for this run; leave target absent
        result = _apply_op(df[bt.field], bt.op, bt.value).astype("boolean")
        result[df[bt.field].isna()] = pd.NA
        df[bt.name] = result
    return df


def field_docs(spec: DomainSpec) -> list[tuple[str, str, str]]:
    """(column, description, unit) triples — used by writer.py's data dictionary."""
    return [(f.name, f.description or f.name, f.unit) for f in spec.fields]
