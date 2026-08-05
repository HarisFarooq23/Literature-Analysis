"""
resolve.py — reconciles field-level disagreements between multiple
model outputs for the same paper. Pure functions, no I/O, fully unit-testable
without touching Ollama.

Whether a field is treated as categorical (exact-match / majority vote) or
numeric (tolerance / median) is read from the Pydantic model's own type
annotation, not from a hardcoded field-name list — this is what lets the
same resolution logic work for every domain (biodiesel, methanol synthesis,
MOF defect engineering, SAF/HEFA, ...) without modification.

A note on "why do so many fields disagree" (a recurring support question):
when a domain uses multi_experiment=True and a paper reports several
catalysts/conditions in one table, TWO DISTINCT problems can look identical
in the logs but have completely different fixes:

  1. Genuine per-field disagreement: primary and cross_check are describing
     THE SAME experiment but read a number differently (misread digit,
     different table column, OCR artefact). Fixing this is what
     numeric_agreement_tolerance and the tie-breaker escalation are for.

  2. Experiment misalignment: primary and cross_check are each internally
     self-consistent but are describing TWO DIFFERENT experiments -- e.g.
     primary extracted the Ni/Al2O3 row, cross_check extracted the
     NiMo/SAPO-11 row from the same table. Comparing them field-by-field
     then shows "disagreement" on every single field (BET area, conversion,
     temperature, everything) because they are, correctly, different
     numbers describing different catalysts. No tolerance value fixes this
     -- the fix is either (a) multi_experiment=True with correct experiment
     matching (this module's match_experiments), so each catalyst is
     compared against its own counterpart, not a stranger's, or (b), if
     already using multi_experiment=True, better experiment identity
     matching so wording differences ("NiMo" vs "Ni-Mo" vs "Ni/Mo") don't
     cause two DESCRIPTIONS of the same experiment to be treated as two
     different, unrelated experiments and paired by position with the
     wrong partner instead.

diagnose_disagreement_pattern() below turns this from a debugging exercise
into an explicit, logged signal: if most/all compared fields disagree for a
single paper, that is reported as a likely experiment-misalignment case
rather than N independent field-level failures, so a person reading the log
gets the right hypothesis immediately instead of "surface area disagree,
temperature disagree, conversion disagree, ..." with no unifying
explanation.
"""

from __future__ import annotations

import math
import statistics
import typing
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment

_EXCLUDED_FROM_AGREEMENT = {
    "extraction_confidence",
    "experiment_confidence",  # per-experiment metadata, not a scientific measurement
    "evidence",               # provenance metadata; always keep primary's span
}

# Minimum absolute difference required to flag a numeric disagreement, on
# top of the relative (percentage) tolerance. Without this floor, two
# small-magnitude readings that are scientifically identical can fail a
# purely relative check: e.g. a pore diameter of 0.05 nm vs 0.06 nm is a
# 20% relative difference but a 0.01 nm absolute difference -- almost
# certainly the same reading with rounding noise, not a real disagreement.
# Set to 0.0 to disable and fall back to pure relative tolerance.
_NUMERIC_ABSOLUTE_FLOOR = 1e-6


def _unwrap_optional(annotation: Any) -> Any:
    if typing.get_origin(annotation) is typing.Union:
        non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def _is_categorical_field(model_cls: type[BaseModel], field_name: str) -> bool:
    """str and Literal[...] fields are categorical; float/int fields are
    numeric. bool is treated as categorical (exact match, no 'median')."""
    annotation = _unwrap_optional(model_cls.model_fields[field_name].annotation)
    if typing.get_origin(annotation) is typing.Literal:
        return True
    return annotation in (str, bool)


@dataclass
class FieldDisagreement:
    field_name: str
    values: dict[str, Any]  # model_name -> value


@dataclass
class ResolutionResult:
    resolved: dict[str, Any]
    disagreements: list[FieldDisagreement] = field(default_factory=list)
    unresolved_fields: list[str] = field(default_factory=list)

    @property
    def agreement_status(self) -> str:
        if not self.disagreements:
            return "full_agreement"
        if self.unresolved_fields:
            return "unresolved_conflict"
        return "resolved_after_escalation"


def _numeric_agree(a: float, b: float, tolerance: float) -> bool:
    """Two numbers 'agree' if they're within `tolerance` relative
    difference OR within _NUMERIC_ABSOLUTE_FLOOR absolute difference,
    whichever is more permissive. The absolute floor guards against
    relative-tolerance blowing up for near-zero values (see module
    docstring / _NUMERIC_ABSOLUTE_FLOOR)."""
    if abs(a - b) <= _NUMERIC_ABSOLUTE_FLOOR:
        return True
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom <= tolerance


def _canonicalized_str_equal(field_name: str, pv: str, cv: str) -> bool:
    """String equality check used for categorical fields, with
    canonicalization applied to metal/support-like field names so wording
    variants of the same entity ("NiMo" vs "Ni/Mo", "HZSM-5" vs "ZSM-5")
    are treated as equal. Without this, _experiment_identity_key can
    correctly match two records as 'the same experiment' while compare_two
    still flags the very field that made them match (active_metal,
    support_material) as disagreeing -- fixing the experiment-pairing bug
    only to immediately reintroduce a field-level false positive on the
    identity field itself.
    """
    if str(pv).strip().lower() == str(cv).strip().lower():
        return True
    if "metal" in field_name:
        from .canonicalize import canonicalize_metal
        _, p_canon = canonicalize_metal(pv)
        _, c_canon = canonicalize_metal(cv)
        return p_canon is not None and p_canon == c_canon
    if "support" in field_name:
        from .canonicalize import canonicalize_support
        _, p_canon = canonicalize_support(pv)
        _, c_canon = canonicalize_support(cv)
        return p_canon is not None and p_canon == c_canon
    return False


def compare_two(
    primary: BaseModel,
    cross_check: BaseModel,
    numeric_tolerance: float,
) -> list[str]:
    """Return the list of field names where primary and cross_check disagree."""
    disagreeing: list[str] = []
    p, c = primary.model_dump(), cross_check.model_dump()
    model_cls = type(primary)

    for field_name in p:
        if field_name in _EXCLUDED_FROM_AGREEMENT:
            continue
        pv, cv = p[field_name], c[field_name]

        if pv is None and cv is None:
            continue
        if pv is None or cv is None:
            disagreeing.append(field_name)  # one found a value, other didn't — ambiguous
            continue

        if _is_categorical_field(model_cls, field_name):
            if not _canonicalized_str_equal(field_name, pv, cv):
                disagreeing.append(field_name)
        else:
            try:
                if not _numeric_agree(float(pv), float(cv), numeric_tolerance):
                    disagreeing.append(field_name)
            except (TypeError, ValueError):
                if pv != cv:
                    disagreeing.append(field_name)

    return disagreeing


def diagnose_disagreement_pattern(
    disagreeing_fields: list[str],
    total_comparable_fields: int,
    threshold: float = 0.6,
) -> Optional[str]:
    """Classify a set of disagreeing fields into an actionable diagnostic
    message, or None if the disagreement looks like ordinary field-level
    noise that doesn't need special attention.

    If disagreeing_fields covers >= `threshold` fraction of all comparable
    fields on the record, this is very unlikely to be N independent
    misreadings -- it is far more consistent with primary and cross_check
    describing two DIFFERENT experiments from the same paper (see module
    docstring). Surfacing this explicitly saves someone re-discovering the
    same root cause from raw field-disagreement logs every time it recurs.
    """
    if total_comparable_fields == 0:
        return None
    disagreement_fraction = len(disagreeing_fields) / total_comparable_fields
    if disagreement_fraction >= threshold:
        return (
            f"{len(disagreeing_fields)}/{total_comparable_fields} fields disagree "
            f"({disagreement_fraction:.0%}) — this pattern usually means the two "
            "models extracted DIFFERENT experiments/catalysts from a multi-catalyst "
            "paper, not that they disagree about the same measurement. If this "
            "domain is not already multi_experiment: true, that is very likely the "
            "fix (see domains/<name>.yaml and the module docstring in resolve.py). "
            "If it IS already multi_experiment: true, this indicates the identity-"
            "key match between primary/cross_check experiments likely failed and "
            "fell back to position-pairing two unrelated experiments — check "
            "canonicalize.py coverage for the catalyst-naming variants in this "
            "paper."
        )
    return None


def _numeric_field_names(model_cls: type[BaseModel]) -> list[str]:
    """Every numeric (non-categorical, non-excluded) field on an experiment
    model -- used to disambiguate identity-key collisions by numeric
    similarity rather than list position. Domain-agnostic: reads the field
    list off the Pydantic model itself, so it works identically whether the
    differentiating field is metal_loading_wt_pct (SAF), reaction_temp_C
    (any domain), or something a future domain adds -- there is nothing to
    hardcode per domain."""
    return [
        name for name in model_cls.model_fields
        if name not in _EXCLUDED_FROM_AGREEMENT
        and not _is_categorical_field(model_cls, name)
    ]


def _numeric_distance(a: BaseModel, b: BaseModel, numeric_fields: list[str]) -> float:
    """Mean relative distance between two experiment records across every
    numeric field both sides actually reported a value for. Used only to
    rank candidates WITHIN an identity-key collision group (see
    _match_within_collision_group) -- it is deliberately not used as a
    primary matching signal on its own, since two genuinely unrelated
    experiments could coincidentally have similar numeric values; it is
    only trustworthy once the qualitative identity (metal, support) is
    already known to match.

    Returns float('inf') if there is no shared numeric field to compare
    (in which case the caller falls back to original list order for that
    pair, same as the pre-existing behaviour for genuinely incomparable
    records).
    """
    a_dump, b_dump = a.model_dump(), b.model_dump()
    diffs = []
    for name in numeric_fields:
        av, bv = a_dump.get(name), b_dump.get(name)
        if av is None or bv is None:
            continue
        try:
            av, bv = float(av), float(bv)
        except (TypeError, ValueError):
            continue
        denom = max(abs(av), abs(bv), 1e-9)
        diffs.append(abs(av - bv) / denom)
    if not diffs:
        return float("inf")
    return sum(diffs) / len(diffs)


def _match_within_collision_group(
    primary_group: list[tuple[int, BaseModel]],
    cross_group: list[tuple[int, BaseModel]],
    numeric_fields: list[str],
) -> list[tuple[int, int]]:
    """Optimal-assignment pairing within a single identity-key collision
    group (i.e. every record here already agrees on catalyst identity --
    metal, support -- and only differs on some numeric axis the identity
    key deliberately excludes, e.g. multiple loading levels of the same
    NiMo/Al2O3 catalyst). Solved exactly via the Hungarian algorithm
    (scipy.optimize.linear_sum_assignment), which minimizes the TOTAL
    assignment cost across the whole group rather than greedily locking in
    the single closest pair first. A purely greedy nearest-neighbour
    approach can produce a strictly worse overall assignment: e.g. primary
    loadings [10, 12] vs cross_check [11, 9] -- greedy pairs (10,11) first
    (smallest single distance) and is then forced into (12,9); the globally
    optimal assignment is (10,9)+(12,11), which has a smaller SUM of
    distances. Collision groups are small in practice (a paper rarely
    reports more than a handful of variants of the same catalyst), so the
    O(n^3) worst case of the exact solver is irrelevant here -- there is no
    accuracy/performance tradeoff being made by using it.

    If there is no numeric field to disambiguate with at all, every
    distance is equal (see _numeric_distance's inf-for-no-fields case,
    capped below at a large finite constant so the solver has a valid cost
    matrix), so the solver's output falls back to arbitrary-but-stable
    pairing -- the same graceful degradation the previous greedy
    implementation provided.
    """
    if not primary_group or not cross_group:
        return []

    n_p, n_c = len(primary_group), len(cross_group)
    cost = np.full((n_p, n_c), fill_value=1e18, dtype=float)
    for i, (_, p_rec) in enumerate(primary_group):
        for j, (_, c_rec) in enumerate(cross_group):
            dist = _numeric_distance(p_rec, c_rec, numeric_fields)
            cost[i, j] = dist if math.isfinite(dist) else 1e18

    row_idx, col_idx = linear_sum_assignment(cost)
    return [(primary_group[i][0], cross_group[j][0]) for i, j in zip(row_idx, col_idx)]


def _experiment_identity_key(record: BaseModel) -> str:
    """A loose identity fingerprint for one experiment record, used to match
    the 'same' experiment across two models' extractions of the same paper
    before comparing it field-by-field. Built from every string-valued
    field on the experiment (catalyst identity, support, reactor type, ...)
    EXCEPT evidence/confidence metadata.

    Canonicalization is applied to metal/support-like fields (via
    canonicalize.py) before building the key, specifically so that wording
    differences between two models describing the SAME catalyst -- "NiMo"
    vs "Ni-Mo" vs "Ni/Mo" vs "Nickel Molybdenum" -- collapse to the same
    key instead of being treated as different experiments. Before this,
    such wording differences caused the two experiments to fall through to
    position-based pairing, which (if the two models also happened to list
    experiments in a different order) paired the WRONG experiments together
    and produced disagreement on every field -- the exact "surface area
    disagree" symptom this module's docstring describes. Any field that
    canonicalize.py doesn't recognise is left as its normalized-lowercase
    original text, same as before.

    This is a fingerprint, not a guarantee: two genuinely different
    experiments that happen to share every string field (e.g. same
    catalyst tested at two different temperatures, if temperature is the
    only differing field and it's numeric) will collide and get paired by
    fallback position instead. That's an acceptable approximation, not a
    silent data-loss bug -- the field-level comparison after matching still
    catches a real conflict if the paired records disagree on the numeric
    fields, and diagnose_disagreement_pattern() flags the resulting high-
    disagreement-fraction case for investigation either way.
    """
    from .canonicalize import canonicalize_metal, canonicalize_support

    d = record.model_dump()
    parts = []
    for k in sorted(d.keys()):
        if k in ("evidence", "experiment_confidence"):
            continue
        v = d[k]
        if not isinstance(v, str):
            continue
        normalized = v.strip().lower()
        if "metal" in k:
            _, canon = canonicalize_metal(v)
            normalized = (canon or normalized).lower()
        elif "support" in k:
            _, canon = canonicalize_support(v)
            normalized = (canon or normalized).lower()
        parts.append(normalized)
    return "|".join(parts)


@dataclass
class ExperimentMatch:
    primary: Optional[BaseModel]
    cross_check: Optional[BaseModel]
    match_method: str  # "identity_key" | "numeric_similarity" | "position" | "primary_only" | "cross_check_only"


def match_experiments(
    primary_experiments: list[BaseModel],
    cross_check_experiments: list[BaseModel],
) -> list[ExperimentMatch]:
    """Pair up experiments from two models' extractions of the same paper.

    Three-tier strategy, applied in order:

    1. Unique identity-key matches — a primary experiment and a cross_check
       experiment whose canonicalized identity key (catalyst + support) is
       unique on both sides. Unambiguous.

    2. Identity-key COLLISION groups — multiple experiments on one or both
       sides share the same identity key. This happens legitimately when a
       paper tests several numeric variants of the same nominal catalyst
       (e.g. 5 wt% vs 10 wt% NiMo/Al2O3 -- same metal, same support,
       different loading, and loading is deliberately excluded from the
       identity key since it's numeric, not a naming-variant problem).
       Within each such group, candidates are paired by nearest numeric
       similarity across every shared numeric field (see
       _match_within_collision_group), NOT by raw list position -- list
       position silently mispairs two variants whenever the two models
       happen to list them in a different order, which is a realistic and
       previously-undetected failure mode for this exact scenario.

    3. Anything still unmatched after (1) and (2) falls back to plain
       position pairing (unrelated experiments where neither identity nor
       numeric similarity is available to disambiguate), and finally
       anything left over on only one side is recorded as
       primary_only/cross_check_only -- a real disagreement signal ('one
       model found an experiment the other one missed'), not silently
       merged or silently dropped.
    """
    primary_keys = [(i, _experiment_identity_key(e)) for i, e in enumerate(primary_experiments)]
    cross_keys = [(i, _experiment_identity_key(e)) for i, e in enumerate(cross_check_experiments)]

    p_key_counts = Counter(k for _, k in primary_keys)
    c_key_counts = Counter(k for _, k in cross_keys)

    matches: list[ExperimentMatch] = []
    matched_primary_idx: set[int] = set()
    matched_cross_idx: set[int] = set()

    # --- Tier 1: unique identity-key matches ---
    for p_idx, p_key in primary_keys:
        if not p_key or p_key_counts[p_key] != 1 or c_key_counts.get(p_key, 0) != 1:
            continue
        for c_idx, c_key in cross_keys:
            if c_idx in matched_cross_idx:
                continue
            if c_key == p_key:
                matches.append(ExperimentMatch(
                    primary=primary_experiments[p_idx],
                    cross_check=cross_check_experiments[c_idx],
                    match_method="identity_key",
                ))
                matched_primary_idx.add(p_idx)
                matched_cross_idx.add(c_idx)
                break

    # --- Tier 2: collision groups (same non-empty key, count > 1 on either side) ---
    numeric_fields = None
    collision_keys = {
        k for k in set(p_key_counts) & set(c_key_counts)
        if k and (p_key_counts[k] > 1 or c_key_counts[k] > 1)
    }
    for key in collision_keys:
        p_group = [(i, primary_experiments[i]) for i, k in primary_keys
                   if k == key and i not in matched_primary_idx]
        c_group = [(i, cross_check_experiments[i]) for i, k in cross_keys
                   if k == key and i not in matched_cross_idx]
        if not p_group or not c_group:
            continue
        if numeric_fields is None:
            numeric_fields = _numeric_field_names(type(primary_experiments[0]))
        for p_idx, c_idx in _match_within_collision_group(p_group, c_group, numeric_fields):
            matches.append(ExperimentMatch(
                primary=primary_experiments[p_idx],
                cross_check=cross_check_experiments[c_idx],
                match_method="numeric_similarity",
            ))
            matched_primary_idx.add(p_idx)
            matched_cross_idx.add(c_idx)

    # --- Tier 3: remaining leftovers -> position pairing, then singles ---
    leftover_primary = [i for i, _ in primary_keys if i not in matched_primary_idx]
    leftover_cross = [i for i, _ in cross_keys if i not in matched_cross_idx]

    for p_idx, c_idx in zip(leftover_primary, leftover_cross):
        matches.append(ExperimentMatch(
            primary=primary_experiments[p_idx],
            cross_check=cross_check_experiments[c_idx],
            match_method="position",
        ))
        matched_primary_idx.add(p_idx)
        matched_cross_idx.add(c_idx)

    for p_idx, _ in primary_keys:
        if p_idx not in matched_primary_idx:
            matches.append(ExperimentMatch(
                primary=primary_experiments[p_idx], cross_check=None,
                match_method="primary_only",
            ))
    for c_idx, _ in cross_keys:
        if c_idx not in matched_cross_idx:
            matches.append(ExperimentMatch(
                primary=None, cross_check=cross_check_experiments[c_idx],
                match_method="cross_check_only",
            ))

    return matches


@dataclass
class ExperimentResolution:
    resolved: Optional[dict]            # None only if both sides were None (shouldn't happen)
    experiment_match_status: str        # "full_agreement" | "resolved_after_escalation" | "unresolved_conflict" | "single_model_only"
    unresolved_fields: list[str] = field(default_factory=list)
    match_method: str = "identity_key"
    diagnostic: Optional[str] = None    # set when diagnose_disagreement_pattern fires


def resolve_experiment_lists(
    primary_experiments: list[BaseModel],
    cross_check_experiments: list[BaseModel],
    tie_breaker_experiments: Optional[list[BaseModel]],
    experiment_model_cls: type[BaseModel],
    numeric_tolerance: float,
) -> list[ExperimentResolution]:
    """Top-level entry point: match primary vs cross_check experiments, then
    for each matched pair, reuse the same field-level compare/resolve logic
    already used for flat records. tie_breaker_experiments (if given,
    e.g. because the paper needed escalation) is matched independently
    against whichever side has more experiments, on the same identity-key
    + position fallback, and used to break ties exactly like a flat-record
    tie-breaker does.

    single_model_only experiments (one model found an experiment the other
    didn't) are kept in the output WITH their full data -- discarding them
    would be worse than flagging them, since a real experiment a second
    model simply missed is still real data -- but flagged so a downstream
    consumer can filter them out if they want only experiments both models
    agree exist.

    Each result carries a `diagnostic` string (or None) from
    diagnose_disagreement_pattern, so pipeline.py can surface it directly
    in the log at the point of escalation rather than a caller having to
    infer the same conclusion from a bare disagreeing-fields list.
    """
    matches = match_experiments(primary_experiments, cross_check_experiments)
    tie_breaker_experiments = tie_breaker_experiments or []

    total_comparable_fields = len(
        [f for f in experiment_model_cls.model_fields if f not in _EXCLUDED_FROM_AGREEMENT]
    )

    results: list[ExperimentResolution] = []
    for m in matches:
        if m.primary is not None and m.cross_check is not None:
            disagreeing = compare_two(m.primary, m.cross_check, numeric_tolerance)
            diagnostic = diagnose_disagreement_pattern(disagreeing, total_comparable_fields)

            if not disagreeing:
                results.append(ExperimentResolution(
                    resolved=m.primary.model_dump(),
                    experiment_match_status="full_agreement",
                    match_method=m.match_method,
                ))
                continue

            tb_match = None
            if tie_breaker_experiments:
                tb_candidates = match_experiments([m.primary], tie_breaker_experiments)
                identity_or_position = [c for c in tb_candidates if c.match_method in ("identity_key", "position")]
                if identity_or_position:
                    tb_match = identity_or_position[0].cross_check

            resolution = resolve_with_tie_breaker(
                m.primary, m.cross_check, tb_match, disagreeing, numeric_tolerance,
            )
            results.append(ExperimentResolution(
                resolved=resolution.resolved,
                experiment_match_status=resolution.agreement_status,
                unresolved_fields=resolution.unresolved_fields,
                match_method=m.match_method,
                diagnostic=diagnostic,
            ))
        else:
            only = m.primary if m.primary is not None else m.cross_check
            results.append(ExperimentResolution(
                resolved=only.model_dump(),
                experiment_match_status="single_model_only",
                match_method=m.match_method,
            ))

    return results


def resolve_with_tie_breaker(
    primary: BaseModel,
    cross_check: BaseModel,
    tie_breaker: Optional[BaseModel],
    disagreeing_fields: list[str],
    numeric_tolerance: float,
) -> ResolutionResult:
    """
    Build the final resolved record.

    Fields with no disagreement: take primary's value (arbitrary but
    consistent choice between two agreeing models).
    Fields with disagreement: if tie_breaker is available, resolve by
    majority vote (categorical) or median of non-null values (numeric);
    if no majority/insufficient data, leave as unresolved (value=None,
    flagged in unresolved_fields for downstream filtering).
    """
    p, c = primary.model_dump(), cross_check.model_dump()
    t = tie_breaker.model_dump() if tie_breaker is not None else {}
    model_cls = type(primary)

    resolved: dict[str, Any] = dict(p)  # start from primary, override below
    disagreements: list[FieldDisagreement] = []
    unresolved: list[str] = []

    for field_name in disagreeing_fields:
        candidates = {"primary": p[field_name], "cross_check": c[field_name]}
        if tie_breaker is not None:
            candidates["tie_breaker"] = t.get(field_name)
        disagreements.append(FieldDisagreement(field_name, dict(candidates)))

        non_null = {k: v for k, v in candidates.items() if v is not None}

        if not non_null:
            resolved[field_name] = None
            continue

        if _is_categorical_field(model_cls, field_name):
            counts: dict[Any, int] = {}
            for v in non_null.values():
                key = str(v).strip().lower()
                counts[key] = counts.get(key, 0) + 1
            top_value, top_count = max(counts.items(), key=lambda kv: kv[1])
            if top_count >= 2:
                # recover original-cased value for whichever model produced the winner
                resolved[field_name] = next(
                    v for v in non_null.values() if str(v).strip().lower() == top_value
                )
            else:
                resolved[field_name] = None
                unresolved.append(field_name)
        else:
            try:
                nums = [float(v) for v in non_null.values()]
                resolved[field_name] = statistics.median(nums)
            except (TypeError, ValueError):
                resolved[field_name] = None
                unresolved.append(field_name)

    return ResolutionResult(resolved=resolved, disagreements=disagreements, unresolved_fields=unresolved)
