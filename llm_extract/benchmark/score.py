"""
benchmark/score.py — scores extracted_records.jsonl (pipeline output)
against a human-completed gold-standard file, producing the per-field
precision/recall/F1 and numeric-accuracy table a methods section needs to
report ("Table 1: LLM extraction benchmark (F1 per entity)").

CENTRAL DESIGN DECISION: experiment alignment between gold and predicted
reuses resolve.match_experiments() -- the EXACT SAME function that aligns
primary-model vs cross-check-model experiments in production. This is
deliberate, not a shortcut: if scoring used a separate, hand-rolled
alignment algorithm, a bug in that second algorithm could inflate or
deflate the reported F1 without the production matching logic being wrong
at all (or vice versa) -- the benchmark number would then not actually
describe the pipeline's real-world matching behaviour. Reusing the same
function means "how well experiments were matched" is measured by the
identical logic that determines it in a live run, not by a parallel
implementation that could drift out of sync as resolve.py evolves.

METRICS PRODUCED:
  - experiment_count_accuracy: fraction of papers where the pipeline
    extracted the correct NUMBER of experiments (the multi_experiment-
    specific failure mode this whole architecture exists to prevent --
    see resolve.py's module docstring). This is reported separately from
    field-level F1 because a paper that extracts the wrong number of
    experiments can still show misleadingly high field-level accuracy on
    whichever experiments happened to align.
  - per_field: for each field defined in the domain schema --
      categorical fields -> precision, recall, F1 (treating "gold has a
        non-null value and predicted matches it" as a true positive,
        "predicted has a value gold doesn't" as a false positive, "gold
        has a value predicted is missing/wrong" as a false negative --
        standard NER-style entity extraction scoring)
      numeric fields -> MAE, RMSE, and % within tolerance (same tolerance
        semantics as extraction.numeric_agreement_tolerance, so the
        reported accuracy is directly comparable to the production
        agreement threshold)
  - unmatched_gold_experiments / unmatched_predicted_experiments: raw
    counts (not folded into the per-field metrics) so a reader can see
    directly how much of the gold standard the pipeline missed entirely,
    and how much of the pipeline's output doesn't correspond to anything
    an annotator found in the paper (a hallucinated experiment).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel

from ..resolve import _canonicalized_str_equal, _is_categorical_field, match_experiments
from .annotation_template import _EXCLUDED_FROM_GOLD_COMPARISON
from ..domain_registry import SchemaBundle, get_schema


@dataclass
class FieldScore:
    field_name: str
    field_type: str  # "categorical" | "numeric"
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0  # both gold and predicted correctly null
    # numeric-only:
    absolute_errors: list[float] = field(default_factory=list)
    n_within_tolerance: int = 0
    n_numeric_comparable: int = 0

    @property
    def precision(self) -> Optional[float]:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else None

    @property
    def recall(self) -> Optional[float]:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def mae(self) -> Optional[float]:
        return statistics.mean(self.absolute_errors) if self.absolute_errors else None

    @property
    def rmse(self) -> Optional[float]:
        if not self.absolute_errors:
            return None
        return statistics.mean(e ** 2 for e in self.absolute_errors) ** 0.5

    @property
    def pct_within_tolerance(self) -> Optional[float]:
        return (self.n_within_tolerance / self.n_numeric_comparable
                if self.n_numeric_comparable else None)

    def to_row(self) -> dict:
        if self.field_type == "categorical":
            return {
                "field": self.field_name, "type": "categorical",
                "n_gold_present": self.true_positives + self.false_negatives,
                "precision": self.precision, "recall": self.recall, "f1": self.f1,
            }
        return {
            "field": self.field_name, "type": "numeric",
            "n_gold_present": self.n_numeric_comparable + self.false_negatives,
            "mae": self.mae, "rmse": self.rmse,
            "pct_within_tolerance": self.pct_within_tolerance,
        }


@dataclass
class BenchmarkResult:
    n_gold_papers: int
    n_matched_papers: int
    n_experiments_gold: int
    n_experiments_predicted: int
    experiment_count_exact_match_fraction: float
    unmatched_gold_experiments: int
    unmatched_predicted_experiments: int
    field_scores: dict[str, FieldScore]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([fs.to_row() for fs in self.field_scores.values()])

    def summary(self) -> str:
        df = self.to_dataframe()
        lines = [
            f"Gold-standard papers: {self.n_gold_papers} "
            f"(matched to predictions: {self.n_matched_papers})",
            f"Experiments — gold: {self.n_experiments_gold}, "
            f"predicted: {self.n_experiments_predicted}",
            f"Per-paper experiment-count exact match: "
            f"{self.experiment_count_exact_match_fraction:.1%}",
            f"Unmatched gold experiments (missed by pipeline): "
            f"{self.unmatched_gold_experiments}",
            f"Unmatched predicted experiments (hallucinated / extra): "
            f"{self.unmatched_predicted_experiments}",
            "",
            df.to_string(index=False),
        ]
        return "\n".join(lines)


def _load_jsonl(path: Path) -> dict[str, dict]:
    """Load a JSONL file keyed by paper_id."""
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["paper_id"]] = rec
    return records


def _build_experiment_instance(exp_dict: dict, experiment_model_cls: type[BaseModel]) -> BaseModel:
    """Build a Pydantic instance from a gold or predicted experiment dict,
    tolerating extra scoring-only keys (experiment_match_status,
    match_method, unresolved_fields) that predicted records carry but the
    schema itself doesn't define.

    experiment_confidence is a required field on the schema (the LLM must
    self-report it), but a human gold annotation has no equivalent concept
    -- an annotator's reading of the paper isn't "70% confident", it's
    either what the paper says or it's left null. Rather than force gold-
    standard JSONL files to carry a meaningless confidence value just to
    satisfy the same Pydantic model the LLM output uses, this injects a
    placeholder (1.0) when the key is absent, which only ever affects
    gold-side records (predicted records always have a real value from the
    LLM) and is excluded from field-level scoring via
    resolve._EXCLUDED_FROM_AGREEMENT / this module's own field_scores dict,
    which is built from schema.spec.experiment_fields() -- the YAML-
    declared fields only, never this auto-added metadata field.
    """
    clean = {k: v for k, v in exp_dict.items() if k in experiment_model_cls.model_fields}
    clean.setdefault("experiment_confidence", 1.0)
    return experiment_model_cls.model_validate(clean)


def score(
    extracted_records_path: Path,
    gold_standard_path: Path,
    schema_name: str,
    domains_dir: Path,
    numeric_tolerance: float = 0.10,
) -> BenchmarkResult:
    """Score pipeline output against a completed gold-standard file.

    Both files must be JSONL with the paper-level shape produced by
    pipeline.py / annotation_template.py (paper_id, fields, experiments,
    ...). Only papers present in BOTH files are scored -- a gold paper the
    pipeline hasn't processed yet, or a predicted paper with no gold
    annotation, is silently excluded from the metrics (not counted as a
    failure), since the benchmark should describe accuracy on the
    annotated subset, not penalize incomplete overlap between the two
    files.
    """
    schema: SchemaBundle = get_schema(schema_name, domains_dir)
    if not schema.multi_experiment:
        raise ValueError(
            f"score() currently supports multi_experiment domains only; "
            f"'{schema_name}' has multi_experiment=False. Flat-domain "
            "scoring would need a different (simpler) comparison path -- "
            "not yet implemented since every current production domain "
            "with a gold-standard benchmark in progress is multi_experiment."
        )
    ExperimentModel = schema.ExperimentFields

    gold_records = _load_jsonl(gold_standard_path)
    predicted_records = _load_jsonl(extracted_records_path)

    paper_ids = sorted(set(gold_records) & set(predicted_records))
    if not paper_ids:
        raise ValueError(
            f"No paper_id appears in both {gold_standard_path} and "
            f"{extracted_records_path} -- nothing to score. Check both "
            "files were built from the same paper corpus."
        )

    field_scores: dict[str, FieldScore] = {
        f.name: FieldScore(
            field_name=f.name,
            field_type="categorical" if _is_categorical_field(ExperimentModel, f.name) else "numeric",
        )
        for f in schema.spec.experiment_fields()
    }

    n_experiments_gold = n_experiments_predicted = 0
    n_exact_count_match = 0
    unmatched_gold = unmatched_predicted = 0

    for paper_id in paper_ids:
        gold_paper = gold_records[paper_id]
        predicted_paper = predicted_records[paper_id]

        gold_exp_dicts = gold_paper.get("experiments", [])
        predicted_exp_dicts = predicted_paper.get("experiments", [])

        # Gold rows an annotator left entirely blank (never filled in --
        # the template's default n_experiment_slots) don't count as real
        # gold experiments; only score rows with at least one non-null
        # field.
        gold_exp_dicts = [
            e for e in gold_exp_dicts
            if any(v is not None for k, v in e.items() if k not in _EXCLUDED_FROM_GOLD_COMPARISON)
        ]

        n_experiments_gold += len(gold_exp_dicts)
        n_experiments_predicted += len(predicted_exp_dicts)
        if len(gold_exp_dicts) == len(predicted_exp_dicts):
            n_exact_count_match += 1

        gold_experiments = [_build_experiment_instance(e, ExperimentModel) for e in gold_exp_dicts]
        predicted_experiments = [_build_experiment_instance(e, ExperimentModel) for e in predicted_exp_dicts]

        matches = match_experiments(gold_experiments, predicted_experiments)

        for m in matches:
            if m.primary is None:
                unmatched_predicted += 1
                continue
            if m.cross_check is None:
                unmatched_gold += 1
                continue

            gold_exp, pred_exp = m.primary, m.cross_check
            gold_dump, pred_dump = gold_exp.model_dump(), pred_exp.model_dump()

            for field_name, fs in field_scores.items():
                gv, pv = gold_dump.get(field_name), pred_dump.get(field_name)

                if fs.field_type == "categorical":
                    if gv is None and pv is None:
                        fs.true_negatives += 1
                    elif gv is not None and pv is None:
                        fs.false_negatives += 1
                    elif gv is None and pv is not None:
                        fs.false_positives += 1
                    else:
                        if _canonicalized_str_equal(field_name, str(gv), str(pv)):
                            fs.true_positives += 1
                        else:
                            fs.false_positives += 1
                            fs.false_negatives += 1
                else:
                    if gv is None and pv is None:
                        fs.true_negatives += 1
                    elif gv is not None and pv is None:
                        fs.false_negatives += 1
                    elif gv is None and pv is not None:
                        fs.false_positives += 1
                    else:
                        try:
                            gv_f, pv_f = float(gv), float(pv)
                        except (TypeError, ValueError):
                            fs.false_positives += 1
                            fs.false_negatives += 1
                            continue
                        fs.n_numeric_comparable += 1
                        fs.absolute_errors.append(abs(gv_f - pv_f))
                        denom = max(abs(gv_f), abs(pv_f), 1e-9)
                        if abs(gv_f - pv_f) / denom <= numeric_tolerance:
                            fs.n_within_tolerance += 1

    return BenchmarkResult(
        n_gold_papers=len(gold_records),
        n_matched_papers=len(paper_ids),
        n_experiments_gold=n_experiments_gold,
        n_experiments_predicted=n_experiments_predicted,
        experiment_count_exact_match_fraction=(
            n_exact_count_match / len(paper_ids) if paper_ids else 0.0
        ),
        unmatched_gold_experiments=unmatched_gold,
        unmatched_predicted_experiments=unmatched_predicted,
        field_scores=field_scores,
    )
