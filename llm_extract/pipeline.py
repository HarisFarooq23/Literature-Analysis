"""
pipeline.py — orchestrates the full extraction run:

    for each eligible paper:
        [optional] describe figures with a vision model, append to context
        call primary model       -> ExtractionFields | failure
        call cross_check model   -> ExtractionFields | failure
        compare field-by-field (flat domains) or experiment-list-by-
            experiment-list (multi_experiment domains, see resolve.py)
        if disagreement: call tie_breaker model, resolve
        append one JSON line to extracted_records.jsonl

extracted_records.jsonl is both the audit trail and the resumability
checkpoint: on restart, paper_ids already present in it are skipped.

Two record shapes are written, matching domain_spec.DomainSpec.multi_experiment:

  flat domain   : {..., "fields": {<flat extracted record>}}
  multi_experiment domain:
                  {..., "fields": {<paper-scope fields>},
                        "experiments": [{<experiment-scope fields>,
                                          "experiment_match_status": ...,
                                          "unresolved_fields": [...]}, ...],
                        "claims": [{"subject":..., "relation":..., ...}, ...]}

normalize.py / writer.py branch on schema.multi_experiment the same way to
turn either shape into the final flattened dataset (one row per experiment
for multi_experiment domains).

DIAGNOSTIC LOGGING: when a paper shows disagreement on most/all comparable
fields at once (e.g. "surface_area", "conversion", "temperature" all flagged
together), resolve.diagnose_disagreement_pattern() is consulted and, if it
fires, its explanation is logged directly alongside the escalation message.
This turns what would otherwise look like N unrelated field-level noise
events into one clear, actionable diagnosis (almost always: the domain
needs multi_experiment: true, or its experiment-matching needs better
canonicalization coverage for this paper's catalyst naming) -- see
resolve.py's module docstring for the full explanation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from .config import AppConfig
from .domain_registry import SchemaBundle, get_schema
from .ollama_client import CallResult, OllamaExtractor
from .reader import PaperRecord, iter_papers
from .resolve import (
    compare_two,
    diagnose_disagreement_pattern,
    resolve_experiment_lists,
    resolve_with_tie_breaker,
)
from .vision import extract_figure_insights


def _load_already_done(records_path: Path) -> set[str]:
    if not records_path.exists():
        return set()
    done: set[str] = set()
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["paper_id"])
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Skipping unparseable line in {records_path}")
    return done


def _build_prompt(template: str, schema_cls, text: str, field_reminders: str = "") -> str:
    return template.format(
        schema=json.dumps(schema_cls.model_json_schema(), indent=2),
        text=text,
        field_reminders=field_reminders,
    )


def _provenance_fields(paper: PaperRecord) -> dict:
    return {
        "paper_id": paper.paper_id,
        "doi": paper.doi,
        "title": paper.title,
        "first_author": paper.first_author,
        "journal": paper.journal,
        "publication_year": paper.publication_year,
        "material_class": paper.material_class,
        "is_review": paper.is_review,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }


def _failure_record(paper: PaperRecord, call_results: dict[str, CallResult]) -> dict:
    return {
        **_provenance_fields(paper),
        "agreement_status": "extraction_failed",
        "models_used": list(call_results.keys()),
        "errors": {m: r.error for m, r in call_results.items() if not r.ok},
        "fields": None,
    }


def _assemble_text_with_vision(paper: PaperRecord, config: AppConfig) -> str:
    """Append a vision-model description of the paper's figures to the
    section/table text already assembled by reader.py, if enabled."""
    if not config.extraction.enable_figure_vision:
        return paper.text_for_extraction

    vision_text = extract_figure_insights(
        paper_dir=paper.output_dir,
        model=config.models.vision,  # validated non-None in config.load_config()
        host=config.models.host,
        options=config.models.resolved_options(config.models.vision),
        max_figures=config.extraction.max_figures_per_paper,
        max_retries=config.extraction.vision_max_retries,
        retry_wait_seconds=config.extraction.vision_retry_wait_seconds,
    )
    if not vision_text:
        return paper.text_for_extraction
    return f"{paper.text_for_extraction}\n\n=== FIGURES ===\n{vision_text}"


def _merge_claims(primary_claims: list[dict], cross_claims: list[dict]) -> list[dict]:
    """Union claims from both models, de-duplicated by (subject, relation,
    object) case-insensitively. Claims are advisory metadata, not used for
    binary targets or ML features directly, so unlike experiment fields they
    are not put through majority-vote/tie-breaker reconciliation -- a claim
    either model surfaced and the other simply didn't repeat verbatim is
    still worth keeping, since natural-language paraphrasing means the two
    models describing the 'same' claim rarely produce byte-identical text
    anyway. The dedup key only collapses genuine duplicates (both models
    extracted the literal same subject/relation/object triple)."""
    seen: dict[tuple[str, str, str], dict] = {}
    for claim in primary_claims + cross_claims:
        key = (
            str(claim.get("subject", "")).strip().lower(),
            str(claim.get("relation", "")).strip().lower(),
            str(claim.get("object", "")).strip().lower(),
        )
        if key not in seen:
            seen[key] = claim
    return list(seen.values())


def _run_flat_paper(
    paper: PaperRecord,
    prompt: str,
    schema: SchemaBundle,
    extractor: OllamaExtractor,
    config: AppConfig,
    counters: dict[str, int],
) -> dict:
    primary_result = extractor.extract(config.models.primary, prompt, schema.ExtractionFields)
    cross_result = extractor.extract(config.models.cross_check, prompt, schema.ExtractionFields)
    call_results = {config.models.primary: primary_result, config.models.cross_check: cross_result}

    if not (primary_result.ok and cross_result.ok):
        logger.error(f"Extraction failed for {paper.paper_id[:8]} — both models must succeed")
        counters["failed"] += 1
        return _failure_record(paper, call_results)

    disagreeing = compare_two(
        primary_result.parsed, cross_result.parsed,
        config.extraction.numeric_agreement_tolerance,
    )

    tie_breaker_parsed = None
    if disagreeing:
        total_fields = len(
            [f for f in type(primary_result.parsed).model_fields if f != "extraction_confidence"]
        )
        diagnostic = diagnose_disagreement_pattern(disagreeing, total_fields)
        logger.info(
            f"  {len(disagreeing)} field(s) disagree "
            f"({', '.join(disagreeing)}) — escalating to {config.models.tie_breaker}"
        )
        if diagnostic:
            logger.warning(f"  [{paper.paper_id[:8]}] {diagnostic}")

        tie_result = extractor.extract(config.models.tie_breaker, prompt, schema.ExtractionFields)
        call_results[config.models.tie_breaker] = tie_result
        if tie_result.ok:
            tie_breaker_parsed = tie_result.parsed
        counters["escalated"] += 1
    else:
        counters["full_agreement"] += 1

    resolution = resolve_with_tie_breaker(
        primary_result.parsed, cross_result.parsed, tie_breaker_parsed,
        disagreeing, config.extraction.numeric_agreement_tolerance,
    )
    counters["unresolved_fields"] += len(resolution.unresolved_fields)

    return {
        **_provenance_fields(paper),
        "agreement_status": resolution.agreement_status,
        "models_used": list(call_results.keys()),
        "disagreeing_fields": disagreeing,
        "unresolved_fields": resolution.unresolved_fields,
        "primary_confidence": primary_result.parsed.extraction_confidence,
        "cross_check_confidence": cross_result.parsed.extraction_confidence,
        "fields": resolution.resolved,
    }


def _run_multi_experiment_paper(
    paper: PaperRecord,
    prompt: str,
    schema: SchemaBundle,
    extractor: OllamaExtractor,
    config: AppConfig,
    counters: dict[str, int],
) -> dict:
    primary_result = extractor.extract(config.models.primary, prompt, schema.ExtractionFields)
    cross_result = extractor.extract(config.models.cross_check, prompt, schema.ExtractionFields)
    call_results = {config.models.primary: primary_result, config.models.cross_check: cross_result}

    if not (primary_result.ok and cross_result.ok):
        logger.error(f"Extraction failed for {paper.paper_id[:8]} — both models must succeed")
        counters["failed"] += 1
        return _failure_record(paper, call_results)

    primary_paper, cross_paper = primary_result.parsed, cross_result.parsed
    paper_field_names = [
        name for name in type(primary_paper).model_fields if name not in ("experiments", "claims")
    ]

    # Paper-scope fields: same flat compare/resolve logic as a regular
    # field-level record, just restricted to the paper-scope subset.
    primary_paper_only = {k: getattr(primary_paper, k) for k in paper_field_names}
    cross_paper_only = {k: getattr(cross_paper, k) for k in paper_field_names}
    paper_disagreeing = [k for k in paper_field_names
                          if str(primary_paper_only[k]).strip().lower()
                          != str(cross_paper_only[k]).strip().lower()]

    n_primary_exp = len(primary_paper.experiments)
    n_cross_exp = len(cross_paper.experiments)

    # NOTE: experiments is Field(max_length=spec.max_experiments), and that
    # bound is embedded as "maxItems" in the JSON schema handed to Ollama's
    # schema-constrained (grammar-level) decoding -- the model is physically
    # unable to emit more than max_experiments array elements, and no error
    # is raised when it hits that ceiling; it just stops. A count that lands
    # EXACTLY on the ceiling is therefore indistinguishable, from this log
    # alone, between "the paper genuinely reports max_experiments rows" and
    # "the paper reports more and got silently truncated" -- the latter is a
    # real risk for combinatorial catalyst-screening papers and would
    # undercount exactly the kind of paper most valuable to this dataset.
    # Surface it so it's checked by hand rather than discovered later as an
    # unexplained gap in the extracted corpus.
    for label, n_exp in (("primary", n_primary_exp), ("cross_check", n_cross_exp)):
        if n_exp == schema.spec.max_experiments:
            logger.warning(
                f"  [{paper.paper_id[:8]}] {label} extracted exactly "
                f"max_experiments ({schema.spec.max_experiments}) rows -- this "
                "may mean the paper reports MORE catalysts/conditions than "
                "the schema allows and the list was silently truncated at "
                "generation time (schema-constrained decoding cannot exceed "
                "maxItems, and raises no error when it hits it). Manually "
                "check this paper's tables, and raise max_experiments in "
                f"domains/{schema.domain_name}.yaml if it genuinely has more."
            )

    needs_tie_breaker = bool(paper_disagreeing) or n_primary_exp != n_cross_exp

    tie_breaker_paper = None
    provisional_results = None
    if not needs_tie_breaker:
        # Even with matching counts, individual experiments may still
        # disagree field-by-field -- resolve_experiment_lists will detect
        # that and we escalate retroactively if it does.
        provisional_results = resolve_experiment_lists(
            primary_paper.experiments, cross_paper.experiments, None,
            schema.ExperimentFields, config.extraction.numeric_agreement_tolerance,
        )
        needs_tie_breaker = any(
            r.experiment_match_status != "full_agreement" for r in provisional_results
        )

    if needs_tie_breaker:
        logger.info(
            f"  experiment-level disagreement "
            f"(primary={n_primary_exp} exp, cross_check={n_cross_exp} exp) "
            f"— escalating to {config.models.tie_breaker}"
        )
        # Surface any diagnostic already computed during the provisional
        # pass (e.g. widespread per-field disagreement inside a matched
        # experiment pair, or a position-fallback match instead of a
        # confident identity-key match) directly at escalation time.
        if provisional_results:
            for r in provisional_results:
                if r.diagnostic:
                    logger.warning(f"  [{paper.paper_id[:8]}] {r.diagnostic}")
                if r.match_method == "position":
                    logger.warning(
                        f"  [{paper.paper_id[:8]}] an experiment pair was matched by "
                        "position (no unique identity-key match) — verify catalyst "
                        "naming is covered by canonicalize.py for this paper if "
                        "downstream results look wrong for this record"
                    )
        elif n_primary_exp != n_cross_exp:
            logger.warning(
                f"  [{paper.paper_id[:8]}] primary and cross_check found a different "
                f"NUMBER of experiments ({n_primary_exp} vs {n_cross_exp}) — one model "
                "likely missed a catalyst/condition row in a table; check "
                "'single_model_only' entries in the resolved output for this paper"
            )

        tie_result = extractor.extract(config.models.tie_breaker, prompt, schema.ExtractionFields)
        call_results[config.models.tie_breaker] = tie_result
        if tie_result.ok:
            tie_breaker_paper = tie_result.parsed
            n_tie_exp = len(tie_breaker_paper.experiments)
            if n_tie_exp == schema.spec.max_experiments:
                logger.warning(
                    f"  [{paper.paper_id[:8]}] tie_breaker also extracted exactly "
                    f"max_experiments ({schema.spec.max_experiments}) rows -- see "
                    "the primary/cross_check ceiling warning above; this "
                    "increases the likelihood the paper was genuinely "
                    "truncated rather than one model undercounting."
                )
        counters["escalated"] += 1
    else:
        counters["full_agreement"] += 1

    experiment_results = resolve_experiment_lists(
        primary_paper.experiments,
        cross_paper.experiments,
        tie_breaker_paper.experiments if tie_breaker_paper is not None else None,
        schema.ExperimentFields,
        config.extraction.numeric_agreement_tolerance,
    )

    resolved_paper_fields = dict(primary_paper_only)
    n_unresolved_paper_fields = 0
    if paper_disagreeing:
        for k in paper_disagreeing:
            tb_value = getattr(tie_breaker_paper, k) if tie_breaker_paper is not None else None
            candidates = [v for v in (primary_paper_only[k], cross_paper_only[k], tb_value) if v is not None]
            if len(candidates) >= 2 and str(candidates[0]).strip().lower() == str(candidates[1]).strip().lower():
                resolved_paper_fields[k] = candidates[0]
            elif len(candidates) == 3 and str(candidates[1]).strip().lower() == str(candidates[2]).strip().lower():
                resolved_paper_fields[k] = candidates[1]
            else:
                resolved_paper_fields[k] = None
                n_unresolved_paper_fields += 1

    n_unresolved_experiment_fields = sum(len(r.unresolved_fields) for r in experiment_results)
    counters["unresolved_fields"] += n_unresolved_paper_fields + n_unresolved_experiment_fields

    statuses = {r.experiment_match_status for r in experiment_results}
    if n_unresolved_paper_fields == 0 and statuses <= {"full_agreement"}:
        overall_status = "full_agreement"
    elif "unresolved_conflict" in statuses or n_unresolved_paper_fields > 0:
        overall_status = "unresolved_conflict"
    else:
        overall_status = "resolved_after_escalation"

    claims: list[dict] = []
    if schema.extract_claims:
        primary_claims = [c.model_dump() for c in getattr(primary_paper, "claims", [])]
        cross_claims = [c.model_dump() for c in getattr(cross_paper, "claims", [])]
        claims = _merge_claims(primary_claims, cross_claims)

    return {
        **_provenance_fields(paper),
        "agreement_status": overall_status,
        "models_used": list(call_results.keys()),
        "n_experiments_primary": n_primary_exp,
        "n_experiments_cross_check": n_cross_exp,
        "unresolved_paper_fields": [k for k in paper_disagreeing if resolved_paper_fields[k] is None],
        "fields": resolved_paper_fields,
        "experiments": [
            {**r.resolved, "experiment_match_status": r.experiment_match_status,
             "match_method": r.match_method, "unresolved_fields": r.unresolved_fields}
            for r in experiment_results
        ],
        "claims": claims,
    }


def run(config: AppConfig) -> Path:
    """Run the full extraction pipeline. Returns the path to extracted_records.jsonl."""
    schema: SchemaBundle = get_schema(config.extraction.schema_name, config.paths.domains_dir)
    logger.info(
        f"Loaded domain '{schema.domain_name}' from {config.paths.domains_dir} "
        f"(multi_experiment={schema.multi_experiment})"
    )
    if not schema.multi_experiment:
        logger.info(
            "  multi_experiment is False for this domain: each paper produces ONE flat "
            "record. If papers in this corpus routinely compare multiple catalysts or "
            "conditions in one table, widespread field disagreement between primary/"
            "cross_check is expected (each model may pick a different row) -- set "
            "multi_experiment: true in this domain's YAML to extract one record per "
            "experiment instead. See resolve.py's module docstring for details."
        )

    out_root = config.paths.extraction_output_root
    records_path = out_root / "extracted_records.jsonl"

    already_done = _load_already_done(records_path)
    if already_done:
        logger.info(f"Resuming — {len(already_done)} papers already extracted in a prior run")

    extractor = OllamaExtractor(
        host=config.models.host,
        models_config=config.models,
        max_retries=config.extraction.max_retries,
        retry_wait_seconds=config.extraction.retry_wait_seconds,
    )

    papers = iter_papers(
        pipeline_output_root=config.paths.pipeline_output_root,
        relevant_section_keywords=config.extraction.relevant_section_keywords,
        max_input_chars=config.extraction.max_input_chars,
        include_tables=config.extraction.include_tables,
        max_table_chars=config.extraction.max_table_chars,
        include_formulas=config.extraction.include_formulas,
        max_formula_chars=config.extraction.max_formula_chars,
        already_done=already_done,
    )

    counters = {"full_agreement": 0, "escalated": 0, "failed": 0, "unresolved_fields": 0}
    run_paper = _run_multi_experiment_paper if schema.multi_experiment else _run_flat_paper

    with open(records_path, "a", encoding="utf-8") as out_f:
        for i, paper in enumerate(papers, start=1):
            logger.info(f"[{i}] Extracting: {paper.first_author} ({paper.publication_year}) — {paper.paper_id[:8]}")

            context_text = _assemble_text_with_vision(paper, config)
            prompt = _build_prompt(
                schema.PROMPT_TEMPLATE, schema.ExtractionFields, context_text,
                field_reminders=schema.spec.field_reminders_block(),
            )

            record = run_paper(paper, prompt, schema, extractor, config, counters)
            out_f.write(json.dumps(record, default=str) + "\n")
            out_f.flush()  # one paper's progress must never depend on the next paper's success

    logger.info(
        f"Run complete — full_agreement={counters['full_agreement']}, "
        f"escalated={counters['escalated']}, failed={counters['failed']}, "
        f"unresolved_fields_total={counters['unresolved_fields']}"
    )
    return records_path
