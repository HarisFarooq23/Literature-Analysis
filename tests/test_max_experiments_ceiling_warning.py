"""
test_max_experiments_ceiling_warning.py — coverage for the warning added to
_run_multi_experiment_paper when a model's experiment count lands exactly on
max_experiments. Regression context: schema-constrained (grammar-level)
decoding embeds max_length as "maxItems" in the JSON schema, so a model
physically cannot emit more than max_experiments rows and raises no error
when it hits that ceiling -- a paper that genuinely reports MORE rows than
the schema allows is silently truncated with no signal in the log. This was
found (not yet observed as an actual overflow, but the ceiling-hit pattern
itself was observed 3 times) while reviewing a production run log.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from loguru import logger

from llm_extract.config import AppConfig, ExtractionConfig, LoggingConfig, ModelsConfig, PathsConfig
from llm_extract.domain_registry import get_schema
from llm_extract.ollama_client import CallResult
from llm_extract.pipeline import OllamaExtractor, _run_multi_experiment_paper
from llm_extract.reader import PaperRecord


def _make_paper_record(paper_dir: Path) -> PaperRecord:
    return PaperRecord(
        paper_id="ceiling00", title="T", first_author="A", journal="J",
        publication_year=2024, doi="10.1/x", material_class="test_ceiling",
        is_review=False, output_dir=paper_dir, abstract="",
        text_for_extraction="dummy text",
    )


def _make_call_result(parsed) -> CallResult:
    return CallResult(model="m", ok=True, parsed=parsed, raw_response="{}",
                       error=None, attempts=1, elapsed_seconds=0.1)


def _write_test_domain(tmp_path: Path, max_experiments: int) -> Path:
    domains_dir = tmp_path / "domains"
    domains_dir.mkdir(exist_ok=True)
    (domains_dir / "test_ceiling.yaml").write_text(yaml.dump({
        "domain_name": "test_ceiling",
        "prompt_template_path": "test_ceiling_prompt.txt",
        "multi_experiment": True,
        "min_experiments": 1,
        "max_experiments": max_experiments,
        "fields": [
            {"name": "active_metal", "type": "str", "scope": "experiment"},
        ],
    }), encoding="utf-8")
    (domains_dir / "test_ceiling_prompt.txt").write_text(
        "Schema: {schema}\nText: {text}\n", encoding="utf-8",
    )
    return domains_dir


def _build_schema_and_config(tmp_path: Path, max_experiments: int = 2):
    domains_dir = _write_test_domain(tmp_path, max_experiments)
    schema = get_schema("test_ceiling", domains_dir)

    config = AppConfig(
        paths=PathsConfig(
            pipeline_output_root=tmp_path, extraction_output_root=tmp_path / "out",
            domains_dir=domains_dir,
        ),
        models=ModelsConfig(primary="primary_model", cross_check="cross_check_model",
                             tie_breaker="tie_breaker_model"),
        extraction=ExtractionConfig(schema_name="test_ceiling"),
        logging=LoggingConfig(level="INFO", log_dir=tmp_path / "logs"),
    )
    return schema, config


def _capture_warnings():
    captured = []
    handler_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    return captured, handler_id


def _run(paper_dir, prompt_text, schema, extractor, config):
    return _run_multi_experiment_paper(
        _make_paper_record(paper_dir), prompt_text, schema, extractor, config,
        counters={"failed": 0, "escalated": 0, "full_agreement": 0, "unresolved_fields": 0},
    )


def test_warning_fires_when_primary_hits_max_experiments_ceiling(tmp_path):
    schema, config = _build_schema_and_config(tmp_path, max_experiments=2)
    Model = schema.ExtractionFields

    # Primary hits the ceiling exactly (2 == max_experiments); cross_check
    # does not (1 < max_experiments) -- this ALSO triggers the pre-existing
    # "different NUMBER of experiments" warning, which is fine, both should
    # fire.
    primary_parsed = Model(experiments=[
        {"active_metal": "Ni", "evidence": {}, "experiment_confidence": 0.9},
        {"active_metal": "Pt", "evidence": {}, "experiment_confidence": 0.9},
    ])
    cross_parsed = Model(experiments=[
        {"active_metal": "Ni", "evidence": {}, "experiment_confidence": 0.9},
    ])
    tie_parsed = Model(experiments=[
        {"active_metal": "Ni", "evidence": {}, "experiment_confidence": 0.9},
    ])

    def fake_extract(model, prompt, response_model):
        return _make_call_result(
            {"primary_model": primary_parsed, "cross_check_model": cross_parsed}
            .get(model, tie_parsed)
        )

    captured, handler_id = _capture_warnings()
    try:
        with patch("llm_extract.pipeline.OllamaExtractor.extract", side_effect=fake_extract):
            extractor = OllamaExtractor(host="http://localhost:11434", models_config=config.models)
            _run(tmp_path, "prompt text", schema, extractor, config)
    finally:
        logger.remove(handler_id)

    ceiling_warnings = [w for w in captured if "exactly max_experiments" in w]
    assert len(ceiling_warnings) == 1
    assert "primary" in ceiling_warnings[0]
    assert "ceiling00"[:8] in ceiling_warnings[0]


def test_no_ceiling_warning_when_no_model_hits_the_bound(tmp_path):
    schema, config = _build_schema_and_config(tmp_path, max_experiments=5)
    Model = schema.ExtractionFields

    primary_parsed = Model(experiments=[
        {"active_metal": "Ni", "evidence": {}, "experiment_confidence": 0.9},
    ])
    cross_parsed = Model(experiments=[
        {"active_metal": "Ni", "evidence": {}, "experiment_confidence": 0.9},
    ])

    def fake_extract(model, prompt, response_model):
        return _make_call_result(
            primary_parsed if model == "primary_model" else cross_parsed
        )

    captured, handler_id = _capture_warnings()
    try:
        with patch("llm_extract.pipeline.OllamaExtractor.extract", side_effect=fake_extract):
            extractor = OllamaExtractor(host="http://localhost:11434", models_config=config.models)
            _run(tmp_path, "prompt text", schema, extractor, config)
    finally:
        logger.remove(handler_id)

    assert not [w for w in captured if "exactly max_experiments" in w]


def test_tie_breaker_ceiling_hit_is_also_flagged(tmp_path):
    """When primary/cross_check disagree (forcing escalation) and the
    tie_breaker's own result ALSO lands on the ceiling, that should be
    flagged too -- the tie_breaker result is often what downstream
    resolution actually prefers."""
    schema, config = _build_schema_and_config(tmp_path, max_experiments=2)
    Model = schema.ExtractionFields

    primary_parsed = Model(experiments=[
        {"active_metal": "Ni", "evidence": {}, "experiment_confidence": 0.9},
    ])
    cross_parsed = Model(experiments=[
        {"active_metal": "Ni", "evidence": {}, "experiment_confidence": 0.9},
        {"active_metal": "Pt", "evidence": {}, "experiment_confidence": 0.9},
    ])
    tie_parsed = Model(experiments=[
        {"active_metal": "Ni", "evidence": {}, "experiment_confidence": 0.9},
        {"active_metal": "Pt", "evidence": {}, "experiment_confidence": 0.9},
    ])

    def fake_extract(model, prompt, response_model):
        return _make_call_result({
            "primary_model": primary_parsed,
            "cross_check_model": cross_parsed,
            "tie_breaker_model": tie_parsed,
        }[model])

    captured, handler_id = _capture_warnings()
    try:
        with patch("llm_extract.pipeline.OllamaExtractor.extract", side_effect=fake_extract):
            extractor = OllamaExtractor(host="http://localhost:11434", models_config=config.models)
            _run(tmp_path, "prompt text", schema, extractor, config)
    finally:
        logger.remove(handler_id)

    assert any("tie_breaker also extracted exactly max_experiments" in w for w in captured)
