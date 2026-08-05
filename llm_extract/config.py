"""
Config loading and validation for llm_extract.

A single Pydantic model mirrors config.yaml exactly, so a malformed config
fails fast at startup with a clear error instead of an obscure KeyError
three hours into a corpus run.

Nothing here is biodiesel-specific or model-specific: model names are plain
strings (point them at whatever you have pulled in Ollama), and the
extraction schema/prompt/targets all come from domains_dir, not from this
file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class PathsConfig(BaseModel):
    # Folder where you drop raw PDFs for `llm-extract from-pdfs`.
    # Optional for legacy `run` (which reads pipeline_output_root only).
    pdfs_input_dir: Path | None = None
    pipeline_output_root: Path
    extraction_output_root: Path
    domains_dir: Path


class ModelsConfig(BaseModel):
    primary: str
    cross_check: str
    tie_breaker: str
    vision: str | None = None   # only required if extraction.enable_figure_vision is true
    host: str = "http://localhost:11434"
    options: dict[str, Any] = Field(default_factory=dict)
    # Optional per-model overrides merged on top of `options`, e.g. a model
    # that needs a different num_ctx than the rest:
    #   per_model_options: { "nemotron-3-nano:30b": { num_ctx: 32768 } }
    per_model_options: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def resolved_options(self, model: str) -> dict[str, Any]:
        return {**self.options, **self.per_model_options.get(model, {})}


class ExtractionConfig(BaseModel):
    schema_name: str = "biodiesel"   # matches a file in paths.domains_dir, e.g. domains/biodiesel.yaml
    relevant_section_keywords: list[str] = Field(default_factory=list)
    max_input_chars: int = 18000
    numeric_agreement_tolerance: float = 0.10
    max_retries: int = 3
    retry_wait_seconds: int = 4

    include_tables: bool = True
    max_table_chars: int = 6000

    # Off by default: most domain schemas extract values that live in prose
    # and tables, not equations. Turn on per-domain/per-run if a schema
    # genuinely needs cited formulas (e.g. a rate law or thermodynamic
    # relation) -- see formulas.py for the rationale.
    include_formulas: bool = False
    max_formula_chars: int = 3000

    enable_figure_vision: bool = False
    max_figures_per_paper: int = 5
    vision_max_retries: int = 2
    vision_retry_wait_seconds: int = 3


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: Path


class AppConfig(BaseModel):
    paths: PathsConfig
    models: ModelsConfig
    extraction: ExtractionConfig
    logging: LoggingConfig

    @field_validator("paths")
    @classmethod
    def _required_dirs_must_exist(cls, v: PathsConfig) -> PathsConfig:
        # pipeline_output_root is created by `from-pdfs` (or by an external
        # parser). Allow missing at load time; mkdir happens in load_config.
        if not v.domains_dir.exists():
            raise ValueError(
                f"domains_dir does not exist: {v.domains_dir}. "
                "This must point at the directory containing <name>.yaml domain specs. "
                "Run commands from the project folder that contains domains/."
            )
        return v


def load_config(path: str | Path) -> AppConfig:
    """Load and validate config.yaml. Raises a clear ValueError/ValidationError
    on any structural problem — fail fast, before any Ollama calls are made."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config = AppConfig.model_validate(raw)

    if config.extraction.enable_figure_vision and not config.models.vision:
        raise ValueError(
            "extraction.enable_figure_vision is true but models.vision is not set. "
            "Set models.vision to a multimodal Ollama model, e.g. gemma3:12b."
        )

    config.paths.pipeline_output_root.mkdir(parents=True, exist_ok=True)
    config.paths.extraction_output_root.mkdir(parents=True, exist_ok=True)
    config.logging.log_dir.mkdir(parents=True, exist_ok=True)
    if config.paths.pdfs_input_dir is not None:
        config.paths.pdfs_input_dir.mkdir(parents=True, exist_ok=True)
    return config
