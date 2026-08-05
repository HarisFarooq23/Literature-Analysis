"""
cli.py — command-line entry point.

Usage:
    llm-extract from-pdfs --config config.yaml   # PDF folder -> parse -> LLM extract
    llm-extract run --config config.yaml         # LLM extract only (parsed folders)
    llm-extract build-dataset --config config.yaml
    llm-extract make-annotation-template --config config.yaml --n-papers 30 \
        --out gold_standard_template.jsonl
    llm-extract benchmark --config config.yaml --gold gold_standard.jsonl
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from .benchmark.annotation_template import write_annotation_template
from .benchmark.score import score as run_benchmark_score
from .config import load_config
from .logging_config import configure_logging
from .normalize import normalize
from .ollama_client import check_ollama_ready
from .pdf_ingest import ingest_pdfs
from .pipeline import run as run_pipeline
from .writer import write_outputs

app = typer.Typer(add_completion=False)


def _run_llm_extraction(cfg) -> Path:
    """Shared path: Ollama readiness check -> dual-model extract -> dataset write."""
    required_models = [cfg.models.primary, cfg.models.cross_check, cfg.models.tie_breaker]
    if cfg.extraction.enable_figure_vision:
        required_models.append(cfg.models.vision)

    logger.info(f"Checking Ollama server at {cfg.models.host} and required models...")
    check_ollama_ready(cfg.models, required_models)
    logger.info("Ollama reachable, all configured models present. Starting extraction run")

    records_path = run_pipeline(cfg)

    logger.info("Extraction complete, building clean dataset")
    tables = normalize(records_path, cfg.extraction.schema_name, cfg.paths.domains_dir)
    out_dir = write_outputs(cfg, tables["clean"], tables["failures"], tables["unresolved_field_flags"])
    logger.info(f"Done. Clean dataset and manifest written to {out_dir}")
    return out_dir


@app.command(name="from-pdfs")
def from_pdfs(
    config: Path = typer.Option(..., "--config", "-c", help="Path to config.yaml"),
) -> None:
    """Parse every PDF in paths.pdfs_input_dir, then run LLM extraction.

    This is the simple end-to-end path: put PDFs in the input folder, run
    this command, read results under paths.extraction_output_root.
    """
    cfg = load_config(config)
    configure_logging(cfg.logging.log_dir, cfg.logging.level)

    if cfg.paths.pdfs_input_dir is None:
        raise typer.BadParameter(
            "paths.pdfs_input_dir is not set in config.yaml. "
            "Add e.g. pdfs_input_dir: \"./input_pdfs\" under paths:."
        )

    logger.info(f"Step 1/2 — parsing PDFs from {cfg.paths.pdfs_input_dir}")
    ingest_pdfs(cfg.paths.pdfs_input_dir, cfg.paths.pipeline_output_root)

    logger.info("Step 2/2 — LLM structured extraction")
    _run_llm_extraction(cfg)


@app.command()
def run(config: Path = typer.Option(..., "--config", "-c", help="Path to config.yaml")) -> None:
    """Run LLM extraction on already-parsed paper folders (pipeline_output_root)."""
    cfg = load_config(config)
    configure_logging(cfg.logging.log_dir, cfg.logging.level)
    _run_llm_extraction(cfg)


@app.command(name="build-dataset")
def build_dataset(config: Path = typer.Option(..., "--config", "-c")) -> None:
    """Re-run normalization + dataset writing from an existing
    extracted_records.jsonl without calling Ollama again. Use this after
    changing binary-target thresholds in a domain YAML file."""
    cfg = load_config(config)
    configure_logging(cfg.logging.log_dir, cfg.logging.level)

    records_path = cfg.paths.extraction_output_root / "extracted_records.jsonl"
    if not records_path.exists():
        raise typer.BadParameter(f"No extracted_records.jsonl at {records_path}; run 'llm-extract run' first.")

    tables = normalize(records_path, cfg.extraction.schema_name, cfg.paths.domains_dir)
    out_dir = write_outputs(cfg, tables["clean"], tables["failures"], tables["unresolved_field_flags"])
    logger.info(f"Rebuilt clean dataset at {out_dir}")


@app.command(name="make-annotation-template")
def make_annotation_template(
    config: Path = typer.Option(..., "--config", "-c", help="Path to config.yaml"),
    n_papers: int = typer.Option(..., "--n-papers", help="Number of papers to sample for annotation"),
    out: Path = typer.Option(..., "--out", help="Output path for the gold-standard template JSONL"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducible sampling"),
    n_experiment_slots: int = typer.Option(
        3, "--n-experiment-slots",
        help="Starting number of blank experiment rows per paper (annotators add/remove as needed)",
    ),
    existing_gold: Optional[Path] = typer.Option(
        None, "--existing-gold",
        help="Path to an existing gold-standard JSONL to exclude already-annotated papers from re-sampling",
    ),
) -> None:
    """Generate a fill-in-the-blank gold-standard annotation file: N randomly
    sampled papers, in the exact schema shape the pipeline itself produces,
    for a human to complete by reading the actual papers. This is the
    methodology-validation step a paper needs before reporting any
    downstream ML result -- see llm-extract benchmark for scoring the
    completed file against pipeline output."""
    cfg = load_config(config)
    configure_logging(cfg.logging.log_dir, cfg.logging.level)

    already_annotated_ids = None
    if existing_gold and existing_gold.exists():
        import json
        already_annotated_ids = set()
        with open(existing_gold, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    already_annotated_ids.add(json.loads(line)["paper_id"])
        logger.info(f"Excluding {len(already_annotated_ids)} already-annotated papers from {existing_gold}")

    out_path = write_annotation_template(
        out_path=out,
        schema_name=cfg.extraction.schema_name,
        domains_dir=cfg.paths.domains_dir,
        pipeline_output_root=cfg.paths.pipeline_output_root,
        n_papers=n_papers,
        seed=seed,
        n_experiment_slots=n_experiment_slots,
        already_annotated_ids=already_annotated_ids,
    )
    logger.info(
        f"Wrote annotation template for {n_papers} papers to {out_path}. "
        "Fill in the null values by reading each paper, then run "
        "'llm-extract benchmark --gold {out_path}' once complete."
    )


@app.command()
def benchmark(
    config: Path = typer.Option(..., "--config", "-c", help="Path to config.yaml"),
    gold: Path = typer.Option(..., "--gold", help="Path to the completed gold-standard JSONL"),
    tolerance: float = typer.Option(
        0.10, "--tolerance",
        help="Relative tolerance for numeric field agreement (default matches "
             "extraction.numeric_agreement_tolerance's typical setting)",
    ),
    out_csv: Optional[Path] = typer.Option(
        None, "--out-csv", help="Optional path to write the per-field score table as CSV"
    ),
) -> None:
    """Score extracted_records.jsonl against a completed gold-standard file.
    Prints a per-field precision/recall/F1 (categorical) and MAE/RMSE/
    within-tolerance (numeric) summary -- the table a methods section
    needs to report extraction accuracy before any downstream ML claim."""
    cfg = load_config(config)
    configure_logging(cfg.logging.log_dir, cfg.logging.level)

    records_path = cfg.paths.extraction_output_root / "extracted_records.jsonl"
    if not records_path.exists():
        raise typer.BadParameter(f"No extracted_records.jsonl at {records_path}; run 'llm-extract run' first.")

    result = run_benchmark_score(
        extracted_records_path=records_path,
        gold_standard_path=gold,
        schema_name=cfg.extraction.schema_name,
        domains_dir=cfg.paths.domains_dir,
        numeric_tolerance=tolerance,
    )

    print(result.summary())

    if out_csv:
        result.to_dataframe().to_csv(out_csv, index=False)
        logger.info(f"Per-field score table written to {out_csv}")


if __name__ == "__main__":
    app()
