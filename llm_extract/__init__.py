"""
llm_extract — LLM-based structured extraction layer.

Two entry paths:
  - `llm-extract from-pdfs` — parse PDFs in paths.pdfs_input_dir, then extract
  - `llm-extract run` — extract from already-parsed paper folders under
    paths.pipeline_output_root (metadata.csv + cleaned_full_text.txt, plus
    optional sections/ tables/ figures/)

Produces a clean, typed dataset (parquet/csv) for downstream ML packages.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("llm-extract")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
