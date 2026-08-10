
$env:Path += ";$env:LOCALAPPDATA\Programs\Ollama"

# config.yaml — single source of truth for the extraction layer.
#
# BEGINNER SETUP:
#   1. Put your PDF files in ./input_pdfs/
#   2. Install Ollama + pull the models listed under models: below
#   3. From this folder run:  llm-extract from-pdfs --config config.yaml
#   4. Open results in ./extraction_output/extracted_dataset.csv
#
# Nothing in this package should hardcode a path, model name, prompt, or
# extracted property outside of this file and the files it points to:
#   - which properties get extracted, and the prompt wording -> domains/<schema_name>.yaml + prompts/<schema_name>.txt
#   - which models do the extracting                          -> models: below (any Ollama model name works)
#   - everything else (paths, thresholds, retries)             -> below

paths:
  # Folder where YOU drop raw PDF files. Created automatically if missing.
  # Windows paths: prefer forward slashes, e.g. "C:/Users/You/Papers"
  pdfs_input_dir: "./input_pdfs"

  # Where parsed paper folders are written (text, tables, figures).
  # Created automatically by `llm-extract from-pdfs`.
  pipeline_output_root: "./pipeline_output"

  # Where the final CSV / parquet dataset is written.
  extraction_output_root: "./extraction_output"

  # Domain YAML files. Keep this as ./domains and run commands from this
  # project folder (the one that contains pyproject.toml).
  domains_dir: "./domains"

models:
  # Bulk pass — every paper goes through both of these. Any model name you
  # have pulled in `ollama list` works here.
  # Tip for beginners / small PCs: set all three to the SAME small model
  # (e.g. "llama3.2:3b") so you only download one model.
  primary: "granite4.1:8b"
  cross_check: "qwen3:8b"

  # Escalation pass — only papers where primary/cross_check disagree.
  tie_breaker: "gemma3:12b"

  # Multimodal model for optional figure/chart description. Only required
  # if extraction.enable_figure_vision is true below.
  vision: "qwen2.5vl:3b"