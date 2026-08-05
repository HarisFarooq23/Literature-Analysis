# llm_extract

LLM-based structured extraction from scientific PDFs (or from already-parsed
paper folders). Drop PDFs in `input_pdfs/`, run one command, get a clean
dataset.

```
[your PDFs]  -->  [pdf ingest]  -->  [llm_extract]  -->  [extracted_dataset.parquet]
                       ^                  ^
                 (built-in)         (this repo)
```

You can still point `pipeline_output_root` at folders written by an external
`scientific-pdf-pipeline` and use `llm-extract run` if you prefer that path.

## What changed from v0.1

| Capability | How |
|---|---|
| **Tables** | `tables/<id>.json` (written by the pipeline's Exporter) are rendered as markdown-ish text and appended to the prompt context by default. Catalyst/condition/property combinations are routinely reported only in a table, not in prose — skipping them was throwing away signal. |
| **Figures / charts** | Optional: a multimodal model (`models.vision`, e.g. `gemma3:12b`) describes each figure and appends the description as lower-confidence supporting context. Off by default (`extraction.enable_figure_vision: false`) since it roughly doubles model calls per paper. |
| **Any LLM model** | Always true, but worth restating: `models.primary` / `cross_check` / `tie_breaker` / `vision` are plain strings — point them at anything in `ollama list`. |
| **Editable prompts** | Prompts now live in `prompts/<domain>.txt` as plain text, not embedded in Python. Edit the wording, re-run, no code touched. |
| **New domains without code** | What gets extracted is defined in `domains/<name>.yaml` — field names, types, units, enums, and the binary "meets spec" targets. Adding methanol synthesis or MOF defect engineering meant writing a YAML file each, not a line of Python. Both are included as ready-to-use examples alongside `biodiesel.yaml`. |

## What this package reads

One directory per paper under `pipeline_output_root`, each containing
`metadata.csv` (required) and `cleaned_full_text.txt`, plus optionally
`sections/`, `tables/`, and `figures/`. This is exactly
scientific-pdf-pipeline's existing output — nothing needs to change there.

## What this package writes

Everything under `extraction_output_root`:

| File | Purpose |
|---|---|
| `extracted_dataset.parquet` | The primary artifact — typed, ready for `pandas.read_parquet()`. |
| `extracted_dataset.csv` | Same data, human-inspectable. |
| `extracted_dataset.schema.json` | JSON Schema describing every column, generated from whichever domain YAML produced this run. |
| `data_dictionary.md` | Human-readable column/unit reference, auto-generated from the domain YAML. |
| `manifest.json` | Run provenance: domain name, model names, timestamp, paper counts, agreement-rate breakdown. |
| `extracted_records.jsonl` | Append-only audit log, one line per paper. Also the resumability checkpoint. |
| `failed_extractions.csv` | Papers where neither model produced valid output after retries. |
| `unresolved_field_flags.csv` | (paper, field) pairs where even the tie-breaker model didn't reach a majority. |

A downstream ML/optimization package should depend on **only**
`extracted_dataset.parquet` + `extracted_dataset.schema.json`.

## Usage (simple: PDFs in → dataset out)

```bash
pip install -e ".[dev]"
# put PDFs in ./input_pdfs/
# edit config.yaml model names to match `ollama list`
llm-extract from-pdfs --config config.yaml
```

On Windows PowerShell you can also run `.\run_pipeline.ps1` from this folder.

## Usage (legacy: already-parsed paper folders)

```bash
pip install -e ".[dev]"
# edit config.yaml paths + model names
llm-extract run --config config.yaml
```

Re-running the same command resumes automatically — papers already present
in `extracted_records.jsonl` are skipped.

Change only a binary-target threshold (e.g. raise `meets_cetane_spec` to 52)
without re-calling Ollama:

```bash
# edit domains/biodiesel.yaml's binary_targets section, then:
llm-extract build-dataset --config my_run.yaml
```

## Switching domains

```yaml
# config.yaml
extraction:
  schema_name: "methanol_synthesis"   # or "mof_defect_engineering", or "biodiesel"
```

That's the entire change required to retarget the whole pipeline.

## Adding a brand-new domain

1. Copy `domains/biodiesel.yaml` to `domains/<your_domain>.yaml`. Edit
   `domain_name`, `description`, `fields` (name/type/description/unit/enum/
   required), and `binary_targets` (field/op/value/description). Supported
   `op` values: `>=`, `<=`, `==`, `!=`, `between` (value is `[low, high]`).
2. Copy `prompts/biodiesel.txt` to `prompts/<your_domain>.txt` and rewrite
   the domain-specific instructions (the bottom `{schema}` / `{text}`
   placeholders must stay).
3. Point `extraction.schema_name` at `<your_domain>` in config.yaml.

No Python file in this package needs to change. `domain_spec.py` builds the
Pydantic model, prompt, and binary-target logic dynamically from the YAML at
runtime — `tests/test_domain_registry.py` is the regression check that every
shipped domain still loads cleanly.

## Including tables and figures

```yaml
extraction:
  include_tables: true        # on by default
  max_table_chars: 6000       # separate, additive budget from max_input_chars

  enable_figure_vision: false # off by default -- doubles model calls per paper
  max_figures_per_paper: 5
models:
  vision: "gemma3:12b"        # any multimodal Ollama model; gemma3:12b/27b and
                               # gemma4:e4b/31b are vision-capable out of the box
```

When figure vision is on, each paper's `figures/figures_index.json` entries
are sent to `models.vision` with a prompt asking it to describe any
quantitative content (axis labels, values, trends) — not to caption the
image generically. The result is appended to the prompt context under a
`=== FIGURES ===` marker and explicitly flagged to the extraction model as
lower-confidence, model-generated context, not primary-source text.

## Reliability design

- **Dual-model extraction + escalation tie-breaker.** Every paper goes
  through two independent models; disagreements above a configurable
  tolerance get a third model as tie-breaker. Categorical vs. numeric
  agreement logic is read from each domain's own Pydantic type annotations,
  not hardcoded per domain, so the same resolution code works for every
  domain.
- **Nullable everything.** "Not reported in the paper" and "reported and
  fails the spec" are kept as distinct values (`pandas.NA` vs `False`) all
  the way through to the final dataset, for every domain, automatically.
- **Fail loud, not silent.** A paper either yields a fully validated record
  or an explicit entry in `failed_extractions.csv` with the model errors
  attached.
- **Resumable by construction.** `extracted_records.jsonl` is both the audit
  log and the checkpoint.
- **Vision failures degrade gracefully.** If `models.vision` turns out not
  to support image input, `vision.py` logs a warning and returns no figure
  context for that paper rather than crashing the run.
- **Fail fast on startup, not three hours in.** `llm-extract run` calls
  `check_ollama_ready()` before touching a single paper: it confirms the
  Ollama server is reachable and that every configured model (`primary`,
  `cross_check`, `tie_breaker`, and `vision` if figure-vision is on) has
  actually been pulled. A typo'd model name or an un-started `ollama serve`
  fails immediately with a clear message, not mid-corpus. `build-dataset`
  deliberately skips this check, since it only re-runs normalization on
  already-extracted records and needs no live Ollama server at all.
- **The schema-fallback path only fires when it should.** The first attempt
  at each model call uses Ollama's schema-constrained generation. If the
  server is simply unreachable (`ConnectionError`), that's propagated
  immediately -- retrying with `format='json'` against a server that's down
  wouldn't help and would just double the failed requests. The fallback to
  `format='json'` is reserved for `ollama.ResponseError` -- a server
  actively rejecting the schema parameter (e.g. an older Ollama version).

## Testing

```bash
pytest
```

45 tests cover: the reader's file-contract parsing (including tables),
table rendering from the three JSON shapes the pipeline can emit, the YAML
domain-spec engine (dynamic Pydantic model construction, validation
errors), the dual-model agreement/resolution logic (type-driven, not
domain-specific), normalization against the real shipped `biodiesel.yaml`,
a parametrized smoke test that every shipped domain
(biodiesel/methanol_synthesis/mof_defect_engineering) loads and builds
correctly, and the Ollama reliability fixes (fail-fast readiness check,
and the corrected connection-error-vs-schema-rejection fallback logic --
mocked, no live server needed). `vision.py`'s actual image-description
call is the one piece without a unit test here -- verify it against your
real vision model with a small pilot run before scaling up.
