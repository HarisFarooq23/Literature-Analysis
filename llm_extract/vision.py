"""
vision.py — OPTIONAL pass that asks a multimodal model to describe the
figures scientific-pdf-pipeline extracted (figures/figures_index.json +
the corresponding PNGs), so chart/plot data that never appears as prose or
in a table still reaches the extraction step.

Disabled by default (extraction.enable_figure_vision: false in config.yaml)
because it adds model calls per paper and not every model in a tiered setup
needs to support vision — only the one named in models.vision does.
gemma3:12b / gemma3:27b / gemma4:e4b / gemma4:31b are all multimodal in
Ollama, so no new model download is required if you already have one of
those.

BATCHING, NOT PER-FIGURE CALLS: all of a paper's figures (up to
max_figures) are sent to the vision model in a SINGLE schema-constrained
request, rather than one request per figure. This is deliberate and
matters more than it looks:

  Every Ollama request that names a different model than the one
  currently resident in VRAM forces a model swap. With `models.primary`
  (e.g. granite4.1:8b) and `models.vision` (e.g. gemma3:12b) alternating
  N+1 times per paper (N sequential vision calls, then the primary/
  cross_check/tie_breaker calls), a 5-figure paper causes repeated
  load/evict cycles on a single-GPU box. On top of the throughput cost,
  this package's sibling `scientific-pdf-pipeline` project already hit a
  hard failure mode from concurrent multi-process CUDA access on this
  exact class of hardware (RTX 5090) -- heap corruption that required
  hardcoding the Docling worker pool to size 1 and gating it behind an
  asyncio.Semaphore(1). Reaching for thread-level concurrency here to
  paper over the vision-call latency would reintroduce that same risk
  surface for a much smaller win. Reducing the NUMBER of model swaps
  (this file) is the safe fix; adding concurrency around them is not.

  One request also means one retry/backoff cycle instead of up to
  max_figures separate ones, so a paper with several figures doesn't
  multiply worst-case wall-clock time under transient server errors.

Trade-off, stated explicitly: batching asks one model call to describe
multiple images at once and report back which description belongs to
which figure_id. For a handful of figures (max_figures_per_paper defaults
to 5) this is reliable in practice; if a run's figure counts are much
higher, or you observe misattributed descriptions in the audit log
(figure_id in the response not matching what was sent), lower
max_figures_per_paper rather than reverting to per-image calls.

This module never raises on a model that turns out not to support vision,
or on a batch response that fails to parse — it logs a warning and returns
an empty string, so a misconfigured vision_model degrades gracefully
instead of crashing a multi-hour run.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import ollama
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_VISION_BATCH_PROMPT = """These are {n} figures from a single scientific paper about \
catalysis, in the order listed below. For EACH figure, describe in 2-4 \
sentences any quantitative information visible that is relevant to \
catalytic performance or reaction conditions: axis labels and ranges, \
specific data values, trends, or comparisons between catalysts/conditions. \
If a figure is a chemical structure, schematic, or device diagram with no \
quantitative data, say so in one short sentence for that figure instead of \
describing it in detail. Do not speculate about values you cannot actually \
read in an image.

Figures, in the order the images are attached (match your response to \
these IDs and captions, one entry per figure, in the SAME order):
{figure_list}

Return ONLY valid JSON matching this shape, no prose or markdown:
{{"figures": [{{"figure_id": "<id>", "description": "<your description>"}}, ...]}}
"""


class _FigureDescription(BaseModel):
    figure_id: str
    description: str


class _FigureBatchResponse(BaseModel):
    figures: list[_FigureDescription] = Field(default_factory=list)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _call_vision_batch(
    client: ollama.Client,
    model: str,
    images: list[bytes],
    prompt: str,
    options: dict,
) -> str:
    """One request, all images attached. Mirrors ollama_client.py's
    schema-first-then-format=json fallback: a ConnectionError means the
    server is unreachable and is propagated immediately (retrying with a
    different `format` won't help); ollama.ResponseError means the server
    actively rejected the schema-constrained request (e.g. an older Ollama
    version, or -- relevant here -- a model that errors on `format` with
    multiple images even though it accepts them for plain generation), so
    it's worth retrying once with the unconstrained format.
    """
    try:
        response = client.generate(
            model=model,
            prompt=prompt,
            images=images,
            format=_FigureBatchResponse.model_json_schema(),
            options=options,
        )
        text = response["response"]
        if text and text.strip():
            return text
        logger.debug(
            f"[vision:{model}] schema-constrained batch call returned an "
            "empty response body; falling back to format='json'"
        )
    except ConnectionError:
        raise
    except ollama.ResponseError as exc:
        logger.debug(
            f"[vision:{model}] server rejected schema-constrained batch "
            f"request (status {getattr(exc, 'status_code', '?')}: {exc}); "
            "falling back to format='json'"
        )

    response = client.generate(
        model=model, prompt=prompt, images=images, format="json", options=options,
    )
    return response["response"]


def _describe_figures_batch(
    client: ollama.Client,
    model: str,
    entries: list[dict],
    images: list[bytes],
    options: dict,
    max_retries: int,
    retry_wait_seconds: int,
) -> Optional[_FigureBatchResponse]:
    figure_list = "\n".join(
        f"  {i + 1}. id=\"{e.get('id', e.get('image'))}\" caption=\"{e.get('caption') or '(no caption)'}\""
        for i, e in enumerate(entries)
    )
    prompt = _VISION_BATCH_PROMPT.format(n=len(entries), figure_list=figure_list)

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            raw = _call_vision_batch(client, model, images, prompt, options)
            parsed = _FigureBatchResponse.model_validate(json.loads(_strip_fences(raw)))
            return parsed
        except ConnectionError as exc:
            # Server unreachable -- retrying immediately with the same
            # request wouldn't help within this attempt, but the outer
            # loop's backoff sleep gives a transient outage a chance to
            # clear before the next try.
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                f"[vision:{model}] attempt {attempt}/{max_retries} "
                f"failed (server unreachable): {last_error}"
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                f"[vision:{model}] attempt {attempt}/{max_retries} "
                f"produced invalid batch output: {last_error}"
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                f"[vision:{model}] attempt {attempt}/{max_retries} failed: {last_error}"
            )

        if attempt < max_retries:
            time.sleep(retry_wait_seconds)

    logger.warning(
        f"[vision:{model}] giving up on this paper's figures after "
        f"{max_retries} attempts ({len(entries)} figure(s) skipped): {last_error}"
    )
    return None


def extract_figure_insights(
    paper_dir: Path,
    model: str,
    host: str,
    options: dict,
    max_figures: int = 5,
    max_retries: int = 2,
    retry_wait_seconds: int = 3,
) -> str:
    """Return a text block describing up to max_figures figures for this
    paper, or '' if there are no figures / the figures index is missing /
    the vision model failed on every retry. One Ollama call per paper
    (all figures batched together), not one call per figure -- see the
    module docstring for why this matters beyond raw throughput.
    """
    figures_dir = paper_dir / "figures"
    index_path = figures_dir / "figures_index.json"
    if not index_path.exists():
        return ""

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning(f"Unreadable figures_index.json in {paper_dir}")
        return ""

    entries = []
    images: list[bytes] = []
    for entry in index[:max_figures]:
        image_name = entry.get("image")
        if not image_name:
            continue
        image_path = figures_dir / image_name
        if not image_path.exists():
            continue
        entries.append(entry)
        images.append(image_path.read_bytes())

    if not entries:
        return ""

    client = ollama.Client(host=host)
    result = _describe_figures_batch(
        client, model, entries, images, options, max_retries, retry_wait_seconds,
    )
    if result is None or not result.figures:
        return ""

    # Preserve the paper's figure order even if the model reordered its
    # response; fall back to response order for any figure_id it didn't
    # echo back exactly (e.g. minor id normalization by the model).
    by_id = {d.figure_id: d.description for d in result.figures}
    blocks: list[str] = []
    for i, entry in enumerate(entries):
        fig_id = str(entry.get("id", entry.get("image")))
        description = by_id.get(fig_id) or (
            result.figures[i].description if i < len(result.figures) else ""
        )
        if description:
            blocks.append(f"Figure {fig_id}: {description}")

        if blocks:
            print("\n" + "=" * 80)
            print(f"Successfully extracted {len(blocks)} figure(s) from {paper_dir.name}")
            print("=" * 80)

            for block in blocks:
                print(block)
                print("-" * 80)

    if not blocks:
        return ""

    return (
        "NOTE: the following are model-generated descriptions of figures/charts, "
        "not text extracted verbatim from the paper. Treat as lower-confidence "
        "supporting context, not a primary source.\n\n" + "\n\n".join(blocks)
    )
