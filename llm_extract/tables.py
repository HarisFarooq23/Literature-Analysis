"""
tables.py — renders the tables/<id>.json files written by
scientific-pdf-pipeline's Exporter into plain text suitable for an LLM
prompt. Pure file I/O, no network calls, so it stays unit-testable
alongside reader.py.

Table.data, per the pipeline's Table dataclass, is one of:
  - list[list]   — structured rows, first row usually a header
  - dict         — less common, e.g. {"header": [...], "rows": [[...]]}
  - str          — raw/unstructured text (OCR fallback, or a table the
                   parser couldn't structure)

Catalysis papers routinely put the exact catalyst/condition/yield
combinations in tables rather than prose — skipping tables throws away a
large fraction of the extractable signal, so this is included by default.

TRUNCATION STRATEGY: max_chars is a hard budget, but a naive text[:max_chars]
slice across the concatenated tables can cut a table off mid-row, producing
a malformed markdown fragment (e.g. a row with 6 of 10 columns) that is
silently sent to the LLM with no indication anything was lost. For a wide
multi-component product-distribution table (e.g. C8-C16 fraction breakdown
across several catalysts), that missing context is exactly the kind of
value the extraction schema needs. render_tables_text instead: (1) includes
whole tables until the budget is exhausted, then (2) if the very next table
alone would still fit if row-truncated, includes as many WHOLE rows of it as
fit, and (3) logs a warning naming exactly which table(s)/row(s) were
dropped, so truncation is a visible, debuggable event instead of a silent
one.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger


def _render_rows(rows: list[list]) -> str:
    lines = []
    for i, row in enumerate(rows):
        cells = [str(c) for c in row]
        lines.append(" | ".join(cells))
        if i == 0:
            lines.append(" | ".join("---" for _ in cells))
    return "\n".join(lines)


def _render_table_data(data) -> str:
    if isinstance(data, list) and data and isinstance(data[0], list):
        return _render_rows(data)
    if isinstance(data, dict) and "header" in data and "rows" in data:
        return _render_rows([data["header"], *data["rows"]])
    if isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False)
    return str(data)


def _truncate_table_by_rows(header_block: str, data, remaining_chars: int) -> tuple[str, bool]:
    """For a single table that doesn't fit whole: if it's row-structured,
    include as many WHOLE rows (starting from the header, since a data
    table without its header is close to useless to the model) as fit in
    remaining_chars. Returns (rendered_text, was_truncated). For
    non-row-structured data (a dict or raw string the parser couldn't
    tabulate), there is no row boundary to truncate at, so it's either
    included whole or dropped entirely -- never character-sliced mid-value.
    """
    is_row_structured = isinstance(data, list) and data and isinstance(data[0], list)
    if not is_row_structured:
        rendered = _render_table_data(data)
        full_block = f"{header_block}\n{rendered}"
        if len(full_block) <= remaining_chars:
            return full_block, False
        return "", True  # can't safely partial-render; drop rather than corrupt

    rows = data
    kept_rows: list[list] = []
    for row in rows:
        candidate_rows = kept_rows + [row]
        candidate_text = f"{header_block}\n{_render_rows(candidate_rows)}"
        if len(candidate_text) > remaining_chars:
            break
        kept_rows = candidate_rows

    was_truncated = len(kept_rows) < len(rows)
    if not kept_rows:
        return "", True
    return f"{header_block}\n{_render_rows(kept_rows)}", was_truncated


def render_tables_text(paper_dir: Path, max_chars: int) -> str:
    """Concatenate tables in paper_dir/tables/*.json into a single text
    block, capped at max_chars, truncating at table/row boundaries rather
    than mid-row (see module docstring). Returns '' if no tables exist.
    """
    tables_dir = paper_dir / "tables"
    if not tables_dir.is_dir():
        return ""

    table_paths = sorted(tables_dir.glob("*.json"))
    included_blocks: list[str] = []
    dropped_table_ids: list[str] = []
    truncated_table_ids: list[str] = []
    used_chars = 0
    separator = "\n\n"

    for table_path in table_paths:
        try:
            table = json.loads(table_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        table_id = table.get("id", table_path.stem)
        caption = table.get("caption", "") or "(no caption)"
        page = table.get("page", "?")
        header_block = f"Table {table_id} (page {page}): {caption}"
        data = table.get("data", "")
        full_rendered = f"{header_block}\n{_render_table_data(data)}"

        sep_cost = len(separator) if included_blocks else 0
        remaining = max_chars - used_chars - sep_cost

        if remaining <= 0:
            dropped_table_ids.append(str(table_id))
            continue

        if len(full_rendered) <= remaining:
            included_blocks.append(full_rendered)
            used_chars += sep_cost + len(full_rendered)
            continue

        # Doesn't fit whole -- try row-level truncation instead of a blind
        # character slice that could cut a row in half.
        partial, was_truncated = _truncate_table_by_rows(header_block, data, remaining)
        if partial:
            included_blocks.append(partial)
            used_chars += sep_cost + len(partial)
            if was_truncated:
                truncated_table_ids.append(str(table_id))
        else:
            dropped_table_ids.append(str(table_id))

    if dropped_table_ids or truncated_table_ids:
        logger.warning(
            f"{paper_dir.name}: table budget (max_table_chars={max_chars}) exceeded — "
            f"{len(truncated_table_ids)} table(s) row-truncated {truncated_table_ids}, "
            f"{len(dropped_table_ids)} table(s) dropped entirely {dropped_table_ids}. "
            "Consider raising extraction.max_table_chars in config.yaml if this "
            "paper's tables carry values you need."
        )

    return separator.join(included_blocks)
