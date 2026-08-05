"""
test_tables_truncation.py — tests for the table-truncation rewrite from a
code review: the original render_tables_text() did `text[:max_chars]` across
ALL concatenated tables, which can cut a table off mid-row -- e.g. a 10-
column product-distribution row truncated to 6 columns, silently sent to
the LLM as if it were the complete row. Confirmed as a real risk for SAF
papers with wide multi-catalyst comparison tables (see module docstring in
tables.py for the full rationale).

These tests lock in the fix: truncation happens at whole-table boundaries
first, then whole-row boundaries within an oversized single table, and a
row is never split.
"""

import json
from pathlib import Path

from llm_extract.tables import render_tables_text


def _write_table(paper_dir: Path, table_id: str, table: dict) -> None:
    tables_dir = paper_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    (tables_dir / f"{table_id}.json").write_text(json.dumps(table), encoding="utf-8")


def _row_count_in_output(text: str, table_marker: str) -> int:
    """Count rendered data rows (excluding header/separator/caption lines)
    for the table whose header line contains table_marker."""
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if table_marker in line)
    count = 0
    for line in lines[start + 1:]:
        if line.startswith("Table "):  # next table's header line
            break
        if line.strip() and not set(line.replace("|", "").replace(" ", "")) <= {"-"}:
            if " | " in line:
                count += 1
    return count - 1  # subtract the header row itself


def test_wide_table_truncated_at_row_boundary_never_mid_row(tmp_path: Path):
    """A 10-column table with many rows, budget too small for all rows:
    every row actually included must have the full 10 columns -- never a
    partial row with fewer columns than the header."""
    header = ["catalyst"] + [f"c8c{i}_pct" for i in range(9)]
    rows = [header]
    for i in range(20):
        rows.append([f"cat_{i}"] + [str(i + j) for j in range(9)])

    _write_table(tmp_path, "table_0001", {
        "id": "table_0001", "caption": "Wide product distribution", "page": 5,
        "data": rows,
    })

    text = render_tables_text(tmp_path, max_chars=400)
    assert text != ""
    assert len(text) <= 400

    data_lines = [
        line for line in text.split("\n")
        if " | " in line and not set(line.replace("|", "").replace(" ", "")) <= {"-"}
    ]
    n_columns_expected = len(header)
    for line in data_lines:
        n_columns_actual = len(line.split(" | "))
        assert n_columns_actual == n_columns_expected, (
            f"Row was cut mid-column: expected {n_columns_expected} columns, "
            f"got {n_columns_actual} in line: {line!r}"
        )


def test_multiple_tables_earlier_ones_included_whole_before_truncating_later_ones(tmp_path: Path):
    """When several tables together exceed the budget, whole tables that
    fit are included in full; only the table where the budget runs out
    gets row-truncated (or dropped), never an earlier one that already fit
    comfortably."""
    small_table_rows = [["catalyst", "conversion"], ["Ni", "78"], ["Pt", "88"]]
    _write_table(tmp_path, "table_0001", {
        "id": "table_0001", "caption": "Small summary", "page": 1, "data": small_table_rows,
    })

    big_rows = [["catalyst", "temp", "pressure", "yield"]]
    for i in range(50):
        big_rows.append([f"cat_{i}", str(300 + i), str(10 + i), str(50 + i)])
    _write_table(tmp_path, "table_0002", {
        "id": "table_0002", "caption": "Large screening table", "page": 2, "data": big_rows,
    })

    budget = len("Table table_0001 (page 1): Small summary\ncatalyst | conversion\n--- | ---\nNi | 78\nPt | 88") + 200
    text = render_tables_text(tmp_path, max_chars=budget)

    assert "table_0001" in text
    assert "Ni | 78" in text and "Pt | 88" in text  # small table fully intact
    assert "table_0002" in text  # large table at least partially included (header + some rows)


def test_table_dropped_entirely_logs_warning(tmp_path: Path, caplog):
    """A table that cannot fit even a single row must be dropped (not
    corrupted), and this must be logged, not silent."""
    from loguru import logger
    import sys

    warnings = []
    handler_id = logger.add(lambda msg: warnings.append(msg), level="WARNING")

    rows = [["catalyst", "very_long_column_name_that_takes_lots_of_space"], ["Ni", "78"]]
    _write_table(tmp_path, "table_0001", {
        "id": "table_0001", "caption": "c", "page": 1, "data": rows,
    })
    # Budget too small to fit even the header row.
    text = render_tables_text(tmp_path, max_chars=10)

    logger.remove(handler_id)
    assert text == ""
    assert any("dropped" in str(w) or "table_0001" in str(w) for w in warnings)


def test_raw_string_table_data_is_never_character_sliced(tmp_path: Path):
    """Non-row-structured (raw string / OCR fallback) table data has no
    row boundary to truncate at -- it must be included whole or dropped
    entirely, never character-sliced mid-value (which could fabricate a
    truncated number, e.g. '215' turning into '21')."""
    _write_table(tmp_path, "table_0001", {
        "id": "table_0001", "caption": "c", "page": 1, "data": "BET surface area: 215.4 m2/g",
    })
    # Budget too small for the whole string.
    text = render_tables_text(tmp_path, max_chars=20)
    # Must be empty (dropped), not a truncated fragment like "BET surface are"
    assert text == "" or "BET surface area: 215.4 m2/g" in text


def test_row_structured_table_that_fits_whole_is_unaffected(tmp_path: Path):
    """Sanity check: normal case (table comfortably fits) is completely
    unaffected by the truncation-boundary rewrite."""
    rows = [["catalyst", "conversion"], ["Ni", "78"], ["NiMo", "95"]]
    _write_table(tmp_path, "table_0001", {
        "id": "table_0001", "caption": "Summary", "page": 3, "data": rows,
    })
    text = render_tables_text(tmp_path, max_chars=10_000)
    assert "Ni | 78" in text
    assert "NiMo | 95" in text
    assert "catalyst | conversion" in text
