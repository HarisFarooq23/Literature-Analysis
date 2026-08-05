import json
from pathlib import Path

from llm_extract.tables import render_tables_text


def _write_table(paper_dir: Path, table_id: str, payload: dict) -> None:
    tables_dir = paper_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    (tables_dir / f"{table_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_no_tables_dir_returns_empty_string(tmp_path: Path):
    assert render_tables_text(tmp_path, max_chars=1000) == ""


def test_renders_list_of_lists_as_markdown_table(tmp_path: Path):
    _write_table(tmp_path, "table_0001", {
        "id": "table_0001", "caption": "Catalyst screening results", "page": 4,
        "data": [["Catalyst", "Yield (%)"], ["CaO", "96"], ["MgO", "88"]],
    })
    text = render_tables_text(tmp_path, max_chars=10_000)
    assert "Catalyst screening results" in text
    assert "CaO" in text and "96" in text
    assert "---" in text  # markdown header separator


def test_renders_dict_with_header_and_rows(tmp_path: Path):
    _write_table(tmp_path, "table_0002", {
        "id": "table_0002", "caption": "c", "page": 1,
        "data": {"header": ["A", "B"], "rows": [["x", "1"], ["y", "2"]]},
    })
    text = render_tables_text(tmp_path, max_chars=10_000)
    assert "x" in text and "1" in text


def test_renders_string_data_as_is(tmp_path: Path):
    _write_table(tmp_path, "table_0003", {
        "id": "table_0003", "caption": "OCR fallback", "page": 2,
        "data": "raw unstructured table text",
    })
    text = render_tables_text(tmp_path, max_chars=10_000)
    assert "raw unstructured table text" in text


def test_multiple_tables_are_concatenated_in_sorted_order(tmp_path: Path):
    _write_table(tmp_path, "table_0002", {"id": "table_0002", "caption": "second", "page": 2, "data": "B"})
    _write_table(tmp_path, "table_0001", {"id": "table_0001", "caption": "first", "page": 1, "data": "A"})
    text = render_tables_text(tmp_path, max_chars=10_000)
    assert text.index("first") < text.index("second")


def test_respects_max_chars_budget(tmp_path: Path):
    _write_table(tmp_path, "table_0001", {
        "id": "table_0001", "caption": "c", "page": 1, "data": "X" * 1000,
    })
    text = render_tables_text(tmp_path, max_chars=50)
    assert len(text) <= 50


def test_malformed_json_is_skipped_not_raised(tmp_path: Path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "broken.json").write_text("{not valid json", encoding="utf-8")
    assert render_tables_text(tmp_path, max_chars=1000) == ""
