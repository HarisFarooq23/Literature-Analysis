import json
from pathlib import Path

from llm_extract.formulas import render_formulas_text


def _write_formulas(paper_dir: Path, records: list[dict]) -> None:
    formulas_dir = paper_dir / "formulas"
    formulas_dir.mkdir(exist_ok=True)
    (formulas_dir / "formulas.json").write_text(json.dumps(records), encoding="utf-8")


def test_no_formulas_dir_returns_empty_string(tmp_path: Path):
    assert render_formulas_text(tmp_path, max_chars=1000) == ""


def test_no_formulas_file_returns_empty_string(tmp_path: Path):
    (tmp_path / "formulas").mkdir()
    assert render_formulas_text(tmp_path, max_chars=1000) == ""


def test_empty_list_returns_empty_string(tmp_path: Path):
    _write_formulas(tmp_path, [])
    assert render_formulas_text(tmp_path, max_chars=1000) == ""


def test_renders_high_confidence_latex_formula(tmp_path: Path):
    _write_formulas(tmp_path, [
        {"id": "formula_0001", "latex": r"\Delta G = \Delta H - T\Delta S",
         "is_latex": True, "context": "the Gibbs free energy change was calculated as",
         "page": 5, "confidence": 0.98},
    ])
    text = render_formulas_text(tmp_path, max_chars=10_000)
    assert "formula_0001" in text
    assert r"\Delta G = \Delta H - T\Delta S" in text
    assert "LaTeX" in text
    assert "Gibbs free energy" in text


def test_low_confidence_formula_is_labelled_as_such(tmp_path: Path):
    _write_formulas(tmp_path, [
        {"id": "formula_0002", "latex": "k = A exp(-Ea/RT)",
         "is_latex": False, "context": "", "page": 6, "confidence": 0.4},
    ])
    text = render_formulas_text(tmp_path, max_chars=10_000)
    assert "low-confidence" in text
    assert "k = A exp(-Ea/RT)" in text


def test_formula_with_no_latex_text_is_skipped(tmp_path: Path):
    _write_formulas(tmp_path, [
        {"id": "formula_0003", "latex": "", "is_latex": False, "context": "x", "page": 1},
        {"id": "formula_0004", "latex": "E = mc^2", "is_latex": True, "context": "", "page": 2},
    ])
    text = render_formulas_text(tmp_path, max_chars=10_000)
    assert "formula_0003" not in text
    assert "E = mc^2" in text


def test_multiple_formulas_are_concatenated(tmp_path: Path):
    _write_formulas(tmp_path, [
        {"id": "formula_0001", "latex": "A", "is_latex": True, "context": "", "page": 1},
        {"id": "formula_0002", "latex": "B", "is_latex": True, "context": "", "page": 2},
    ])
    text = render_formulas_text(tmp_path, max_chars=10_000)
    assert text.index("formula_0001") < text.index("formula_0002")


def test_respects_max_chars_budget(tmp_path: Path):
    _write_formulas(tmp_path, [
        {"id": "formula_0001", "latex": "X" * 1000, "is_latex": True, "context": "", "page": 1},
    ])
    text = render_formulas_text(tmp_path, max_chars=50)
    assert len(text) <= 50


def test_malformed_json_returns_empty_string(tmp_path: Path):
    formulas_dir = tmp_path / "formulas"
    formulas_dir.mkdir()
    (formulas_dir / "formulas.json").write_text("{not valid json", encoding="utf-8")
    assert render_formulas_text(tmp_path, max_chars=1000) == ""
