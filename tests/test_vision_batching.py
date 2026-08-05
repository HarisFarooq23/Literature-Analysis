"""
test_vision_batching.py — unit tests for vision.py's single-call-per-paper
batching (see vision.py's module docstring for why this replaced N
sequential per-image calls). All ollama.Client interaction is mocked --
these tests never touch a network or a real vision model.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import ollama
import pytest

from llm_extract.vision import extract_figure_insights


def _make_paper_dir(tmp_path: Path, entries: list[dict]) -> Path:
    paper_dir = tmp_path / "paper1"
    figures_dir = paper_dir / "figures"
    figures_dir.mkdir(parents=True)
    (figures_dir / "figures_index.json").write_text(json.dumps(entries), encoding="utf-8")
    for e in entries:
        (figures_dir / e["image"]).write_bytes(b"fake-png-bytes")
    return paper_dir


def _entries(n: int) -> list[dict]:
    return [
        {"id": f"fig{i+1}", "image": f"fig{i+1}.png", "caption": f"Caption {i+1}"}
        for i in range(n)
    ]


def test_no_figures_index_returns_empty_string(tmp_path):
    paper_dir = tmp_path / "no_figures"
    paper_dir.mkdir()
    assert extract_figure_insights(paper_dir, "gemma3:12b", "http://localhost:11434", {}) == ""


def test_unreadable_index_returns_empty_string_not_crash(tmp_path):
    paper_dir = tmp_path / "bad_index"
    figures_dir = paper_dir / "figures"
    figures_dir.mkdir(parents=True)
    (figures_dir / "figures_index.json").write_text("{not valid json", encoding="utf-8")
    assert extract_figure_insights(paper_dir, "gemma3:12b", "http://localhost:11434", {}) == ""


def test_all_figures_sent_in_a_single_generate_call(tmp_path, mocker):
    """The core fix: N figures must produce exactly ONE ollama.Client.generate
    call (all images attached together), not N separate calls."""
    entries = _entries(3)
    paper_dir = _make_paper_dir(tmp_path, entries)

    mock_client = MagicMock()
    mock_client.generate.return_value = {
        "response": json.dumps({
            "figures": [
                {"figure_id": "fig1", "description": "Conversion vs temperature, rising trend."},
                {"figure_id": "fig2", "description": "Schematic diagram, no quantitative data."},
                {"figure_id": "fig3", "description": "Selectivity bar chart, 80-95% range."},
            ]
        })
    }
    mocker.patch("llm_extract.vision.ollama.Client", return_value=mock_client)

    result = extract_figure_insights(paper_dir, "gemma3:12b", "http://localhost:11434", {}, max_figures=5)

    assert mock_client.generate.call_count == 1
    call_kwargs = mock_client.generate.call_args.kwargs
    assert len(call_kwargs["images"]) == 3  # all three images in one call

    assert "Figure fig1" in result
    assert "Figure fig2" in result
    assert "Figure fig3" in result
    assert "rising trend" in result
    assert "lower-confidence" in result  # advisory framing preserved


def test_respects_max_figures_cap(tmp_path, mocker):
    entries = _entries(8)
    paper_dir = _make_paper_dir(tmp_path, entries)

    mock_client = MagicMock()
    mock_client.generate.return_value = {
        "response": json.dumps({
            "figures": [{"figure_id": f"fig{i+1}", "description": f"desc{i+1}"} for i in range(4)]
        })
    }
    mocker.patch("llm_extract.vision.ollama.Client", return_value=mock_client)

    extract_figure_insights(paper_dir, "gemma3:12b", "http://localhost:11434", {}, max_figures=4)

    assert mock_client.generate.call_count == 1
    assert len(mock_client.generate.call_args.kwargs["images"]) == 4  # capped, not all 8


def test_connection_error_propagates_within_retry_loop_then_degrades_to_empty(tmp_path, mocker):
    """A genuinely unreachable server must not crash the run: after
    exhausting retries, the paper degrades to no figure context."""
    entries = _entries(1)
    paper_dir = _make_paper_dir(tmp_path, entries)

    mock_client = MagicMock()
    mock_client.generate.side_effect = ConnectionError("server down")
    mocker.patch("llm_extract.vision.ollama.Client", return_value=mock_client)
    mocker.patch("llm_extract.vision.time.sleep")  # don't actually sleep in tests

    result = extract_figure_insights(
        paper_dir, "gemma3:12b", "http://localhost:11434", {}, max_retries=2, retry_wait_seconds=0,
    )
    assert result == ""
    assert mock_client.generate.call_count == 2  # retried max_retries times, then gave up


def test_response_error_falls_back_to_format_json(tmp_path, mocker):
    """Server actively rejects schema-constrained generation -> fall back
    to format='json' within the SAME attempt, matching ollama_client.py's
    established fallback contract."""
    entries = _entries(1)
    paper_dir = _make_paper_dir(tmp_path, entries)

    mock_client = MagicMock()
    fallback_response = {"response": json.dumps({"figures": [{"figure_id": "fig1", "description": "ok"}]})}
    mock_client.generate.side_effect = [
        ollama.ResponseError("schema not supported", status_code=400),
        fallback_response,
    ]
    mocker.patch("llm_extract.vision.ollama.Client", return_value=mock_client)

    result = extract_figure_insights(paper_dir, "gemma3:12b", "http://localhost:11434", {})

    assert mock_client.generate.call_count == 2
    assert "Figure fig1: ok" in result
    # second call must have used the unconstrained format
    assert mock_client.generate.call_args_list[1].kwargs["format"] == "json"


def test_markdown_fenced_response_is_parsed(tmp_path, mocker):
    entries = _entries(1)
    paper_dir = _make_paper_dir(tmp_path, entries)

    mock_client = MagicMock()
    mock_client.generate.return_value = {
        "response": "```json\n" + json.dumps({
            "figures": [{"figure_id": "fig1", "description": "fenced description"}]
        }) + "\n```"
    }
    mocker.patch("llm_extract.vision.ollama.Client", return_value=mock_client)

    result = extract_figure_insights(paper_dir, "gemma3:12b", "http://localhost:11434", {})
    assert "fenced description" in result


def test_missing_image_files_are_skipped_not_fatal(tmp_path, mocker):
    """One figure's PNG referenced in the index but absent on disk should
    be silently skipped, not crash the whole paper's vision pass."""
    entries = _entries(2)
    paper_dir = _make_paper_dir(tmp_path, entries)
    (paper_dir / "figures" / "fig2.png").unlink()  # simulate missing file

    mock_client = MagicMock()
    mock_client.generate.return_value = {
        "response": json.dumps({"figures": [{"figure_id": "fig1", "description": "solo figure"}]})
    }
    mocker.patch("llm_extract.vision.ollama.Client", return_value=mock_client)

    result = extract_figure_insights(paper_dir, "gemma3:12b", "http://localhost:11434", {})
    assert len(mock_client.generate.call_args.kwargs["images"]) == 1
    assert "solo figure" in result


def test_malformed_batch_response_exhausts_retries_and_degrades(tmp_path, mocker):
    entries = _entries(1)
    paper_dir = _make_paper_dir(tmp_path, entries)

    mock_client = MagicMock()
    mock_client.generate.return_value = {"response": "not json at all"}
    mocker.patch("llm_extract.vision.ollama.Client", return_value=mock_client)
    mocker.patch("llm_extract.vision.time.sleep")

    result = extract_figure_insights(
        paper_dir, "gemma3:12b", "http://localhost:11434", {}, max_retries=2, retry_wait_seconds=0,
    )
    assert result == ""
    assert mock_client.generate.call_count == 2
