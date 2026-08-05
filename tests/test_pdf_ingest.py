"""Smoke tests for the built-in PDF ingest step."""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")  # pymupdf

from llm_extract.pdf_ingest import ingest_one_pdf, ingest_pdfs


def _make_tiny_pdf(path: Path, text: str = "Experimental\nCatalyst CaO at 60C.\nResults\nYield 96%.") -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


def test_ingest_one_pdf_writes_metadata_and_text(tmp_path: Path) -> None:
    pdf = _make_tiny_pdf(tmp_path / "demo_paper.pdf")
    out_root = tmp_path / "pipeline_output"
    paper_dir = ingest_one_pdf(pdf, out_root)
    assert paper_dir is not None
    assert (paper_dir / "metadata.csv").exists()
    assert (paper_dir / "cleaned_full_text.txt").exists()
    text = (paper_dir / "cleaned_full_text.txt").read_text(encoding="utf-8")
    assert "Catalyst CaO" in text


def test_ingest_pdfs_finds_all_pdfs(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "input_pdfs"
    pdf_dir.mkdir()
    _make_tiny_pdf(pdf_dir / "a.pdf", "Introduction\nHello\nMethods\nDo stuff")
    _make_tiny_pdf(pdf_dir / "b.pdf", "Abstract\nWorld\nResults\nOk")
    out_root = tmp_path / "pipeline_output"
    done = ingest_pdfs(pdf_dir, out_root)
    assert len(done) == 2
