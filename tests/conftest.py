"""Shared pytest fixtures — build a tiny fake scientific-pdf-pipeline output
directory on disk so reader.py can be tested without any real PDFs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

_METADATA_FIELDS = [
    "id", "title", "authors", "first_author", "journal", "publication_year",
    "doi", "keywords", "abstract", "material_class", "is_review",
    "hash_sha256", "hash_blake3", "file_size", "num_pages", "is_scanned",
    "extraction_method", "processing_status", "duplicate_status",
    "duplicate_of", "input_path", "output_dir", "error_count",
]


def _write_paper(root: Path, folder: str, overrides: dict, sections: dict[str, str], tables: dict | None = None) -> Path:
    paper_dir = root / folder
    paper_dir.mkdir(parents=True)

    row = {f: "" for f in _METADATA_FIELDS}
    row.update({
        "title": "Effect of heterogeneous catalysts on WCO biodiesel properties",
        "first_author": "Khan",
        "journal": "Fuel",
        "publication_year": "2022",
        "doi": "10.1000/example",
        "material_class": "catalyst",
        "is_review": "False",
        "hash_blake3": folder,  # use folder name as a convenient stable id in tests
        "processing_status": "completed",
        "duplicate_status": "False",
    })
    row.update(overrides)

    with open(paper_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_METADATA_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    (paper_dir / "cleaned_full_text.txt").write_text(
        "TITLE: test\n\nFULL TEXT FALLBACK CONTENT", encoding="utf-8"
    )

    if sections:
        sections_dir = paper_dir / "sections"
        sections_dir.mkdir()
        for name, content in sections.items():
            (sections_dir / f"{name}.txt").write_text(content, encoding="utf-8")

    if tables:
        tables_dir = paper_dir / "tables"
        tables_dir.mkdir()
        for table_id, payload in tables.items():
            (tables_dir / f"{table_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    return paper_dir


@pytest.fixture
def fake_pipeline_output(tmp_path: Path) -> Path:
    root = tmp_path / "pipeline_output"
    root.mkdir()

    _write_paper(
        root, "Khan_Fuel_2022_aaaa1111",
        overrides={},
        sections={
            "experimental_section": "Catalyst: CaO heterogeneous, loading 3 wt%, "
                                     "methanol:oil ratio 9:1, 60C, 90 min. FAME yield 96%.",
            "results_and_discussion": "Cetane number was 54. Kinematic viscosity at 40C was 4.8 cSt.",
            "references": "[1] Should be excluded by keyword matching anyway.",
        },
        tables={
            "table_0001": {
                "id": "table_0001", "caption": "Catalyst screening", "page": 5,
                "data": [["Catalyst", "Yield (%)"], ["CaO", "96"], ["MgO", "88"]],
            },
        },
    )
    _write_paper(
        root, "Sani_EnergyFuels_2021_bbbb2222",
        overrides={"first_author": "Sani", "journal": "Energy & Fuels", "publication_year": "2021"},
        sections={},  # no sections dir -> must fall back to cleaned_full_text.txt
    )
    _write_paper(
        root, "Failed_Paper_2020_cccc3333",
        overrides={"processing_status": "failed"},
        sections={},
    )
    _write_paper(
        root, "Dup_Paper_2020_dddd4444",
        overrides={"duplicate_status": "True"},
        sections={},
    )

    return root
