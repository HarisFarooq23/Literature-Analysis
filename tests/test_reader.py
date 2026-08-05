from pathlib import Path

from llm_extract.reader import discover_paper_dirs, iter_papers


def test_discover_paper_dirs_finds_all_four(fake_pipeline_output: Path) -> None:
    dirs = discover_paper_dirs(fake_pipeline_output)
    assert len(dirs) == 4


def test_iter_papers_skips_failed_and_duplicate(fake_pipeline_output: Path) -> None:
    papers = list(iter_papers(
        fake_pipeline_output,
        relevant_section_keywords=["experimental", "result"],
        max_input_chars=10_000,
    ))
    # Only the two valid (completed, non-duplicate) papers should remain
    assert len(papers) == 2
    ids = {p.paper_id for p in papers}
    assert "Khan_Fuel_2022_aaaa1111" in ids
    assert "Sani_EnergyFuels_2021_bbbb2222" in ids
    assert "Failed_Paper_2020_cccc3333" not in ids
    assert "Dup_Paper_2020_dddd4444" not in ids


def test_section_keyword_matching_excludes_references(fake_pipeline_output: Path) -> None:
    papers = {p.paper_id: p for p in iter_papers(
        fake_pipeline_output,
        relevant_section_keywords=["experimental", "result"],
        max_input_chars=10_000,
    )}
    khan = papers["Khan_Fuel_2022_aaaa1111"]
    assert "CaO heterogeneous" in khan.text_for_extraction
    assert "Cetane number was 54" in khan.text_for_extraction
    assert "Should be excluded" not in khan.text_for_extraction


def test_falls_back_to_full_text_when_no_sections_dir(fake_pipeline_output: Path) -> None:
    papers = {p.paper_id: p for p in iter_papers(
        fake_pipeline_output,
        relevant_section_keywords=["experimental", "result"],
        max_input_chars=10_000,
    )}
    sani = papers["Sani_EnergyFuels_2021_bbbb2222"]
    assert "FULL TEXT FALLBACK CONTENT" in sani.text_for_extraction


def test_already_done_papers_are_skipped(fake_pipeline_output: Path) -> None:
    papers = list(iter_papers(
        fake_pipeline_output,
        relevant_section_keywords=["experimental", "result"],
        max_input_chars=10_000,
        already_done={"Khan_Fuel_2022_aaaa1111"},
    ))
    assert len(papers) == 1
    assert papers[0].paper_id == "Sani_EnergyFuels_2021_bbbb2222"


def test_max_input_chars_is_respected(fake_pipeline_output: Path) -> None:
    papers = list(iter_papers(
        fake_pipeline_output,
        relevant_section_keywords=["experimental", "result"],
        max_input_chars=20,
        include_tables=False,  # tables have their own, separate, additive budget
    ))
    assert all(len(p.text_for_extraction) <= 20 for p in papers)


def test_tables_are_included_by_default(fake_pipeline_output: Path) -> None:
    papers = {p.paper_id: p for p in iter_papers(
        fake_pipeline_output,
        relevant_section_keywords=["experimental", "result"],
        max_input_chars=10_000,
    )}
    khan = papers["Khan_Fuel_2022_aaaa1111"]
    assert "=== TABLES ===" in khan.text_for_extraction
    assert "Catalyst screening" in khan.text_for_extraction
    assert "CaO" in khan.text_for_extraction and "96" in khan.text_for_extraction


def test_tables_can_be_disabled(fake_pipeline_output: Path) -> None:
    papers = {p.paper_id: p for p in iter_papers(
        fake_pipeline_output,
        relevant_section_keywords=["experimental", "result"],
        max_input_chars=10_000,
        include_tables=False,
    )}
    khan = papers["Khan_Fuel_2022_aaaa1111"]
    assert "=== TABLES ===" not in khan.text_for_extraction
    assert "Catalyst screening" not in khan.text_for_extraction
