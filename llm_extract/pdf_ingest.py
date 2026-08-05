"""
pdf_ingest.py — turn raw PDFs into the on-disk paper folders that
reader.py already understands (metadata.csv + cleaned_full_text.txt +
optional sections/ tables/ figures/).

This replaces the separate scientific-pdf-pipeline for simple local use:
drop PDFs in a folder, run `llm-extract from-pdfs`, get structured
extraction without a second project.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from loguru import logger

_METADATA_FIELDS = [
    "id", "title", "authors", "first_author", "journal", "publication_year",
    "doi", "keywords", "abstract", "material_class", "is_review",
    "hash_sha256", "hash_blake3", "file_size", "num_pages", "is_scanned",
    "extraction_method", "processing_status", "duplicate_status",
    "duplicate_of", "input_path", "output_dir", "error_count",
]

# Common paper section headings (case-insensitive, start-of-line-ish).
_SECTION_HEADING_RE = re.compile(
    r"(?im)^(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(abstract|introduction|experimental(?:\s+section)?|materials?\s+and\s+methods?"
    r"|methods?(?:\s+and\s+materials?)?|procedure|synthesis|preparation|"
    r"catalyst(?:\s+preparation)?|characterization|reaction\s+conditions?|"
    r"results?(?:\s+and\s+discussion)?|discussion|performance|conclusions?|"
    r"references|acknowledg(?:e)?ments?)\s*$"
)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_stem(name: str, max_len: int = 60) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("._-")
    return (cleaned or "paper")[:max_len]


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _guess_title(doc, full_text: str, pdf_path: Path) -> str:
    meta_title = (doc.metadata or {}).get("title") or ""
    if meta_title.strip() and meta_title.strip().lower() not in {"untitled", "title"}:
        return meta_title.strip()[:300]
    for line in full_text.splitlines()[:40]:
        line = line.strip()
        if 15 <= len(line) <= 250 and not line.lower().startswith("http"):
            return line
    return pdf_path.stem.replace("_", " ").replace("-", " ")


def _split_sections(full_text: str) -> dict[str, str]:
    """Split full text on common scientific headings. Returns {} if nothing useful."""
    matches = list(_SECTION_HEADING_RE.finditer(full_text))
    if len(matches) < 2:
        return {}

    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        heading = match.group(1).strip().lower()
        body = full_text[start:end].strip()
        if not body:
            continue
        key = _safe_stem(heading.replace(" ", "_"), max_len=80)
        # Keep the longest body if the same heading appears twice.
        if key not in sections or len(body) > len(sections[key]):
            sections[key] = body
    return sections


def _extract_with_pymupdf(pdf_path: Path, paper_dir: Path) -> dict:
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    page_texts: list[str] = []
    tables_dir = paper_dir / "tables"
    figures_dir = paper_dir / "figures"
    table_count = 0
    figure_entries: list[dict] = []

    for page_index, page in enumerate(doc):
        page_texts.append(page.get_text("text"))

        # Tables (pymupdf table finder; skip quietly if unavailable).
        try:
            finder = page.find_tables()
            for t_i, table in enumerate(getattr(finder, "tables", []) or []):
                try:
                    data = table.extract()
                except Exception:
                    continue
                if not data or len(data) < 2:
                    continue
                table_count += 1
                table_id = f"table_{table_count:04d}"
                tables_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "id": table_id,
                    "caption": f"Table on page {page_index + 1}",
                    "page": page_index + 1,
                    "data": [[("" if c is None else str(c)) for c in row] for row in data],
                }
                (tables_dir / f"{table_id}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception as exc:
            logger.debug(f"Table find skipped on page {page_index + 1} of {pdf_path.name}: {exc}")

        # Embedded images as figures (cap later in vision; store a generous set).
        for img_i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n >= 5:  # CMYK etc.
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                # Skip tiny icons / bullets.
                if pix.width < 80 or pix.height < 80:
                    continue
                figures_dir.mkdir(parents=True, exist_ok=True)
                fig_id = f"fig_p{page_index + 1}_{img_i + 1}"
                image_name = f"{fig_id}.png"
                pix.save(figures_dir / image_name)
                figure_entries.append({
                    "id": fig_id,
                    "image": image_name,
                    "caption": f"Embedded image from page {page_index + 1}",
                    "page": page_index + 1,
                })
            except Exception as exc:
                logger.debug(f"Image extract skipped ({pdf_path.name} p{page_index + 1}): {exc}")

    full_text = "\n\n".join(t.strip() for t in page_texts if t and t.strip())
    is_scanned = len(full_text.strip()) < 200
    title = _guess_title(doc, full_text, pdf_path)
    num_pages = doc.page_count
    doc.close()

    if figure_entries:
        (figures_dir / "figures_index.json").write_text(
            json.dumps(figure_entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return {
        "full_text": full_text,
        "title": title,
        "num_pages": num_pages,
        "is_scanned": is_scanned,
        "table_count": table_count,
        "figure_count": len(figure_entries),
    }


def ingest_one_pdf(pdf_path: Path, pipeline_output_root: Path) -> Path | None:
    """Parse one PDF into a paper folder under pipeline_output_root.
    Returns the paper directory, or None on failure."""
    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        logger.warning(f"Skipping non-PDF: {pdf_path}")
        return None

    file_hash = _file_hash(pdf_path)
    short = file_hash[:8]
    folder_name = f"{_safe_stem(pdf_path.stem)}_{short}"
    paper_dir = pipeline_output_root / folder_name

    # Resume: already ingested this exact file.
    meta_path = paper_dir / "metadata.csv"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8", newline="") as f:
                row = next(csv.DictReader(f), {})
            if row.get("hash_sha256") == file_hash and (paper_dir / "cleaned_full_text.txt").exists():
                logger.info(f"Already ingested, skipping: {pdf_path.name} -> {folder_name}")
                return paper_dir
        except Exception:
            pass

    paper_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ingesting PDF: {pdf_path.name} -> {folder_name}")

    try:
        extracted = _extract_with_pymupdf(pdf_path, paper_dir)
    except Exception as exc:
        logger.error(f"Failed to parse {pdf_path.name}: {exc}")
        row = {f: "" for f in _METADATA_FIELDS}
        row.update({
            "id": folder_name,
            "title": pdf_path.stem,
            "first_author": "Unknown",
            "hash_sha256": file_hash,
            "hash_blake3": file_hash,
            "file_size": str(pdf_path.stat().st_size),
            "processing_status": "failed",
            "duplicate_status": "False",
            "input_path": str(pdf_path),
            "output_dir": str(paper_dir),
            "error_count": "1",
            "extraction_method": "pymupdf",
            "material_class": "unknown",
            "is_review": "False",
        })
        with open(paper_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_METADATA_FIELDS)
            writer.writeheader()
            writer.writerow(row)
        return None

    full_text = extracted["full_text"]
    (paper_dir / "cleaned_full_text.txt").write_text(full_text or "", encoding="utf-8")

    sections = _split_sections(full_text)
    if sections:
        sections_dir = paper_dir / "sections"
        sections_dir.mkdir(parents=True, exist_ok=True)
        for name, body in sections.items():
            (sections_dir / f"{name}.txt").write_text(body, encoding="utf-8")

    status = "completed" if full_text.strip() else "failed"
    if extracted["is_scanned"]:
        logger.warning(
            f"{pdf_path.name} looks scanned / image-only (very little text). "
            "OCR is not included in this simple pipeline — use a text PDF if possible."
        )

    row = {f: "" for f in _METADATA_FIELDS}
    row.update({
        "id": folder_name,
        "title": extracted["title"],
        "first_author": "Unknown",
        "journal": "",
        "publication_year": "",
        "doi": "",
        "abstract": "",
        "material_class": "unknown",
        "is_review": "False",
        "hash_sha256": file_hash,
        "hash_blake3": file_hash,
        "file_size": str(pdf_path.stat().st_size),
        "num_pages": str(extracted["num_pages"]),
        "is_scanned": str(extracted["is_scanned"]),
        "extraction_method": "pymupdf",
        "processing_status": status,
        "duplicate_status": "False",
        "input_path": str(pdf_path),
        "output_dir": str(paper_dir),
        "error_count": "0" if status == "completed" else "1",
    })
    with open(paper_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_METADATA_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    logger.info(
        f"  pages={extracted['num_pages']} tables={extracted['table_count']} "
        f"figures={extracted['figure_count']} chars={len(full_text)} status={status}"
    )
    return paper_dir if status == "completed" else None


def ingest_pdfs(pdfs_input_dir: Path, pipeline_output_root: Path) -> list[Path]:
    """Ingest every *.pdf under pdfs_input_dir (non-recursive)."""
    pdfs_input_dir = Path(pdfs_input_dir)
    pipeline_output_root = Path(pipeline_output_root)
    pipeline_output_root.mkdir(parents=True, exist_ok=True)

    if not pdfs_input_dir.is_dir():
        raise FileNotFoundError(
            f"PDF input folder does not exist: {pdfs_input_dir}. "
            "Create it and put your .pdf files inside."
        )

    pdfs = sorted(pdfs_input_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF files found in {pdfs_input_dir}. "
            "Copy your papers there (file extension must be .pdf)."
        )

    logger.info(f"Found {len(pdfs)} PDF(s) in {pdfs_input_dir}")
    done: list[Path] = []
    for pdf in pdfs:
        paper_dir = ingest_one_pdf(pdf, pipeline_output_root)
        if paper_dir is not None:
            done.append(paper_dir)
    logger.info(f"PDF ingest complete: {len(done)}/{len(pdfs)} paper folder(s) ready")
    return done
