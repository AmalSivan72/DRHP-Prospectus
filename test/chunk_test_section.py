
import fitz
import json
import os
import pickle
import re
import logging
import spacy
from typing import List, Dict, Any, Optional


nlp = spacy.load("en_core_web_sm")

def chunk_text_spacy(text: str, chunk_size: int = 2000, overlap_sentences: int = 2) -> List[str]:
    """Split text into sentence-based chunks using spaCy, with overlap."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current_chunk:
            # finalize current chunk
            chunks.append(" ".join(current_chunk))
            # start new chunk with overlap (last N sentences)
            current_chunk = current_chunk[-overlap_sentences:] + [sent]
            current_len = sum(len(s) for s in current_chunk)
        else:
            current_chunk.append(sent)
            current_len += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def convert_toc_list_to_hierarchy(toc_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert a flat TOC list into a hierarchical dict structure."""
    hierarchy: Dict[str, Any] = {}
    for entry in toc_entries:
        section = entry["title"]
        page = int(entry.get("page", 1))
        if "subsections" in entry:
            hierarchy[section] = {
                "subsections": {
                    sub["title"]: {
                        "pdf_start_page": int(sub.get("page", page)),
                        "pdf_end_page": int(sub.get("page", page)),
                    } for sub in entry["subsections"]
                }
            }
        else:
            hierarchy[section] = {
                "pdf_start_page": page,
                "pdf_end_page": page
            }
    return hierarchy

def derive_section_page_range(section_data: Dict[str, Any]) -> Optional[tuple]:
    """
    Return (start_page, end_page) for a section.
    Prefer explicit section range if present, else derive from subsections.
    """
    if "pdf_start_page" in section_data and "pdf_end_page" in section_data:
        return int(section_data["pdf_start_page"]), int(section_data["pdf_end_page"])
    subsections = section_data.get("subsections", {})
    if subsections:
        starts = []
        ends = []
        for sub in subsections.values():
            if "pdf_start_page" in sub and "pdf_end_page" in sub:
                starts.append(int(sub["pdf_start_page"]))
                ends.append(int(sub["pdf_end_page"]))
        if starts and ends:
            return min(starts), max(ends)
    return None


DOT_HEAVY_LINE_PATTERN = re.compile(r"(\s*\.\s*){5,}")

def is_dot_heavy(text: str, threshold: float = 0.5) -> bool:
    """Heuristic: proportion of lines containing long dot sequences."""
    lines = text.splitlines()
    if not lines:
        return False
    dot_lines = sum(1 for line in lines if DOT_HEAVY_LINE_PATTERN.search(line))
    return (dot_lines / len(lines)) > threshold

def alpha_density(text: str) -> float:
    """Alpha-numeric density heuristic; useful to detect mostly-non-text pages."""
    if not text:
        return 0.0
    alpha_num = sum(ch.isalnum() for ch in text)
    return alpha_num / max(1, len(text))

def clean_text(text: str) -> str:
    """Remove dot leaders and 3+ dots that often come from TOC pages."""
    text = re.sub(r"(\s*\.\s*){5,}", " ", text)  # Remove long dot sequences
    text = re.sub(r"\.{3,}", " ", text)          # Replace 3+ dots with space
    return text

def extract_text_from_range(doc: fitz.Document, start_page: int, end_page: int) -> str:
    """Extract text from a 1-indexed inclusive page range."""
    text_parts: List[str] = []
    for i in range(start_page, end_page + 1):
        if 1 <= i <= len(doc):
            page_text = doc[i - 1].get_text()
            if not page_text.strip():
                logging.warning(f"⚠️ Page {i} has no extractable text")
            text_parts.append(page_text)
        else:
            logging.warning(f"⏭️ Skipping page {i}: out of bounds (doc has {len(doc)} pages)")
    return "".join(text_parts)


def chunk_pdf_by_toc(
    pdf_path: str,
    toc_json_path: str,
    output_pkl_path: str,
    chunk_size: int = 2000,
    overlap_sentences: int = 2,
    create_section_level_chunks: bool = True,
    prepend_titles: bool = True,
    skip_dot_heavy: bool = False,           # default False: clean instead of skipping
    dot_heavy_threshold: float = 0.5,
    min_alpha_density: float = 0.02,        # filter extremely low-text extracts
) -> List[Dict[str, Any]]:

    logging.info(f"📄 Starting chunking for PDF: {pdf_path}")

    # Load TOC structure
    with open(toc_json_path, "r", encoding="utf-8") as f:
        toc_entries = json.load(f)

    if isinstance(toc_entries, dict) and "hierarchy" in toc_entries:
        hierarchy = toc_entries["hierarchy"]
    elif isinstance(toc_entries, list):
        hierarchy = convert_toc_list_to_hierarchy(toc_entries)
    elif isinstance(toc_entries, dict):
        hierarchy = toc_entries
    else:
        logging.error("Unsupported TOC format: must be a list or dict")
        raise ValueError("Unsupported TOC format: must be a list or dict")

    doc = fitz.open(pdf_path)
    all_chunks: List[Dict[str, Any]] = []

    def process_and_chunk(
        section: str,
        subsection: str,
        start_page: int,
        end_page: int,
        granularity: str,
    ) -> None:
        """Extract, clean, chunk, and append metadata."""
        logging.info(f"➡️ Processing {granularity} '{section}' → '{subsection}' pages {start_page}-{end_page}")

        raw_text = extract_text_from_range(doc, start_page, end_page)
        if not raw_text.strip():
            logging.warning(f"⚠️ Skipping {section} → {subsection}: no extractable text in range {start_page}–{end_page}")
            return

        # Optional dot-heavy guard
        if skip_dot_heavy and is_dot_heavy(raw_text, threshold=dot_heavy_threshold):
            logging.info(f"⏭️ Skipping dot-heavy range: {section} → {subsection} ({start_page}–{end_page})")
            return

        # Clean and simple quality check
        text = clean_text(raw_text)
        if alpha_density(text) < min_alpha_density:
            logging.info(f"⏭️ Skipping low alpha-density text: {section} → {subsection} ({start_page}–{end_page})")
            return

        # Prepend titles so queries on headings retrieve the first chunk with the intro paragraph
        if prepend_titles:
            title_line = section if (subsection in ["FULL", "NONE"]) else f"{section} — {subsection}"
            text = f"{title_line}\n\n{text}"

        # Chunk sentence-wise with overlap
        chunks = chunk_text_spacy(text, chunk_size=chunk_size, overlap_sentences=overlap_sentences)

        # Persist chunks with metadata
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{section}__{subsection}__{idx}",
                "section": section,
                "subsection": subsection,
                "granularity": granularity,    # "section" or "subsection"
                "chunk_index": idx,
                "pdf_start_page": start_page,
                "pdf_end_page": end_page,
                "char_count": len(chunk),
                "text": chunk,
                "source": os.path.basename(pdf_path),
            })

        logging.info(f"✅ Created {len(chunks)} chunks for {section} → {subsection} [{granularity}]")

    # Iterate sections
    for section, data in hierarchy.items():
        subsections = data.get("subsections", {})

        # 1) SECTION-LEVEL CHUNKS (FULL)
        if create_section_level_chunks:
            sec_range = derive_section_page_range(data)
            if sec_range is not None:
                sec_start, sec_end = sec_range
                process_and_chunk(section, "FULL", int(sec_start), int(sec_end), granularity="section")
            else:
                logging.warning(f"⚠️ Skipping section-level chunks for '{section}': no page range available")

        # 2) SUBSECTION-LEVEL CHUNKS
        if subsections:
            for subsection, subdata in subsections.items():
                if "pdf_start_page" in subdata and "pdf_end_page" in subdata:
                    process_and_chunk(
                        section,
                                               subsection,
                        int(subdata["pdf_start_page"]),
                        int(subdata["pdf_end_page"]),
                        granularity="subsection",
                    )
                else:
                    logging.error(f"❌ Skipping {section} → {subsection}: missing page range")
        else:
            # No subsections; if not producing FULL above, ensure at least one chunk
            if not create_section_level_chunks and "pdf_start_page" in data and "pdf_end_page" in data:
                process_and_chunk(
                    section, "FULL", int(data["pdf_start_page"]), int(data["pdf_end_page"]), granularity="section"
                )

    # Save to pickle
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(all_chunks, f)

    logging.info(
        f"✅ Saved {len(all_chunks)} chunks to {output_pkl_path} "
        f"(chunk_size={chunk_size}, overlap_sentences={overlap_sentences}, prepend_titles={prepend_titles})"
    )
