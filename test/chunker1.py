
import fitz
import json
import os
import pickle
import re
import logging
import spacy
from typing import List, Dict, Any, Optional

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

def chunk_text_spacy(text: str, chunk_size: int = 2000, overlap_sentences: int = 2) -> List[str]:
    """Split text into sentence-based chunks using spaCy, with overlap."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    chunks = []
    current_chunk = []
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
    """
    Normalize a TOC list to a hierarchy that includes:
    - per-section overall page range (derived from subsections if present)
    - per-subsection page ranges
    """
    hierarchy: Dict[str, Any] = {}
    for entry in toc_entries:
        section = entry["title"]
        page = int(entry["page"])
        if "subsections" in entry and entry["subsections"]:
            subs = {
                sub["title"]: {
                    "pdf_start_page": int(sub["page"]),
                    "pdf_end_page": int(sub["page"]),
                }
                for sub in entry["subsections"]
            }
            # derive section-level page range from subsections
            sub_pages = [int(sub["page"]) for sub in entry["subsections"]]
            hierarchy[section] = {
                "pdf_start_page": min(sub_pages),   # overall section start
                "pdf_end_page": max(sub_pages),     # overall section end
                "subsections": subs,
            }
        else:
            # no subsections: section-level is just the given page
            hierarchy[section] = {
                "pdf_start_page": page,
                "pdf_end_page": page,
                "subsections": {},  # keep uniform shape
            }
    return hierarchy


def is_dot_heavy(text: str) -> bool:
    dot_lines = sum(1 for line in text.splitlines() if re.search(r"(\s*\.\s*){5,}", line))
    total_lines = len(text.splitlines())
    return total_lines > 0 and dot_lines / total_lines > 0.5


def clean_text(text: str) -> str:
    text = re.sub(r"(\s*\.\s*){5,}", " ", text)  # Remove long dot sequences
    text = re.sub(r"\.{3,}", " ", text)          # Replace 3+ dots with space
    return text


def chunk_pdf_by_toc(
    pdf_path: str,
    toc_json_path: str,
    output_pkl_path: str,
    chunk_size: int = 2000,
    overlap_sentences: int = 2
) -> List[Dict[str, Any]]:

    logging.info(f"📄 Starting chunking for PDF: {pdf_path}")

    # Strict skip: if file exists, don't re-chunk
    if os.path.exists(output_pkl_path):
        logging.info(f"🔁 Output file already exists: {output_pkl_path}. Skipping chunking completely.")
        return []  # or load and return pickle if downstream expects chunks

    # Load TOC
    with open(toc_json_path, "r", encoding="utf-8") as f:
        toc_entries = json.load(f)

    # Normalize hierarchy
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

    def extract_text_range(start_page: int, end_page: int, section: str, subsection: Optional[str]) -> Optional[str]:
        text = ""
        for i in range(start_page, end_page + 1):
            if 1 <= i <= len(doc):
                page_text = doc[i - 1].get_text()
                if not page_text.strip():
                    logging.warning(f"⚠️ Page {i} in {section} → {subsection or 'FULL'} has no extractable text")
                text += page_text
            else:
                logging.warning(f"⏭️ Skipping page {i}: out of bounds (doc has {len(doc)} pages)")
        if not text.strip():
            logging.warning(f"⚠️ Skipping {section} → {subsection or 'FULL'}: no extractable text in range {start_page}–{end_page}")
            return None
        if is_dot_heavy(text):
            logging.info(f"⏭️ Skipping dot-heavy section: {section} ({start_page}–{end_page})")
            return None
        return clean_text(text)

    def emit_chunks(section: str, subsection: Optional[str], start_page: int, end_page: int):
        label = subsection or "FULL"
        logging.info(f"➡️ Processing section '{section}' subsection '{label}' pages {start_page}-{end_page}")
        text = extract_text_range(start_page, end_page, section, subsection)
        if text is None:
            return
        chunks = chunk_text_spacy(text, chunk_size=chunk_size, overlap_sentences=overlap_sentences)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{section}__{label}__{idx}",
                "section": section,
                "subsection": label,
                "chunk_index": idx,
                "pdf_start_page": start_page,
                "pdf_end_page": end_page,
                "char_count": len(chunk),
                "text": chunk,
                "source": os.path.basename(pdf_path),
            })
        logging.info(f"✅ Created {len(chunks)} chunks for {section} → {label}")

    # --- Main loop: produce BOTH section-level (FULL) and subsection-level chunks ---
    for section, data in hierarchy.items():
        # 1) Section-level FULL chunks (always emit)
        sec_start = int(data.get("pdf_start_page", 1))
        sec_end = int(data.get("pdf_end_page", sec_start))
        emit_chunks(section, None, sec_start, sec_end)

        # 2) Subsection-level chunks (if present)
        subsections = data.get("subsections", {}) or {}
        for subsection, subdata in subsections.items():
            sub_start = int(subdata["pdf_start_page"])
            sub_end = int(subdata["pdf_end_page"])
            emit_chunks(section, subsection, sub_start, sub_end)

    # Save
    os.makedirs(os.path.dirname(output_pkl_path) or ".", exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(all_chunks, f)

    logging.info(f"✅ Saved {len(all_chunks)} chunks to {output_pkl_path} with sentence-based overlap={overlap_sentences}")
    return all_chunks
