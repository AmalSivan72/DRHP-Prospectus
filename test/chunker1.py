import fitz
import json
import os
import pickle
import re
import logging
import spacy
from typing import List, Dict, Any

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
    hierarchy = {}
    for entry in toc_entries:
        section = entry["title"]
        page = int(entry["page"])
        if "subsections" in entry:
            hierarchy[section] = {
                "subsections": {
                    sub["title"]: {
                        "pdf_start_page": int(sub["page"]),
                        "pdf_end_page": int(sub["page"])
                    } for sub in entry["subsections"]
                }
            }
        else:
            hierarchy[section] = {
                "pdf_start_page": page,
                "pdf_end_page": page
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

    with open(toc_json_path, "r", encoding="utf-8") as f:
        toc_entries = json.load(f)

    if isinstance(toc_entries, dict) and "hierarchy" in toc_entries:
        toc_entries = toc_entries["hierarchy"]

    if isinstance(toc_entries, list):
        hierarchy = convert_toc_list_to_hierarchy(toc_entries)
    elif isinstance(toc_entries, dict):
        hierarchy = toc_entries
    else:
        logging.error("Unsupported TOC format: must be a list or dict")
        raise ValueError("Unsupported TOC format: must be a list or dict")

    doc = fitz.open(pdf_path)
    all_chunks = []

    def process_chunk(section: str, subsection: str, subdata: Dict[str, Any]):
        if "pdf_start_page" not in subdata or "pdf_end_page" not in subdata:
            logging.error(f"❌ Skipping {section} → {subsection or 'FULL'}: missing page range")
            return

        start_page = subdata["pdf_start_page"]
        end_page = subdata["pdf_end_page"]
        text = ""

        logging.info(f"➡️ Processing section '{section}' subsection '{subsection or 'FULL'}' pages {start_page}-{end_page}")

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
            return

        if is_dot_heavy(text):
            logging.info(f"⏭️ Skipping dot-heavy section: {section} ({start_page}–{end_page})")
            return

        text = clean_text(text)
        chunks = chunk_text_spacy(text, chunk_size=chunk_size, overlap_sentences=overlap_sentences)

        for idx, chunk in enumerate(chunks):
            chunk_meta = {
                "chunk_id": f"{section}__{subsection or 'FULL'}__{idx}",
                "section": section,
                "subsection": subsection or "FULL",
                "chunk_index": idx,
                "pdf_start_page": start_page,
                "pdf_end_page": end_page,
                "char_count": len(chunk),
                "text": chunk,
                "source": os.path.basename(pdf_path),
            }
            all_chunks.append(chunk_meta)

        logging.info(f"✅ Created {len(chunks)} chunks for {section} → {subsection or 'FULL'}")

    for section, data in hierarchy.items():
        subsections = data.get("subsections", {})
        if subsections:
            for subsection, subdata in subsections.items():
                process_chunk(section, subsection, subdata)
        else:
            process_chunk(section, None, data)

    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(all_chunks, f)

    logging.info(f"✅ Saved {len(all_chunks)} chunks to {output_pkl_path} with sentence-based overlap={overlap_sentences}")
    return all_chunks
